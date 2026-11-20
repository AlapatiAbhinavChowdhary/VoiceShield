"""Builds step4_finetune.ipynb. CELLS is importable by the local test harness."""
import json, sys, ast

CELLS = []

def md(text):
    CELLS.append(("markdown", text.strip("\n")))

def code(text):
    CELLS.append(("code", text.strip("\n")))

md("""
# Step 4: Fine-tune the AntiDeepfake detector (phone channel + Indian languages + commercial voices)

**Inputs:** none needed. If your Step 3 notebook output is attached (right panel, **Add Input → Your Work → Notebooks**) and complete, it is reused; otherwise Cell 3 rebuilds the Step 3 clips with the same seed and speakers (about 30–45 min extra).

**What it does:**
1. Rebuilds NII AntiDeepfake wav2vec2-Large in PyTorch (trainable) and checks it gives the same scores as the ONNX version.
2. Training data (about 1,900 real + 1,900 fake):
   - Step 3 same-speaker kNN-VC (Hindi/Telugu, train speakers only)
   - Extra Kathbath real speech
   - IndicSynth XTTS/VITS/FreeVC fakes
   - Commercial TTS: Polly, Speechify and Luvvoice only
3. The **same** noise / denoise / phone-line / bandwidth effects are applied to real and fake, so "clean = fake" can't be learned.
4. Picks the best checkpoint on dev sets: held-out speakers, an unseen platform (Hume), In-the-Wild.
5. Final test on things never trained on: held-out speakers, **ElevenLabs + Kokoro** (unseen), In-the-Wild. Compared with zero-shot.

**Outputs:** `antideepfake_ft.pt` (fine-tuned weights), `threshold.json`, `step4_results.csv`, `step3_data.zip`.

**Runs on Kaggle or Google Colab (same file):**
- **Kaggle:** GPU T4 x2, Internet ON, secret `HF_TOKEN`. **Save Version → Save & Run All (Commit)**. Outputs in the version's Output tab.
- **Colab:** Runtime → Change runtime type → **T4 GPU**. Key icon (left) → add secret `HF_TOKEN`, turn **Notebook access** on. **Runtime → Run all**, allow Google Drive when asked, keep the tab open. Outputs in **Google Drive → MyDrive/SIH26104/step4**.

About 1.5–2.5 h the first time. A rerun reuses the saved `step3_data.zip` and is about 40 min shorter.
""")

code("""
# CELL 1: installs
!pip uninstall -y -q onnxruntime
!pip install -q onnxruntime-gpu noisereduce datasets
""")

code('''
# CELL 2: setup + audio helpers
import os, re, glob, json, zipfile, tempfile, random, time, shutil
import numpy as np, pandas as pd, librosa, soundfile as sf, torch
import torch.nn as nn, torch.nn.functional as F
import pyarrow.parquet as pq
import noisereduce as nr
from scipy.signal import butter, sosfilt, resample_poly
from huggingface_hub import login, HfApi, hf_hub_download, snapshot_download
from datasets import load_dataset, Audio
from sklearn.metrics import roc_auc_score, roc_curve

try:                                   # Kaggle: Add-ons -> Secrets -> HF_TOKEN
    from kaggle_secrets import UserSecretsClient
    HF_TOKEN, ON_COLAB = UserSecretsClient().get_secret("HF_TOKEN"), False
except ImportError:                    # Google Colab: key icon on the left -> HF_TOKEN, notebook access ON
    from google.colab import userdata, drive
    HF_TOKEN, ON_COLAB = userdata.get("HF_TOKEN"), True
login(token=HF_TOKEN)
api = HfApi()
random.seed(0); np.random.seed(0); torch.manual_seed(0)

SR, T   = 16000, 64000                 # 4.0 s window, same as the ONNX model
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
if ON_COLAB:                           # results go to Google Drive, so a disconnect cannot lose them
    drive.mount("/content/drive")
    WORK, TMP = "/content/drive/MyDrive/SIH26104/step4", "/content/tmp"
else:                                  # WORK = saved Output (model, CSVs, JSON); TMP = scratch audio, not saved
    WORK, TMP = "/kaggle/working", "/kaggle/temp"
os.makedirs(WORK, exist_ok=True)
try:
    os.makedirs(TMP, exist_ok=True)
except OSError:
    TMP = tempfile.gettempdir()
N_EXTRA = 300                          # extra Kathbath real AND IndicSynth fake clips per language

def decode(audio_bytes, name_hint=""):
    ext = os.path.splitext(name_hint or "")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
        f.write(audio_bytes)
    try:
        wav, _ = librosa.load(f.name, sr=SR, mono=True)
    finally:
        os.remove(f.name)
    return wav.astype(np.float32)

def load_wav(path):
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(1)
    if sr != SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    return wav

def trim(wav):
    t, _ = librosa.effects.trim(wav, top_db=30)
    return t if len(t) > SR // 2 else wav

_PHONE_SOS = butter(4, [300, 3400], btype="band", fs=SR, output="sos")

def phone(wav):
    """Simulated phone line: 300-3400 Hz band, 8 kHz, G.711 mu-law, back to 16 kHz."""
    x = sosfilt(_PHONE_SOS, wav)
    x = resample_poly(x, 1, 2)
    x = x / (np.abs(x).max() + 1e-9) * 0.9
    mu = 255.0
    y = np.round(np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu) * 127) / 127
    x = np.sign(y) * ((1 + mu) ** np.abs(y) - 1) / mu
    return resample_poly(x, 2, 1).astype(np.float32)

def colored_noise(n, beta):
    """beta 0 = white, 1 = pink, 2 = brown."""
    spec = np.fft.rfft(np.random.randn(n))
    f = np.fft.rfftfreq(n)
    f[0] = f[1]
    x = np.fft.irfft(spec / f ** (beta / 2), n)
    return x / (x.std() + 1e-9)

def add_noise(wav, snr_db):
    p = np.mean(wav ** 2) + 1e-12
    return wav + colored_noise(len(wav), random.choice([0, 1, 2])) * np.sqrt(p / 10 ** (snr_db / 10))

def lowpass(wav, cutoff):
    return sosfilt(butter(6, cutoff, btype="low", fs=SR, output="sos"), wav)

def crop(wav, random_start=False):
    if len(wav) < T:
        wav = np.tile(wav, T // max(len(wav), 1) + 1)
    s = random.randint(0, len(wav) - T) if random_start else 0
    return wav[s:s + T]

def augment(wav):
    """Identical channel effects for real and fake, so noise/bandwidth cannot reveal the label."""
    x = crop(wav, random_start=True).astype(np.float64)
    if random.random() < 0.25:                                  # denoise (makes some REAL clips clean too)
        x = nr.reduce_noise(y=x, sr=SR, stationary=True, prop_decrease=random.uniform(0.6, 1.0))
    if random.random() < 0.5:                                   # background noise (makes some FAKE clips noisy too)
        x = add_noise(x, random.uniform(5, 30))
    r = random.random()
    if r < 0.4:
        x = phone(x)                                            # phone line
    elif r < 0.6:
        x = lowpass(x, random.uniform(3500, 7000))              # reduced bandwidth
    return np.asarray(crop(x), dtype=np.float32)

def eer_auc(labels, p_fake):
    labels, p_fake = np.asarray(labels), np.asarray(p_fake)
    fpr, tpr, thr = roc_curve(labels, p_fake)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return {"EER_%": 100 * (fpr[i] + fnr[i]) / 2, "AUC": roc_auc_score(labels, p_fake),
            "threshold_at_EER": float(thr[i]),
            "acc_at_0.5_%": 100 * ((p_fake >= 0.5).astype(int) == labels).mean()}

def acc_at(labels, p_fake, thr):
    return 100 * ((np.asarray(p_fake) >= thr).astype(int) == np.asarray(labels)).mean()

print("Device:", torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU - check settings")
''')

code('''
# CELL 3: Step 3 data. Uses an attached Step 3 output if it is complete, otherwise rebuilds it here.
#         Rebuilding runs the Step 3 code with the same seed (same shards, speakers and splits): ~30-45 min.
INPUT  = "/kaggle/input"
S3_DIR = f"{TMP}/step3_data"
N_SHARDS, N_SPK, N_REF, N_PER = 6, 20, 40, 25          # same settings as Step 3

def remap(df, root):
    df = df.copy()
    df["path"] = [root + p[p.index("/step3_data") + len("/step3_data"):] for p in df.path]
    return df

def kathbath(lang, n_shards=N_SHARDS):
    files = sorted(f.path for f in api.list_repo_tree("ai4bharat/Kathbath", path_in_repo=lang,
                                                      repo_type="dataset", recursive=True)
                   if f.path.endswith(".parquet") and "train" in os.path.basename(f.path))
    pick = sorted({files[int(i)] for i in np.linspace(0, len(files) - 1, min(n_shards, len(files)))})
    print(f"{lang}: {len(files)} train shards, downloading {len(pick)}")
    return pd.concat([pq.read_table(hf_hub_download("ai4bharat/Kathbath", p, repo_type="dataset")).to_pandas()
                      for p in pick], ignore_index=True)

def rebuild_step3():
    """The Step 3 notebook's code: same-speaker kNN-VC clones of Kathbath speakers."""
    knn_vc = torch.hub.load("bshall/knn-vc", "knn_vc", prematched=True, trust_repo=True,
                            pretrained=True, device=DEVICE)
    random.seed(0)
    rows = []
    for lang in ["hindi", "telugu"]:
        df = kathbath(lang)
        counts = df.speaker_id.value_counts()
        ok = counts[counts >= N_REF + N_PER].index.tolist()
        gender = df.drop_duplicates("speaker_id").set_index("speaker_id")["gender"].astype(str).str.lower()
        fem = [s for s in ok if gender[s].startswith("f")]
        mal = [s for s in ok if not gender[s].startswith("f")]
        random.shuffle(fem); random.shuffle(mal)
        targets = fem[:N_SPK // 2] + mal[:N_SPK // 2]
        targets += [s for s in ok if s not in targets][:N_SPK - len(targets)]
        print(f"{lang}: {df.speaker_id.nunique()} speakers in shards, {len(ok)} with enough clips, using {len(targets)}")
        order = targets[:]; random.shuffle(order)
        n_tr, n_dev = int(0.7 * len(order)), int(0.15 * len(order))
        split = {s: ("train" if i < n_tr else "dev" if i < n_tr + n_dev else "test") for i, s in enumerate(order)}

        for k, spk in enumerate(targets):
            mine = df[df.speaker_id == spk].sample(frac=1, random_state=k)
            ref_rows, real_rows = mine.iloc[:N_REF], mine.iloc[N_REF:N_REF + N_PER]
            src_rows = df[df.speaker_id != spk].sample(N_PER, random_state=k)
            refs = [torch.from_numpy(trim(decode(a["bytes"], a.get("path")))) for a in ref_rows["audio_filepath"]]
            matching_set = knn_vc.get_matching_set(refs, vad_trigger_level=0)
            sdir = str(spk).replace("/", "_")
            for j, (_, r) in enumerate(real_rows.iterrows()):
                wav = trim(decode(r["audio_filepath"]["bytes"], r["audio_filepath"].get("path")))
                p = f"{S3_DIR}/{lang}/real/{sdir}/{j:03d}.flac"
                os.makedirs(os.path.dirname(p), exist_ok=True); sf.write(p, wav, SR)
                rows.append(dict(path=p, label=0, lang=lang, speaker=spk, gender=gender[spk],
                                 generator="bonafide", source_speaker=spk, split=split[spk], text=r["text"]))
            for j, (_, r) in enumerate(src_rows.iterrows()):
                src = torch.from_numpy(trim(decode(r["audio_filepath"]["bytes"], r["audio_filepath"].get("path"))))
                with torch.inference_mode():
                    out = knn_vc.match(knn_vc.get_features(src), matching_set, topk=4)
                wav = trim(out.detach().cpu().numpy().astype(np.float32))
                p = f"{S3_DIR}/{lang}/fake/{sdir}/{j:03d}.flac"
                os.makedirs(os.path.dirname(p), exist_ok=True); sf.write(p, wav, SR)
                rows.append(dict(path=p, label=1, lang=lang, speaker=spk, gender=gender[spk],
                                 generator="knnvc", source_speaker=r["speaker_id"], split=split[spk], text=r["text"]))
            print(f"  [{lang}] speaker {k + 1}/{len(targets)} ({split[spk]}) done")
        del df
    del knn_vc
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    meta = pd.DataFrame(rows)
    meta.to_csv(f"{S3_DIR}/metadata.csv", index=False)
    return meta

s3 = None
# 1) Step 3 notebook output attached as Kaggle input (only if every audio file made it into the output)
for meta_csv in sorted(glob.glob(f"{INPUT}/**/step3_data/metadata.csv", recursive=True)):
    cand = remap(pd.read_csv(meta_csv), os.path.dirname(meta_csv))
    n_ok = sum(os.path.exists(p) for p in cand.path)
    print(f"Attached Step 3 output: {meta_csv} -> {n_ok}/{len(cand)} audio files present")
    if n_ok == len(cand):
        s3 = cand
        break
# 2) step3_data.zip saved by an earlier run of THIS notebook (Google Drive on Colab, or attached on Kaggle)
if s3 is None:
    for zp in sorted(glob.glob(f"{INPUT}/**/step3_data.zip", recursive=True)) + glob.glob(f"{WORK}/step3_data.zip"):
        try:
            dest = f"{TMP}/step3_from_zip"
            shutil.rmtree(dest, ignore_errors=True)
            with zipfile.ZipFile(zp) as z:
                z.extractall(dest)
            cand = remap(pd.read_csv(f"{dest}/metadata.csv"), dest)
            if all(os.path.exists(p) for p in cand.path):
                print("Reusing saved Step 3 clips:", zp)
                s3 = cand
                break
        except Exception as e:
            print("Could not use", zp, "->", e)
# 3) otherwise rebuild, then save ONE zip file so a rerun can skip this
if s3 is None:
    print("Step 3 audio not attached (or incomplete), so rebuilding it here. About 30-45 min...")
    s3 = rebuild_step3()
    shutil.make_archive(f"{TMP}/step3_data", "zip", S3_DIR)
    shutil.copy(f"{TMP}/step3_data.zip", f"{WORK}/step3_data.zip")
    print("Saved", f"{WORK}/step3_data.zip", "(a rerun will reuse it)")

s3["source"] = "step3_kathbath_knnvc"
HELD_OUT = set(s3.loc[s3.split != "train", "speaker"].astype(str))   # dev/test speakers: never train on them
print(s3.groupby(["lang", "split", "label"]).size().unstack())
''')

code('''
# CELL 4: rebuild AntiDeepfake in PyTorch + check it matches the ONNX model
from transformers import Wav2Vec2Config, Wav2Vec2Model
from safetensors.torch import load_file

class Detector(nn.Module):
    """wav2vec 2.0 Large -> mean over time -> Linear(1024, 2). Output index 0 = FAKE, 1 = REAL."""
    def __init__(self, stable=True):
        super().__init__()
        cfg = Wav2Vec2Config(
            hidden_size=1024, num_hidden_layers=24, num_attention_heads=16, intermediate_size=4096,
            feat_extract_norm="layer", conv_bias=True, do_stable_layer_norm=stable,
            feat_extract_activation="gelu", hidden_act="gelu",
            num_conv_pos_embeddings=128, num_conv_pos_embedding_groups=16,
            hidden_dropout=0.1, attention_dropout=0.1, activation_dropout=0.0,
            feat_proj_dropout=0.1, layerdrop=0.0, apply_spec_augment=False,
            mask_time_prob=0.0, mask_feature_prob=0.0)
        self.ssl = Wav2Vec2Model(cfg)
        self.head = nn.Linear(1024, 2)

    def forward(self, wav):                                  # wav: (B, samples), 16 kHz
        wav = F.layer_norm(wav, wav.shape[-1:])              # per-utterance normalisation
        return self.head(self.ssl(wav).last_hidden_state.mean(1))

    @torch.no_grad()
    def p_fake(self, wav):
        return self.forward(wav).float().softmax(-1)[:, 0]

def fairseq_to_hf(sd, hf_keys):
    """Rename NII/fairseq wav2vec2 weights to Hugging Face Wav2Vec2Model names."""
    out = {}
    for k, v in sd.items():
        if not k.startswith("m_ssl.model."):
            continue
        k = k[len("m_ssl.model."):]
        if k.startswith(("quantizer.", "project_q.", "final_proj.", "mask_emb")):
            continue                                         # pre-training-only parts
        if k.startswith("post_extract_proj."):
            k = "feature_projection.projection." + k[len("post_extract_proj."):]
        elif k.startswith("layer_norm."):
            k = "feature_projection." + k
        k = re.sub(r"conv_layers\\.(\\d+)\\.0\\.", r"conv_layers.\\1.conv.", k)
        k = re.sub(r"conv_layers\\.(\\d+)\\.2\\.1\\.", r"conv_layers.\\1.layer_norm.", k)
        k = k.replace("encoder.pos_conv.0.", "encoder.pos_conv_embed.conv.")
        k = k.replace(".self_attn_layer_norm.", ".layer_norm.").replace(".self_attn.", ".attention.")
        k = k.replace(".fc1.", ".feed_forward.intermediate_dense.").replace(".fc2.", ".feed_forward.output_dense.")
        if k not in hf_keys:                                 # newer PyTorch weight-norm naming
            k = (k.replace("conv.weight_g", "conv.parametrizations.weight.original0")
                  .replace("conv.weight_v", "conv.parametrizations.weight.original1"))
        out[k] = v
    return out

def load_antideepfake(sd, stable=True):
    m = Detector(stable)
    m.ssl.load_state_dict(fairseq_to_hf(sd, set(m.ssl.state_dict())), strict=True)
    m.head.weight.data.copy_(sd["proj_fc.weight"])
    m.head.bias.data.copy_(sd["proj_fc.bias"])
    return m.to(DEVICE).eval()

NII_SD = load_file(hf_hub_download("nii-yamagishilab/wav2vec-large-anti-deepfake", "model.safetensors"))

# Parity check against the ONNX model used in Steps 2-3
import onnxruntime as ort
if hasattr(ort, "preload_dlls"):
    ort.preload_dlls()
sess = ort.InferenceSession(hf_hub_download("SpeechAntiSpoofingBenchmarks/Wav2Vec2-Large-AntiDeepfake",
                                            "wav2vec2-large-antideepfake.onnx"),
                            providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
probe = s3[s3.split == "dev"].groupby("label").head(8)
x = np.stack([crop(load_wav(p)) for p in probe.path]).astype(np.float32)
xn = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-7)
onnx_logits = sess.run(None, {sess.get_inputs()[0].name: xn})[0]

for stable in (True, False):
    model = load_antideepfake(NII_SD, stable)
    with torch.no_grad():
        pt_logits = model(torch.from_numpy(x).to(DEVICE)).float().cpu().numpy()
    diff = float(np.abs(pt_logits - onnx_logits).max())
    soft = lambda l: np.exp(l[:, 0]) / np.exp(l).sum(1)
    print(f"stable_layer_norm={stable}: max logit diff vs ONNX = {diff:.4f}, "
          f"max p_fake diff = {np.abs(soft(pt_logits) - soft(onnx_logits)).max():.4f}")
    if diff < 0.5:                       # a wrong weight mapping gives differences of several units
        break
    del model; torch.cuda.empty_cache()
else:
    raise SystemExit("PyTorch model does not match ONNX - send these numbers to your mentor/Claude.")
del sess
print("PyTorch model matches ONNX. Parameters:", sum(p.numel() for p in model.parameters()) // 1_000_000, "M")
''')

code('''
# CELL 5: build train / dev / test data
os.makedirs(f"{TMP}/step4_data", exist_ok=True)
extra = []

# 5a. Extra REAL Kathbath speech (train shards Step 3 did not use; dev/test speakers excluded)
for lang in ["hindi", "telugu"]:
    files = sorted(f.path for f in api.list_repo_tree("ai4bharat/Kathbath", path_in_repo=lang,
                                                      repo_type="dataset", recursive=True)
                   if f.path.endswith(".parquet") and "train" in os.path.basename(f.path))
    used = {files[int(i)] for i in np.linspace(0, len(files) - 1, min(6, len(files)))}
    pick = [f for f in files if f not in used][::max(1, (len(files) - len(used)) // 3)][:3]
    df = pd.concat([pq.read_table(hf_hub_download("ai4bharat/Kathbath", p, repo_type="dataset")).to_pandas()
                    for p in pick], ignore_index=True)
    df = df[~df.speaker_id.astype(str).isin(HELD_OUT)]
    df = df.sample(frac=1, random_state=1).groupby("speaker_id").head(15).head(N_EXTRA)
    for j, r in enumerate(df.itertuples()):
        p = f"{TMP}/step4_data/{lang}/kathbath_real/{j:04d}.flac"
        os.makedirs(os.path.dirname(p), exist_ok=True)
        sf.write(p, trim(decode(r.audio_filepath["bytes"], r.audio_filepath.get("path"))), SR)
        extra.append(dict(path=p, label=0, lang=lang, speaker=str(r.speaker_id), split="train",
                          source="kathbath_extra", generator="bonafide"))
    print(f"{lang}: {len(df)} extra real clips from {df.speaker_id.nunique()} speakers")
    del df

# 5b. IndicSynth FAKE speech (XTTS-v2 / VITS / FreeVC clones of Kathbath speakers)
for lang, cfg in [("hindi", "Hindi"), ("telugu", "Telugu")]:
    ds = load_dataset("vdivyasharma/IndicSynth", name=cfg, split="train", streaming=True)
    acol = next((k for k, v in (ds.features or {}).items() if isinstance(v, Audio)), "audio")
    ds = ds.cast_column(acol, Audio(decode=False)).shuffle(seed=1, buffer_size=2000)
    n, gens = 0, {}
    for row in ds:
        tgt = str(row.get("Target Speaker ID"))
        if tgt in HELD_OUT:
            continue
        p = f"{TMP}/step4_data/{lang}/indicsynth_fake/{n:04d}.flac"
        os.makedirs(os.path.dirname(p), exist_ok=True)
        sf.write(p, trim(decode(row[acol]["bytes"], row[acol].get("path"))), SR)
        g = row.get("Generative Model", "indicsynth")
        gens[g] = gens.get(g, 0) + 1
        extra.append(dict(path=p, label=1, lang=lang, speaker=f"isyn_{tgt}", split="train",
                          source="indicsynth", generator=g))
        n += 1
        if n >= N_EXTRA:
            break
    print(f"{lang}: {n} IndicSynth fakes, generators: {gens}")

# 5c. Commercial TTS, split so dev/test platforms and real recordings are never trained on
#     fake prefixes: po=Polly sp=Speechify lv=Luvvoice | hu=Hume (dev) | el=ElevenLabs hg=Kokoro (test)
GS = snapshot_download("garystafford/deepfake-audio-detection", repo_type="dataset",
                       allow_patterns=["fake/*", "real/*"])
comm = []
for p in sorted(glob.glob(f"{GS}/fake/*.flac")) + sorted(glob.glob(f"{GS}/real/*.flac")):
    name = os.path.basename(p)
    if "/fake/" in p.replace("\\\\", "/"):
        plat, label = name.split("_")[0], 1
        split = "train" if plat in ("po", "sp", "lv") else "dev" if plat == "hu" else "test"
    else:
        plat, label = name[:7], 0
        split = ("test" if plat in ("yt_0009", "yt_0010", "yt_0002") else
                 "dev" if plat in ("yt_0013", "yt_0001", "yt_0003", "yt_0007") else "train")
    comm.append(dict(path=p, label=label, lang="english", speaker=plat, split=split,
                     source="commercial_tts", generator=plat if label else "bonafide"))
comm = pd.DataFrame(comm)
print(comm.groupby(["split", "label"]).size().unstack())

train_df = pd.concat([s3[s3.split == "train"], pd.DataFrame(extra), comm[comm.split == "train"]],
                     ignore_index=True)
train_df.to_csv(f"{WORK}/step4_train.csv", index=False)
print("\\nTRAINING SET:"); print(train_df.groupby(["source", "label"]).size().unstack(fill_value=0))

# 5d. In-the-Wild: dev sample (checkpoint selection) and a separate test sample
zp = hf_hub_download("mueller91/In-The-Wild", "release_in_the_wild.zip", repo_type="dataset")
zf = zipfile.ZipFile(zp)
meta_name = next(n for n in zf.namelist() if n.endswith("meta.csv"))
root = os.path.dirname(meta_name)
itw = pd.read_csv(zf.open(meta_name))
itw["y"] = itw["label"].astype(str).str.lower().str.contains("spoof").astype(int)
itw_dev = itw.groupby("y").sample(n=200, random_state=1)
itw_test = itw.drop(itw_dev.index).groupby("y").sample(n=300, random_state=2)
def itw_set(d):
    return ([trim(decode(zf.read(f"{root}/{f}" if root else f), f)) for f in d.file], d.y.tolist())

def disk_set(d):
    return [trim(load_wav(p)) for p in d.path], d.label.tolist()

dev3, test3 = disk_set(s3[s3.split == "dev"]), disk_set(s3[s3.split == "test"])
comm_test = disk_set(comm[comm.split == "test"])
DEV = {
    "Hindi/Telugu held-out speakers, clean": dev3,
    "Hindi/Telugu held-out speakers, phone": ([phone(w) for w in dev3[0]], dev3[1]),
    "Commercial, unseen Hume": disk_set(comm[comm.split == "dev"]),
    "In-the-Wild (dev sample)": itw_set(itw_dev),
}
TEST = {
    "Hindi/Telugu test speakers, clean": test3,
    "Hindi/Telugu test speakers, phone": ([phone(w) for w in test3[0]], test3[1]),
    "Commercial, unseen ElevenLabs+Kokoro": comm_test,
    "Commercial, unseen ElevenLabs+Kokoro, phone": ([phone(w) for w in comm_test[0]], comm_test[1]),
    "In-the-Wild (test sample)": itw_set(itw_test),
}
for name, (w, y) in {**DEV, **TEST}.items():
    print(f"{name}: {y.count(0)} real, {y.count(1)} fake")
''')

code('''
# CELL 6: zero-shot scores (before training)
@torch.no_grad()
def score(model, wavs, bs=16):
    model.eval()
    out = []
    for i in range(0, len(wavs), bs):
        x = torch.from_numpy(np.stack([crop(w) for w in wavs[i:i + bs]]).astype(np.float32)).to(DEVICE)
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
            out.append(model.p_fake(x).float().cpu().numpy())
    return np.concatenate(out)

def evaluate(model, sets):
    scores = {k: score(model, w) for k, (w, y) in sets.items()}
    eers = {k: eer_auc(sets[k][1], s)["EER_%"] for k, s in scores.items()}
    return scores, eers

def pooled_threshold(sets, scores):
    y = np.concatenate([sets[k][1] for k in sets])
    s = np.concatenate([scores[k] for k in sets])
    return eer_auc(y, s)["threshold_at_EER"]

zs_dev_scores, zs_dev_eer = evaluate(model, DEV)
zs_metric = float(np.mean(list(zs_dev_eer.values())))
zs_thr = pooled_threshold(DEV, zs_dev_scores)
zs_test_scores, _ = evaluate(model, TEST)
for k, v in zs_dev_eer.items():
    print(f"zero-shot dev EER {v:6.2f}%  {k}")
print(f"zero-shot dev mean EER: {zs_metric:.2f}% | calibrated threshold: {zs_thr:.4f}")
''')

code('''
# CELL 7: fine-tune (keeps the checkpoint with the lowest mean dev EER; zero-shot counts as step 0)
BS, ACCUM, EPOCHS = 8, 2, 3
LR_BACKBONE, LR_HEAD, WARMUP = 5e-6, 5e-5, 50
EVAL_EVERY, PATIENCE = 100, 4
BEST = f"{WORK}/antideepfake_ft.pt"

class TrainSet(torch.utils.data.Dataset):
    def __init__(self, df):
        self.paths, self.labels = df.path.tolist(), df.label.tolist()
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, i):
        x = augment(trim(load_wav(self.paths[i])))
        return torch.from_numpy(x), 1 - self.labels[i]          # model target: 0 = fake, 1 = real

def _seed_worker(_):
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s); random.seed(s)

loader = torch.utils.data.DataLoader(TrainSet(train_df), batch_size=BS, shuffle=True, drop_last=True,
                                     num_workers=min(4, os.cpu_count() or 2), worker_init_fn=_seed_worker,
                                     persistent_workers=True)
if hasattr(model.ssl, "freeze_feature_encoder"):                 # keep the CNN front-end frozen
    model.ssl.freeze_feature_encoder()
for p in model.ssl.feature_extractor.parameters():
    p.requires_grad = False
backbone = [p for p in model.ssl.parameters() if p.requires_grad]
opt = torch.optim.AdamW([{"params": backbone, "lr": LR_BACKBONE},
                         {"params": model.head.parameters(), "lr": LR_HEAD}], weight_decay=1e-4)
total = EPOCHS * len(loader) // ACCUM
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / WARMUP) * max(0.05, 1 - s / total))
scaler = torch.amp.GradScaler("cuda", enabled=DEVICE == "cuda")

def save(m):
    torch.save({k: v.half() for k, v in m.state_dict().items()}, BEST)

save(model)
best, best_step, bad, step, history = zs_metric, 0, 0, 0, [{"step": 0, "dev_mean_EER": zs_metric, **zs_dev_eer}]
t0, run_loss, stop = time.time(), [], False
print(f"{len(train_df)} training clips, {len(loader)} batches/epoch, {total} optimizer steps max")
for epoch in range(EPOCHS):
    for i, (x, y) in enumerate(loader):
        model.train()
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
            loss = F.cross_entropy(model(x.to(DEVICE)), y.to(DEVICE)) / ACCUM
        scaler.scale(loss).backward()
        run_loss.append(loss.item() * ACCUM)
        if (i + 1) % ACCUM:
            continue
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
        step += 1
        if step % 25 == 0:
            print(f"epoch {epoch + 1} step {step}/{total} loss {np.mean(run_loss):.4f} "
                  f"({(time.time() - t0) / 60:.1f} min)")
            run_loss = []
        if step % EVAL_EVERY == 0:
            _, dev_eer = evaluate(model, DEV)
            metric = float(np.mean(list(dev_eer.values())))
            history.append({"step": step, "dev_mean_EER": metric, **dev_eer})
            print(f"  >> dev mean EER {metric:.2f}% (best {best:.2f}% at step {best_step})",
                  {k: round(float(v), 2) for k, v in dev_eer.items()})
            if metric < best:
                best, best_step, bad = metric, step, 0
                save(model)
                print("  >> saved new best")
            else:
                bad += 1
                if bad >= PATIENCE:
                    stop = True
                    break
    if stop:
        print("Early stopping: dev EER stopped improving.")
        break
pd.DataFrame(history).to_csv(f"{WORK}/step4_history.csv", index=False)
print(f"Done in {(time.time() - t0) / 60:.1f} min. Best dev mean EER {best:.2f}% at step {best_step} "
      f"(zero-shot was {zs_metric:.2f}%)")
''')

code('''
# CELL 8: final test - zero-shot vs fine-tuned (all test sets are unseen in training)
model.load_state_dict({k: v.float() for k, v in torch.load(BEST, map_location=DEVICE).items()})
model.eval()
ft_dev_scores, _ = evaluate(model, DEV)
ft_thr = pooled_threshold(DEV, ft_dev_scores)
ft_test_scores, _ = evaluate(model, TEST)

rows = []
for name, (w, y) in TEST.items():
    for tag, sc, thr in [("zero-shot", zs_test_scores, zs_thr), ("fine-tuned", ft_test_scores, ft_thr)]:
        r = eer_auc(y, sc[name])
        rows.append({"test set": name, "model": tag, "EER_%": r["EER_%"], "AUC": r["AUC"],
                     "acc_at_calibrated_thr_%": acc_at(y, sc[name], thr), "acc_at_0.5_%": r["acc_at_0.5_%"]})
res = pd.DataFrame(rows).round(3)
res.to_csv(f"{WORK}/step4_results.csv", index=False)
print(res.to_string(index=False))
print(f"\\nCalibrated thresholds (from dev): zero-shot {zs_thr:.4f} | fine-tuned {ft_thr:.4f}")
print("Best checkpoint step:", best_step, "(0 means fine-tuning did not beat zero-shot)")

json.dump({"threshold_p_fake": ft_thr, "zero_shot_threshold": zs_thr, "best_step": best_step,
           "window_samples": T, "sample_rate": SR, "output_index_fake": 0},
          open(f"{WORK}/threshold.json", "w"), indent=2)

# Latency of one 4-second window (what the real-time system will pay per update)
x1 = torch.from_numpy(crop(TEST["In-the-Wild (test sample)"][0][0])[None].astype(np.float32)).to(DEVICE)
for _ in range(5):
    score(model, [x1[0].cpu().numpy()])
if DEVICE == "cuda":
    torch.cuda.synchronize()
t1 = time.time()
for _ in range(30):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
        model.p_fake(x1)
if DEVICE == "cuda":
    torch.cuda.synchronize()
print(f"Latency per 4 s window on {DEVICE}: {(time.time() - t1) / 30 * 1000:.1f} ms")
''')


def build(out):
    for i, (kind, src) in enumerate(CELLS):
        if kind == "code":
            ast.parse("\n".join(l for l in src.splitlines() if not l.lstrip().startswith("!")), f"cell{i}")
    nb = {"cells": [], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                   "name": "python3"}, "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    for kind, src in CELLS:
        c = {"cell_type": kind, "metadata": {}, "source": [l + "\n" for l in src.splitlines()]}
        if kind == "code":
            c.update(execution_count=None, outputs=[])
        nb["cells"].append(c)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("OK: wrote", out, "with", len(CELLS), "cells; all code cells parse.")


if __name__ == "__main__":
    build(sys.argv[1])
