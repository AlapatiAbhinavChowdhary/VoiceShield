"""Builds step8_dsp_lora.ipynb: DSP-aware parameter-efficient fine-tuning of the AntiDeepfake detector.

Reuses the tested Step 4 cells (setup, Step 3 data, weight loading + ONNX parity check) and replaces
the data, training and evaluation cells.  CELLS is importable by the local test harness.

    python build_step8.py ../step8_dsp_lora.ipynb
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_step4 as b4  # noqa: E402

S4 = [src for kind, src in b4.CELLS if kind == "code"]      # S4[0] = Cell 1 ... S4[7] = Cell 8
CELLS = []


def md(text):
    CELLS.append(("markdown", text.strip("\n")))


def code(text):
    CELLS.append(("code", text.strip("\n")))


md("""
# Step 8: DSP-aware LoRA fine-tuning: fixing false alarms caused by consumer audio processing

**Problem we measured (Steps 5b–6c):** the pre-trained detector mistakes *audio processing* for an AI voice.
On our laptop, Windows noise-suppression moved a real voice from llr -10 (clearly real) to -2 and caused a false HOLD.
Phones, Bluetooth headsets (CVSD / mSBC codecs), WebRTC (Opus) and operating systems all process speech like this.

**Our method:**
1. **LoRA adapters** (~1% of the weights) on every attention and feed-forward layer; the 315 M backbone stays frozen.
   Step 4's full fine-tuning over-fitted and forgot; LoRA keeps the original knowledge.
2. **Consumer-DSP augmentation, applied to real AND fake clips alike**: noise suppression (spectral gating,
   spectral subtraction), auto-gain control, G.711 phone line, Bluetooth CVSD (8 kHz) and mSBC-like (16 kHz),
   Opus at 8-24 kbit/s. Symmetric, so processing can never become a "real" or "fake" cue.
3. **New test sets per DSP effect** plus the false-alarm rate on real speech, compared with the original model.
   The original model counts as step 0: if training does not beat it, nothing changes.

**Inputs:** Internet ON, GPU T4 x2, secret `HF_TOKEN`. Optional: attach your Step 4 notebook (Add Input -> Your Work)
so its `step3_data.zip` is reused (saves ~40 min).
**Outputs:** `step8_results.csv`, `step8_history.csv`, `voiceshield_lora_adapter/` (our trained weights, ~15 MB),
`voiceshield_dsp_lora.onnx` (merged model for the live system), `step8_threshold.json`.
""")

# ---------------------------------------------------------------- Cell 1: installs
code("""
# CELL 1: installs (torchao is removed: Kaggle ships an old torchao that new peft versions refuse to load)
!pip uninstall -y -q onnxruntime torchao
!pip install -q onnxruntime-gpu noisereduce datasets peft onnx
!ffmpeg -version | head -n 1
""")

# ---------------------------------------------------------------- Cell 2: Step 4 setup + DSP effects
code(S4[1] + '''

# ======================= Step 8: consumer-DSP effects (what devices do to a REAL voice) =======================
import shutil as _shutil, subprocess as _sp
from scipy.signal import stft, istft
HAVE_FFMPEG = _shutil.which("ffmpeg") is not None
_BT_SOS = butter(4, [300, 3400], btype="band", fs=SR, output="sos")

def _fit(y, n):
    y = np.asarray(y, np.float64)
    return y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))

def agc(x, target_db=-23.0, max_db=20.0):
    """Automatic gain control: fast attack, slow release, soft limiter (OS / phone style)."""
    x = np.asarray(x, np.float64)
    n = 160; fr = len(x) // n
    if fr < 2:
        return x
    rms = np.sqrt(np.mean(x[:fr * n].reshape(fr, n) ** 2, axis=1) + 1e-10)
    want = np.clip(target_db - 20 * np.log10(rms), -max_db, max_db)
    g = np.empty(fr); cur = want[0]
    for i in range(fr):
        cur += (0.5 if want[i] < cur else 0.05) * (want[i] - cur)
        g[i] = cur
    y = x * np.interp(np.arange(len(x)), np.arange(fr) * n + n / 2, 10 ** (g / 20))
    return np.tanh(1.5 * y) / 1.5

def ns_gate(x):
    """Non-stationary spectral-gating noise suppression (leaves 'musical noise', like many suppressors)."""
    return _fit(nr.reduce_noise(y=np.asarray(x, np.float64), sr=SR, stationary=False,
                                prop_decrease=random.uniform(0.7, 1.0)), len(x))

def ns_specsub(x):
    """Spectral subtraction with over-subtraction, a gain floor and temporal smoothing (classic phone NS)."""
    x = np.asarray(x, np.float64)
    _, _, Z = stft(x, fs=SR, nperseg=512, noverlap=384)
    P = np.abs(Z) ** 2
    noise = np.percentile(P, 10, axis=1, keepdims=True)
    alpha, floor = random.uniform(1.5, 4.0), random.uniform(0.03, 0.15)
    G = np.sqrt(np.maximum(1 - alpha * noise / (P + 1e-12), floor ** 2))
    for t in range(1, G.shape[1]):
        G[:, t] = 0.6 * G[:, t] + 0.4 * G[:, t - 1]
    _, y = istft(Z * G, fs=SR, nperseg=512, noverlap=384)
    return _fit(y, len(x))

def bt_cvsd(x):
    """Bluetooth hands-free (HFP) narrowband voice: 8 kHz audio, CVSD delta modulation at 64 kbit/s."""
    x = np.asarray(x, np.float64)
    x8 = resample_poly(x, 1, 2)
    x8 = x8 / (np.abs(x8).max() + 1e-9) * 0.8
    u = resample_poly(x8, 8, 1)
    out = np.empty_like(u)
    est, step, hist = 0.0, 0.01, 0
    for i, v in enumerate(u):
        bit = 1 if v >= est else 0
        hist = ((hist << 1) | bit) & 15
        step = min(step + 0.01, 0.25) if hist in (0, 15) else max(step * 0.98, 0.002)
        est = 0.999 * est + (step if bit else -step)
        out[i] = est
    y = sosfilt(_BT_SOS, resample_poly(resample_poly(out, 1, 8), 2, 1))
    return _fit(y, len(x))

def bt_msbc(x):
    """Bluetooth wideband speech (mSBC-like): 16 kHz, band-limited to ~7 kHz, 8-bit companded."""
    y = lowpass(np.asarray(x, np.float64), 7000)
    y = y / (np.abs(y).max() + 1e-9) * 0.9
    q = np.round(np.sign(y) * np.log1p(255 * np.abs(y)) / np.log1p(255) * 127) / 127
    return np.sign(q) * ((1 + 255) ** np.abs(q) - 1) / 255

def opus(x, kbps=None):
    """Opus (WebRTC / WhatsApp calls) at a low bit rate, via ffmpeg; falls back to mSBC-like without ffmpeg."""
    if not HAVE_FFMPEG:
        return bt_msbc(x)
    kbps = kbps or random.choice([8, 12, 16, 24])
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
    enc = _sp.run(["ffmpeg", "-loglevel", "quiet", "-f", "s16le", "-ar", "16000", "-ac", "1", "-i", "pipe:0",
                   "-c:a", "libopus", "-b:a", f"{kbps}k", "-f", "ogg", "pipe:1"], input=pcm, capture_output=True).stdout
    dec = _sp.run(["ffmpeg", "-loglevel", "quiet", "-i", "pipe:0", "-f", "s16le", "-ar", "16000", "-ac", "1",
                   "pipe:1"], input=enc, capture_output=True).stdout
    y = np.frombuffer(dec, "<i2").astype(np.float64) / 32768
    return _fit(y, len(x)) if len(y) > 100 else np.asarray(x, np.float64)

DSP = {"noise suppression": ns_gate, "spectral subtraction": ns_specsub, "auto-gain": agc,
       "auto-gain + noise suppression": lambda x: ns_specsub(agc(x)), "phone line (G.711)": phone,
       "Bluetooth CVSD": bt_cvsd, "Bluetooth mSBC": bt_msbc, "Opus": opus}
DSP_NAMES = list(DSP)

def augment(wav):
    """Step 8 augmentation, identical for real and fake: background noise, then ONE consumer-DSP effect."""
    x = crop(wav, random_start=True).astype(np.float64)
    if random.random() < 0.5:
        x = add_noise(x, random.uniform(5, 30))
    r = random.random()
    if r < 0.6:
        x = DSP[random.choice(DSP_NAMES)](x)
    elif r < 0.7:
        x = lowpass(x, random.uniform(3500, 7000))
    return np.asarray(crop(x), dtype=np.float32)

def rates_at(labels, p_fake, thr):
    """False-alarm rate on REAL clips and miss rate on FAKE clips at a fixed threshold (both in %)."""
    labels, p_fake = np.asarray(labels), np.asarray(p_fake)
    return (100 * float((p_fake[labels == 0] >= thr).mean()), 100 * float((p_fake[labels == 1] < thr).mean()))

print("ffmpeg (Opus) available:", HAVE_FFMPEG, "| DSP effects:", ", ".join(DSP_NAMES))
''')

# ---------------------------------------------------------------- Cells 3 and 4: unchanged from Step 4
code(S4[2])
code(S4[3])

# ---------------------------------------------------------------- Cell 5: data + DSP dev/test sets
code('''
# CELL 5: build train / dev / test data (more real speech than Step 4) + DSP test sets
N_REAL_EXTRA, N_FAKE_EXTRA = 450, 450        # per language
os.makedirs(f"{TMP}/step8_data", exist_ok=True)
extra = []

# 5a. Extra REAL Kathbath speech (train shards Step 3 did not use; dev/test speakers excluded)
for lang in ["hindi", "telugu"]:
    files = sorted(f.path for f in api.list_repo_tree("ai4bharat/Kathbath", path_in_repo=lang,
                                                      repo_type="dataset", recursive=True)
                   if f.path.endswith(".parquet") and "train" in os.path.basename(f.path))
    used = {files[int(i)] for i in np.linspace(0, len(files) - 1, min(6, len(files)))}
    pick = [f for f in files if f not in used][::max(1, (len(files) - len(used)) // 4)][:4]
    df = pd.concat([pq.read_table(hf_hub_download("ai4bharat/Kathbath", p, repo_type="dataset")).to_pandas()
                    for p in pick], ignore_index=True)
    df = df[~df.speaker_id.astype(str).isin(HELD_OUT)]
    df = df.sample(frac=1, random_state=1).groupby("speaker_id").head(12).head(N_REAL_EXTRA)
    for j, r in enumerate(df.itertuples()):
        p = f"{TMP}/step8_data/{lang}/kathbath_real/{j:04d}.flac"
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
        p = f"{TMP}/step8_data/{lang}/indicsynth_fake/{n:04d}.flac"
        os.makedirs(os.path.dirname(p), exist_ok=True)
        sf.write(p, trim(decode(row[acol]["bytes"], row[acol].get("path"))), SR)
        g = row.get("Generative Model", "indicsynth")
        gens[g] = gens.get(g, 0) + 1
        extra.append(dict(path=p, label=1, lang=lang, speaker=f"isyn_{tgt}", split="train",
                          source="indicsynth", generator=g))
        n += 1
        if n >= N_FAKE_EXTRA:
            break
    print(f"{lang}: {n} IndicSynth fakes, generators: {gens}")

# 5c. Commercial TTS with the Step 4 split (Hume = dev, ElevenLabs + Kokoro = test, never trained on)
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

train_df = pd.concat([s3[s3.split == "train"], pd.DataFrame(extra), comm[comm.split == "train"]],
                     ignore_index=True)
train_df.to_csv(f"{WORK}/step8_train.csv", index=False)
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

def dsp_set(base, effect, seed):
    """Same clips with background noise (15 dB) and ONE effect, deterministic (seeded)."""
    random.seed(seed); np.random.seed(seed)
    w, y = base
    return [np.asarray(DSP[effect](add_noise(np.asarray(x, np.float64), 15.0)), np.float32) for x in w], y

dev3, test3 = disk_set(s3[s3.split == "dev"]), disk_set(s3[s3.split == "test"])
comm_test = disk_set(comm[comm.split == "test"])
itw_test_set = itw_set(itw_test)
DEV = {
    "held-out speakers, clean": dev3,
    "held-out speakers, phone": ([phone(w) for w in dev3[0]], dev3[1]),
    "Commercial, unseen Hume": disk_set(comm[comm.split == "dev"]),
    "In-the-Wild (dev sample)": itw_set(itw_dev),
    "held-out speakers, noise suppression": dsp_set(dev3, "noise suppression", 11),
    "held-out speakers, auto-gain + noise suppression": dsp_set(dev3, "auto-gain + noise suppression", 12),
    "held-out speakers, Bluetooth CVSD": dsp_set(dev3, "Bluetooth CVSD", 13),
    "held-out speakers, Opus": dsp_set(dev3, "Opus", 14),
}
TEST = {
    "test speakers, clean": test3,
    "test speakers, phone": ([phone(w) for w in test3[0]], test3[1]),
    "test speakers, noise suppression": dsp_set(test3, "noise suppression", 21),
    "test speakers, spectral subtraction": dsp_set(test3, "spectral subtraction", 22),
    "test speakers, auto-gain + noise suppression": dsp_set(test3, "auto-gain + noise suppression", 23),
    "test speakers, Bluetooth CVSD": dsp_set(test3, "Bluetooth CVSD", 24),
    "test speakers, Bluetooth mSBC": dsp_set(test3, "Bluetooth mSBC", 25),
    "test speakers, Opus": dsp_set(test3, "Opus", 26),
    "ElevenLabs+Kokoro, clean": comm_test,
    "ElevenLabs+Kokoro, noise suppression": dsp_set(comm_test, "noise suppression", 27),
    "ElevenLabs+Kokoro, Opus": dsp_set(comm_test, "Opus", 28),
    "In-the-Wild (test sample)": itw_test_set,
    "In-the-Wild, noise suppression": dsp_set(itw_test_set, "noise suppression", 29),
}
for name, (w, y) in {**DEV, **TEST}.items():
    print(f"{name}: {y.count(0)} real, {y.count(1)} fake")
''')

# ---------------------------------------------------------------- Cell 6: zero-shot (Step 4 cell 6, Step 8 names)
code(S4[5].replace("# CELL 6: zero-shot scores (before training)", "# CELL 6: original model (zero-shot) on every dev and test set"))

# ---------------------------------------------------------------- Cell 7: LoRA training
code('''
# CELL 7: DSP-aware LoRA fine-tuning. The backbone is frozen; only LoRA adapters (+ the 2-way head) learn.
# The original model is step 0: a checkpoint is kept only if it beats it on the mean dev EER.
from peft import LoraConfig, get_peft_model
BS, ACCUM, EPOCHS = 8, 2, 3
LR_LORA, LR_HEAD, WARMUP = 1e-4, 1e-5, 60
EVAL_EVERY, PATIENCE = 100, 4
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj", "intermediate_dense", "output_dense"]
BEST = f"{WORK}/step8_best_trainable.pt"

for p in model.parameters():
    p.requires_grad = False
model.ssl = get_peft_model(model.ssl, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
                                                 target_modules=LORA_TARGETS))
for p in model.head.parameters():
    p.requires_grad = True
model.to(DEVICE)
n_all = sum(p.numel() for p in model.parameters())
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"trainable parameters: {n_train / 1e6:.2f} M of {n_all / 1e6:.0f} M ({100 * n_train / n_all:.2f}%)")

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
lora_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n]
opt = torch.optim.AdamW([{"params": lora_params, "lr": LR_LORA, "weight_decay": 0.01},
                         {"params": model.head.parameters(), "lr": LR_HEAD, "weight_decay": 0.0}])
total = EPOCHS * len(loader) // ACCUM
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / WARMUP) * max(0.05, 1 - s / total))
scaler = torch.amp.GradScaler("cuda", enabled=DEVICE == "cuda")

def trainable_state(m):
    return {k: v.detach().half().cpu() for k, v in m.state_dict().items() if "lora_" in k or k.startswith("head.")}

def save(m):
    torch.save(trainable_state(m), BEST)

save(model)                                      # LoRA starts at zero: this IS the original model
best, best_step, bad, step = zs_metric, 0, 0, 0
history = [{"step": 0, "dev_mean_EER": zs_metric, **zs_dev_eer}]
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
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
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
pd.DataFrame(history).to_csv(f"{WORK}/step8_history.csv", index=False)
print(f"Done in {(time.time() - t0) / 60:.1f} min. Best dev mean EER {best:.2f}% at step {best_step} "
      f"(original model: {zs_metric:.2f}%)")
''')

# ---------------------------------------------------------------- Cell 8: final test + export
code('''
# CELL 8: final test (all unseen): original vs DSP-aware LoRA, then export the improved model
model.load_state_dict({k: v.float() for k, v in torch.load(BEST, map_location="cpu").items()}, strict=False)
model.to(DEVICE).eval()
ft_dev_scores, ft_dev_eer = evaluate(model, DEV)
ft_thr = pooled_threshold(DEV, ft_dev_scores)
ft_test_scores, _ = evaluate(model, TEST)

rows = []
for name, (w, y) in TEST.items():
    for tag, sc, thr in [("original", zs_test_scores, zs_thr), ("DSP-aware LoRA", ft_test_scores, ft_thr)]:
        r = eer_auc(y, sc[name])
        fa, miss = rates_at(y, sc[name], thr)
        rows.append({"test set": name, "model": tag, "EER_%": r["EER_%"], "AUC": r["AUC"],
                     "false_alarm_on_real_%": fa, "missed_fakes_%": miss})
res = pd.DataFrame(rows).round(3)
res.to_csv(f"{WORK}/step8_results.csv", index=False)
wide = res.pivot(index="test set", columns="model", values="EER_%").reindex(list(TEST))
wide["change (points)"] = wide["DSP-aware LoRA"] - wide["original"]
print("EER % on unseen test sets (lower is better):")
print(wide.round(2).to_string())
print("\\nFull table (with false alarms on real speech and missed fakes at the dev-calibrated threshold):")
print(res.to_string(index=False))
dsp_rows = [k for k in TEST if any(d in k for d in ("noise suppression", "spectral", "auto-gain", "Bluetooth", "Opus"))]
print(f"\\nMean EER on DSP test sets: original {wide.loc[dsp_rows, 'original'].mean():.2f}% -> "
      f"LoRA {wide.loc[dsp_rows, 'DSP-aware LoRA'].mean():.2f}%")
print(f"Mean EER on all test sets: original {wide['original'].mean():.2f}% -> LoRA {wide['DSP-aware LoRA'].mean():.2f}%")
print("Best checkpoint step:", best_step, "(0 = training did not beat the original model)")

llr = lambda p: float(np.log(p / (1 - p)))
json.dump({"threshold_p_fake": ft_thr, "llr_threshold": llr(ft_thr), "original_threshold_p_fake": zs_thr,
           "original_llr_threshold": llr(zs_thr), "best_step": best_step, "window_samples": T, "sample_rate": SR,
           "output_index_fake": 0, "lora": {"r": 8, "alpha": 16, "targets": LORA_TARGETS},
           "trainable_params": n_train, "total_params": n_all},
          open(f"{WORK}/step8_threshold.json", "w"), indent=2)

if best_step == 0:
    print("Nothing to export: the original model stays in use.")
else:
    model.ssl.save_pretrained(f"{WORK}/voiceshield_lora_adapter")          # our trained weights only
    model.ssl = model.ssl.merge_and_unload()                                # fold LoRA into the weights
    model = model.float().cpu().eval()
    import torch.nn.utils.parametrize as _P                                 # weight-norm -> plain weights for ONNX
    for m in [m for m in model.modules() if _P.is_parametrized(m)]:
        for pname in list(m.parametrizations.keys()):
            _P.remove_parametrizations(m, pname, leave_parametrized=True)
    for m in [m for m in model.modules() if hasattr(m, "weight_g")]:
        torch.nn.utils.remove_weight_norm(m)
    class ExportWrap(nn.Module):
        """Same maths as Detector.forward; layer_norm over a dynamic length written as mean/var for ONNX."""
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, wav):
            wav = (wav - wav.mean(-1, keepdim=True)) / torch.sqrt(wav.var(-1, keepdim=True, unbiased=False) + 1e-5)
            return self.m.head(self.m.ssl(wav).last_hidden_state.mean(1))
    onnx_path = f"{WORK}/voiceshield_dsp_lora.onnx"
    dummy = torch.randn(1, T) * 0.1
    kw = dict(input_names=["wav"], output_names=["logits"], opset_version=17,
              dynamic_axes={"wav": {0: "batch"}, "logits": {0: "batch"}})
    try:
        torch.onnx.export(ExportWrap(model).eval(), dummy, onnx_path, dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(ExportWrap(model).eval(), dummy, onnx_path, **kw)
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    x = np.stack([crop(w) for w in TEST["test speakers, clean"][0][:6]]).astype(np.float32)
    xn = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-7)   # what the live server sends
    with torch.no_grad():
        pt = model(torch.from_numpy(x)).numpy()
    diff = float(np.abs(sess.run(None, {"wav": xn})[0] - pt).max())
    print(f"Exported {onnx_path} ({os.path.getsize(onnx_path) / 1e9:.2f} GB); max logit diff ONNX vs PyTorch {diff:.4f}")
''')


build = b4.build
CELLS_FOR_BUILD = CELLS

if __name__ == "__main__":
    b4.CELLS[:] = CELLS                      # b4.build() writes whatever is in b4.CELLS
    build(sys.argv[1])
