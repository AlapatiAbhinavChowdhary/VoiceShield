"""Local CPU smoke test of the Step 4 notebook logic (random weights, tiny fake data)."""
import sys, types, json, os, tempfile, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import build_step4

# --- stub Kaggle / HF-only modules -------------------------------------------
ks = types.ModuleType("kaggle_secrets")
class _USC:
    def get_secret(self, name): return "dummy"
ks.UserSecretsClient = _USC
sys.modules["kaggle_secrets"] = ks
ds = types.ModuleType("datasets")
class Audio:  # noqa
    def __init__(self, decode=True): pass
ds.Audio, ds.load_dataset = Audio, lambda *a, **k: None
sys.modules["datasets"] = ds
try:
    import pyarrow.parquet  # noqa
except ImportError:
    pa = types.ModuleType("pyarrow"); pq = types.ModuleType("pyarrow.parquet")
    pa.parquet = pq; sys.modules["pyarrow"] = pa; sys.modules["pyarrow.parquet"] = pq
import huggingface_hub
huggingface_hub.login = lambda *a, **k: None

code = [src for kind, src in build_step4.CELLS if kind == "code"]
cell = {i + 1: src for i, src in enumerate(code)}      # cell[2] = helpers, ... cell[8] = final
G = {}

# --- CELL 2: helpers ----------------------------------------------------------
TMPROOT = tempfile.mkdtemp()
def _paths(src):
    for lit in ('"/kaggle/working"', '"/kaggle/temp"', '"/content/drive/MyDrive/SIH26104/step4"', '"/content/tmp"'):
        assert src.count(lit) == 1, lit
    return (src.replace('"/kaggle/working"', repr(f"{TMPROOT}/working"))
              .replace('"/kaggle/temp"', repr(f"{TMPROOT}/temp"))
              .replace('"/content/drive/MyDrive/SIH26104/step4"', repr(f"{TMPROOT}/drive_work"))
              .replace('"/content/tmp"', repr(f"{TMPROOT}/colab_tmp")))
# Colab branch: no kaggle_secrets, fake google.colab
_ks = sys.modules.pop("kaggle_secrets")
gc = types.ModuleType("google.colab"); mounted = []
gc.userdata = types.SimpleNamespace(get=lambda k: "dummy-colab")
gc.drive = types.SimpleNamespace(mount=lambda p: mounted.append(p))
sys.modules["google.colab"] = gc
G2 = {}
exec(_paths(cell[2]), G2)
assert G2["ON_COLAB"] and mounted == ["/content/drive"] and G2["WORK"].endswith("drive_work") and os.path.isdir(G2["WORK"])
assert G2["TMP"].endswith("colab_tmp") and os.path.isdir(G2["TMP"])
print("cell 2 Colab branch OK")
sys.modules["kaggle_secrets"] = _ks; del sys.modules["google.colab"]
exec(_paths(cell[2]), G)
assert not G["ON_COLAB"] and G["TMP"] == f"{TMPROOT}/temp" and os.path.isdir(G["WORK"])
print("cell 2 Kaggle branch OK")
rng = np.random.default_rng(0)
for n in (8000, 80001, 160000):
    w = (rng.standard_normal(n) * 0.1).astype(np.float32)
    a = G["augment"](w)
    assert a.shape == (64000,) and a.dtype == np.float32 and np.isfinite(a).all(), (n, a.shape, a.dtype)
    p = G["phone"](w)
    assert abs(len(p) - n) <= 1 and np.isfinite(p).all()
print("augment/phone OK")
y = np.r_[np.zeros(100), np.ones(100)].astype(int)
r = G["eer_auc"](y, np.r_[rng.uniform(0, .4, 100), rng.uniform(.6, 1, 100)])
assert r["EER_%"] == 0 and r["AUC"] == 1, r
r = G["eer_auc"](y, rng.uniform(0, 1, 200)); assert 30 < r["EER_%"] < 70, r
print("eer_auc OK")

# --- CELL 3: attached-input path and rebuild path (stubbed Kathbath + kNN-VC) ------
import io, shutil, soundfile as sf, pandas as pd
assert G["WORK"] == f"{TMPROOT}/working"
def _flac(sec, seed):
    b = io.BytesIO(); r_ = np.random.default_rng(seed)
    sf.write(b, (r_.standard_normal(int(16000 * sec)) * 0.1).astype(np.float32), 16000, format="FLAC")
    return b.getvalue()
SHARDS = {f"hindi/train-{i:05d}.parquet": i for i in range(4)} | {f"telugu/train-{i:05d}.parquet": i + 10 for i in range(4)}
def _shard_df(p):
    base, rows_ = SHARDS[p], []
    for s in range(3):                                   # 3 speakers per shard, 6 clips each
        for c in range(6):
            rows_.append(dict(speaker_id=f"{p.split('/')[0]}_{base}_{s}", gender="female" if s % 2 else "male",
                              text=f"t{c}", audio_filepath={"bytes": _flac(1.0, base * 100 + s * 10 + c), "path": "a.flac"}))
    return pd.DataFrame(rows_)
class _F:
    def __init__(self, p): self.path = p
class _Api:
    def list_repo_tree(self, repo, path_in_repo, repo_type, recursive):
        return [_F(p) for p in SHARDS if p.startswith(path_in_repo + "/")] + [_F(f"{path_in_repo}/valid-0.parquet")]
class _T:
    def __init__(self, p): self.p = p
    def to_pandas(self): return _shard_df(self.p)
_pq = types.SimpleNamespace(read_table=lambda p: _T(p))
class _Knn:
    def get_matching_set(self, refs, vad_trigger_level=0): assert len(refs) == 3; return torch.zeros(1)
    def get_features(self, src): return src
    def match(self, f, ms, topk=4): return f * 0.5
G.update(api=_Api(), pq=_pq, hf_hub_download=lambda repo, p, repo_type=None: p)
c3 = (cell[3].replace('INPUT  = "/kaggle/input"', f'INPUT  = {repr(TMPROOT + "/input")}')
             .replace("N_SHARDS, N_SPK, N_REF, N_PER = 6, 20, 40, 25", "N_SHARDS, N_SPK, N_REF, N_PER = 2, 4, 3, 2"))
assert c3.count(repr(TMPROOT + "/input")) == 1 and "2, 4, 3, 2" in c3
_hub = torch.hub.load
def run3(expect_rebuild):
    calls = []
    torch.hub.load = lambda *a, **k: (calls.append(1), _Knn())[1]
    try:
        exec(c3, G)
    finally:
        torch.hub.load = _hub
    assert bool(calls) == expect_rebuild, (calls, expect_rebuild)
    s = G["s3"]
    assert all(os.path.exists(p) for p in s.path)
    return s
# (a) nothing attached -> rebuild
s_a = run3(True)
assert len(s_a) == 2 * 4 * 2 * 2, len(s_a)              # langs x speakers x (real+fake) x N_PER
assert set(s_a.split) == {"train", "test"} or {"train", "dev", "test"} >= set(s_a.split)
assert os.path.exists(f"{G['WORK']}/step3_data.zip") and all(G["TMP"] in p for p in s_a.path)
assert G["HELD_OUT"] == set(s_a.loc[s_a.split != "train", "speaker"].astype(str))
# (b) complete Step 3 output attached (paths as Step 3 wrote them) -> reuse, no rebuild
att = f"{TMPROOT}/input/my-step3/step3_data"
shutil.copytree(G["S3_DIR"], att, ignore=shutil.ignore_patterns("metadata.csv"))
m = s_a.drop(columns="source").copy()
m["path"] = [p.replace(G["S3_DIR"], "/kaggle/working/step3_data") for p in m.path]
m.to_csv(f"{att}/metadata.csv", index=False)
s_b = run3(False)
assert all(os.path.normpath(p).startswith(os.path.normpath(att)) for p in s_b.path), s_b.path.iloc[0]
assert list(s_b.split) == list(s_a.split)
# (c) attached but incomplete (Kaggle dropped files) + zip from an earlier run -> reuse the zip
os.remove(s_b.path.iloc[-1])
s_c = run3(False)
assert all("step3_from_zip" in p for p in s_c.path) and list(s_c.split) == list(s_a.split)
# (d) incomplete attachment + corrupt zip (e.g. disconnect while saving) -> rebuild, same speakers/splits
open(f"{G['WORK']}/step3_data.zip", "wb").write(b"not a zip")
s_d = run3(True)
assert dict(zip(s_d.speaker, s_d.split)) == dict(zip(s_a.speaker, s_a.split)), "rebuild not reproducible"
print("cell 3 OK (rebuild / attached / zip reuse / corrupt zip -> rebuild); held-out:", sorted(G["HELD_OUT"]))

# --- CELL 4: model definition + weight mapping (zeros with the REAL header shapes)
defs = cell[4].split("NII_SD = ")[0]
exec(defs, G)
hdr = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "hdr.json")))
hdr.pop("__metadata__", None)
sd = {k: torch.zeros(v["shape"]) for k, v in hdr.items()}
for stable in (True, False):
    m = G["load_antideepfake"](sd, stable)       # strict=True inside: raises on any mismatch
    print(f"strict load OK (stable={stable})")
model = G["load_antideepfake"]({k: torch.randn_like(v) * 0.02 for k, v in sd.items()}, True)
torch.manual_seed(0)
for prm in model.parameters():                   # random but sane weights
    if prm.dim() > 1: torch.nn.init.xavier_uniform_(prm)
with torch.no_grad():
    out = model(torch.randn(2, 16000))
assert out.shape == (2, 2), out.shape
print("forward OK", out.shape, "p_fake", model.p_fake(torch.randn(2, 16000)).tolist())
G["model"] = model

# --- tiny fake dataset on disk -------------------------------------------------
import soundfile as sf, pandas as pd
tmp = tempfile.mkdtemp()
rows = []
for i in range(4):
    p = f"{tmp}/{i}.flac"; sf.write(p, (rng.standard_normal(16000 * 5) * 0.1).astype(np.float32), 16000)
    rows.append(dict(path=p, label=i % 2))
G["train_df"] = pd.DataFrame(rows)
tiny = ([(rng.standard_normal(70000) * 0.1).astype(np.float32) for _ in range(4)], [0, 1, 0, 1])
G["DEV"] = {"dev a": tiny, "dev b": tiny}
G["TEST"] = {"t": tiny, "In-the-Wild (test sample)": tiny}
G["WORK"] = tmp

# --- CELL 6: zero-shot eval -------------------------------------------------------
exec(cell[6], G)
print("cell 6 OK, zs_metric", G["zs_metric"], "thr", G["zs_thr"])

# --- CELL 7: training (patched: tiny batch, no worker processes on Windows) ------
c7 = (cell[7].replace("BS, ACCUM, EPOCHS = 8, 2, 3", "BS, ACCUM, EPOCHS = 2, 1, 1")
              .replace("EVAL_EVERY, PATIENCE = 100, 4", "EVAL_EVERY, PATIENCE = 1, 4")
              .replace("num_workers=min(4, os.cpu_count() or 2), worker_init_fn=_seed_worker,\n"
                       "                                     persistent_workers=True",
                       "num_workers=0, worker_init_fn=_seed_worker"))
assert c7 != cell[7]
t = time.time(); exec(c7, G); print(f"cell 7 OK ({time.time() - t:.0f}s), history:", G["history"])
fe_grad = [p.requires_grad for p in G["model"].ssl.feature_extractor.parameters()]
assert not any(fe_grad), "feature extractor not frozen"

# --- CELL 8: final eval (latency loop shortened) -----------------------------------
c8 = cell[8].replace("range(30)", "range(2)").replace("/ 30 *", "/ 2 *").replace("range(5)", "range(1)")
exec(c8, G)
print("cell 8 OK; threshold.json:", open(f"{tmp}/threshold.json").read())
print("\nALL LOCAL CHECKS PASSED")
