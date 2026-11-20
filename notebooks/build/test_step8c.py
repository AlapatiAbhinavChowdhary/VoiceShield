"""Local CPU smoke test of the Step 8c notebook (stub data, real-shaped random weights, real ONNX export).

    python test_step8c.py      (needs torch, transformers, onnx, onnxruntime, noisereduce, scipy, sklearn, pandas)
"""
import json
import os
import shutil
import sys
import tempfile
import time
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import build_step8c  # noqa: E402

try:                                         # a standalone ffmpeg so the Opus path runs too
    import imageio_ffmpeg
    d = tempfile.mkdtemp()
    shutil.copy(imageio_ffmpeg.get_ffmpeg_exe(), os.path.join(d, "ffmpeg.exe" if os.name == "nt" else "ffmpeg"))
    os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]
except ImportError:
    pass

ks = types.ModuleType("kaggle_secrets")
ks.UserSecretsClient = type("U", (), {"get_secret": lambda self, n: "dummy"})
sys.modules["kaggle_secrets"] = ks
ds = types.ModuleType("datasets")
ds.Audio = type("Audio", (), {"__init__": lambda self, decode=True: None})
ds.load_dataset = lambda *a, **k: None
sys.modules["datasets"] = ds
import huggingface_hub  # noqa: E402
huggingface_hub.login = lambda *a, **k: None

code = [src for kind, src in build_step8c.CELLS if kind == "code"]
cell = {i + 1: src for i, src in enumerate(code)}
assert len(cell) == 11, len(cell)
G = {}
TMPROOT = tempfile.mkdtemp()
WORK = f"{TMPROOT}/working"

# --- CELL 2: setup + DSP effects; CELL 4: model with zero-mean random weights of the REAL shapes ---------
exec(cell[2].replace('"/kaggle/working"', repr(WORK)).replace('"/kaggle/temp"', repr(f"{TMPROOT}/temp")), G)
exec(cell[4].split("NII_SD = ")[0], G)
hdr = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "hdr.json")))
hdr.pop("__metadata__", None)
torch.manual_seed(0)
G["model"] = G["load_antideepfake"]({k: torch.randn(v["shape"]) * 0.02 for k, v in hdr.items()}, True)
src5 = cell[5]
exec(src5[src5.index("def dsp_set"):src5.index("dev3, test3 =")], G)          # dsp_set from Cell 5

# --- tiny data with the real set names -------------------------------------------------------------------
import pandas as pd  # noqa: E402
import soundfile as sf  # noqa: E402
rng = np.random.default_rng(0)
rows = []
for i in range(4):
    p = f"{TMPROOT}/{i}.flac"
    sf.write(p, (rng.standard_normal(16000 * 5) * 0.1).astype(np.float32), 16000)
    rows.append(dict(path=p, label=i % 2))
G["train_df"] = pd.DataFrame(rows)


def tiny():
    return [(rng.standard_normal(70000) * 0.1).astype(np.float32) for _ in range(4)], [0, 1, 0, 1]


G["dev3"] = tiny()
G["DEV"] = {k: tiny() for k in [
    "held-out speakers, clean", "held-out speakers, phone", "Commercial, unseen Hume", "In-the-Wild (dev sample)",
    "held-out speakers, noise suppression", "held-out speakers, auto-gain + noise suppression",
    "held-out speakers, Bluetooth CVSD", "held-out speakers, Opus"]}
G["TEST"] = {k: tiny() for k in ["test speakers, clean", "test speakers, Opus", "ElevenLabs+Kokoro, noise suppression",
                                  "In-the-Wild (test sample)"]}

# --- CELLS 6-11 ------------------------------------------------------------------------------------------
c6 = cell[6].replace("num_workers=min(4, os.cpu_count() or 2)", "num_workers=0")
assert c6 != cell[6]
for n, src in [(6, c6), (7, cell[7]), (8, cell[8]), (9, cell[9]), (10, cell[10]), (11, cell[11])]:
    t = time.time()
    exec(src, G)
    print(f"cell {n} OK ({time.time() - t:.0f} s)")

assert G["NL"] == 25 and G["Htr"].shape == (12, 25, 1024)
assert set(G["cond_of"](k) for k in G["FE"]) >= {"clean", "Opus", "noise suppression", "spectral subtraction",
                                                  "auto-gain", "Bluetooth mSBC", "phone line (G.711)"}
assert G["cond_of"]("held-out speakers, auto-gain + noise suppression") == "auto-gain + noise suppression"
for f in ["step8c_layers.csv", "step8c_fusion.csv", "step8c_results.csv", "step8c_shift.csv", "step8c_summary.json",
          "step8c_backend.npz", "voiceshield_layers.onnx"]:
    assert os.path.exists(f"{WORK}/{f}"), f
assert os.path.getsize(f"{WORK}/voiceshield_layers.onnx") > 1e9
res = pd.read_csv(f"{WORK}/step8c_results.csv")
assert len(res) == 4 * 3 and set(res.scorer) == {"original", "back-end", "fused"}

# --- the live-server path: ONNX layers + NumPy back-end must reproduce the notebook's scores --------------
import onnxruntime as ort  # noqa: E402
B = np.load(f"{WORK}/step8c_backend.npz")
sess = ort.InferenceSession(f"{WORK}/voiceshield_layers.onnx", providers=["CPUExecutionProvider"])
name = "test speakers, Opus"
x = np.stack([G["crop"](w) for w in G["TEST"][name][0]]).astype(np.float32)
xn = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-7)
logits, lay = sess.run(None, {"wav": xn})


def softmax(a):
    e = np.exp(a - a.max())
    return e / e.sum()


def backend(lay, o):
    z = (lay - B["mu"]) / B["sd"]
    p = (softmax(B["det_a"])[None, :, None] * z).sum(1) @ B["det_w"] + B["det_b"][0]
    w = float(B["fusion_w"])
    fused = (1 - w) * (o - B["mo"]) / B["so"] + w * (p - B["mp"]) / B["sp"]
    cond = ((softmax(B["cls_a"])[None, :, None] * z).sum(1) @ B["cls_w"].T + B["cls_b"]).argmax(1)
    return fused, cond


# 1. NumPy back-end on the notebook's own features must give exactly the notebook's scores (formula check)
H, L, _ = G["FE"][name]
fused, cond = backend(H.astype(np.float32), L)
d_f, d_c = np.abs(fused - G["S"]["fused"][name]).max(), (cond == G["PRED"][name]).mean()
print(f"NumPy back-end vs notebook (same features): fused diff {d_f:.5f}, same processing class {100 * d_c:.0f}%")
assert d_f < 1e-3 and d_c == 1.0
# 2. ONNX layers vs the notebook's fp16 features (the live-server input)
o = logits[:, 0] - logits[:, 1]
rel = np.abs(lay - H.astype(np.float32)).max() / np.abs(H.astype(np.float32)).max()
print(f"ONNX vs notebook: original llr diff {np.abs(o - L).max():.4f}, layer features max relative diff {rel:.5f}; "
      f"smallest feature spread in the random test weights {float(B['sd'].min()):.5f}")
assert np.abs(o - L).max() < 0.05 and rel < 1e-2
print("\nALL LOCAL CHECKS PASSED")
