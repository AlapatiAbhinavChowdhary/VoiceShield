"""Local CPU smoke test of the Step 8 notebook (stub data, real-shaped weights, real LoRA + ONNX export).

    python test_step8.py            (needs torch, transformers, peft, onnx, onnxruntime, noisereduce, scipy)
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

import importlib  # noqa: E402
MOD = sys.argv[1] if len(sys.argv) > 1 else "build_step8"
build_step8 = importlib.import_module(MOD)  # noqa: E402
print("testing", MOD)

# --- optional: a standalone ffmpeg so the Opus path is exercised too ---------------------------
try:
    import imageio_ffmpeg
    d = tempfile.mkdtemp()
    shutil.copy(imageio_ffmpeg.get_ffmpeg_exe(), os.path.join(d, "ffmpeg.exe" if os.name == "nt" else "ffmpeg"))
    os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]
except ImportError:
    pass

# --- stub Kaggle / HF-only modules --------------------------------------------------------------
ks = types.ModuleType("kaggle_secrets")
ks.UserSecretsClient = type("U", (), {"get_secret": lambda self, n: "dummy"})
sys.modules["kaggle_secrets"] = ks
ds = types.ModuleType("datasets")
ds.Audio = type("Audio", (), {"__init__": lambda self, decode=True: None})
ds.load_dataset = lambda *a, **k: None
sys.modules["datasets"] = ds
import huggingface_hub  # noqa: E402
huggingface_hub.login = lambda *a, **k: None

code = [src for kind, src in build_step8.CELLS if kind == "code"]
cell = {i + 1: src for i, src in enumerate(code)}
G = {}
TMPROOT = tempfile.mkdtemp()

# --- CELL 2: setup + DSP effects ---------------------------------------------------------------
c2 = (cell[2].replace('"/kaggle/working"', repr(f"{TMPROOT}/working"))
             .replace('"/kaggle/temp"', repr(f"{TMPROOT}/temp")))
exec(c2, G)
print("ffmpeg available:", G["HAVE_FFMPEG"])
rng = np.random.default_rng(0)
speechlike = (np.sin(2 * np.pi * 180 * np.arange(64000) / 16000) * (0.2 + 0.1 * rng.standard_normal(64000))).astype(np.float32)
for name, fn in G["DSP"].items():
    t = time.time()
    y = fn(G["add_noise"](speechlike.astype(np.float64), 15))
    assert len(y) == len(speechlike) and np.isfinite(y).all(), name
    assert np.abs(y).max() > 1e-3, f"{name} returned silence"
    print(f"  DSP {name:30s} ok  ({(time.time() - t) * 1000:.0f} ms, rms {np.sqrt(np.mean(np.square(y))):.3f})")
for n in (8000, 80001, 160000):
    a = G["augment"]((rng.standard_normal(n) * 0.1).astype(np.float32))
    assert a.shape == (64000,) and a.dtype == np.float32 and np.isfinite(a).all()
assert G["rates_at"]([0, 0, 1, 1], [0.1, 0.9, 0.8, 0.2], 0.5) == (50.0, 50.0)
print("augment / rates_at OK")

# --- CELL 4: model definition + weight mapping with zero tensors of the REAL shapes --------------
exec(cell[4].split("NII_SD = ")[0], G)
hdr = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "hdr.json")))
hdr.pop("__metadata__", None)
torch.manual_seed(0)
sd = {k: torch.randn(v["shape"]) * 0.02 for k, v in hdr.items()}
model = G["load_antideepfake"](sd, True)
G["model"] = model
print("model built:", sum(p.numel() for p in model.parameters()) // 1_000_000, "M parameters")

# --- tiny data ------------------------------------------------------------------------------------
import pandas as pd  # noqa: E402
import soundfile as sf  # noqa: E402
rows = []
for i in range(4):
    p = f"{TMPROOT}/{i}.flac"
    sf.write(p, (rng.standard_normal(16000 * 5) * 0.1).astype(np.float32), 16000)
    rows.append(dict(path=p, label=i % 2))
G["train_df"] = pd.DataFrame(rows)
tiny = ([(rng.standard_normal(70000) * 0.1).astype(np.float32) for _ in range(4)], [0, 1, 0, 1])
G["DEV"] = {"held-out speakers, clean": tiny, "held-out speakers, noise suppression": tiny}
G["TEST"] = {"test speakers, clean": tiny, "test speakers, Opus": tiny, "In-the-Wild (test sample)": tiny}

# --- CELL 6: original model on dev/test --------------------------------------------------------------
exec(cell[6], G)
print("cell 6 OK, zs_metric", G["zs_metric"])

# --- CELL 7: LoRA training (tiny, CPU) -----------------------------------------------------------------
c7 = (cell[7].replace("BS, ACCUM, EPOCHS = 8, 2, 3", "BS, ACCUM, EPOCHS = 2, 1, 1")
              .replace("EVAL_EVERY, PATIENCE = 100, 4", "EVAL_EVERY, PATIENCE = 1, 4")
              .replace("EVAL_EVERY, PATIENCE = 100, 6", "EVAL_EVERY, PATIENCE = 1, 6")
              .replace("num_workers=min(4, os.cpu_count() or 2), worker_init_fn=_seed_worker,\n"
                       "                                     persistent_workers=True",
                       "num_workers=0, worker_init_fn=_seed_worker"))
assert c7 != cell[7]
t = time.time()
exec(c7, G)
frac = G["n_train"] / G["n_all"]
assert 0.005 < frac < 0.02, frac
lora_grads = [n for n, p in G["model"].named_parameters() if p.requires_grad]
assert all("lora_" in n or n.startswith("head.") for n in lora_grads)
saved = torch.load(G["BEST"])
assert saved and all("lora_" in k or k.startswith("head.") for k in saved), list(saved)[:3]
print(f"cell 7 OK ({time.time() - t:.0f} s): {G['n_train'] / 1e6:.2f} M trainable ({100 * frac:.2f}%), "
      f"checkpoint {sum(v.numel() for v in saved.values()) / 1e6:.2f} M values")

# --- CELL 8: final test + export (export forced even if training did not win) ---------------------------
c8 = cell[8].replace("if best_step == 0:", "if False:")
t = time.time()
exec(c8, G)
onnx_path = f"{TMPROOT}/working/voiceshield_dsp_lora.onnx"
assert os.path.exists(onnx_path) and os.path.getsize(onnx_path) > 1e9
assert os.path.isdir(f"{TMPROOT}/working/voiceshield_lora_adapter")
import glob as _g
thr = json.load(open(_g.glob(f"{TMPROOT}/working/step8*_threshold.json")[0]))
assert {"llr_threshold", "original_llr_threshold", "trainable_params"} <= set(thr)
print(f"cell 8 OK ({time.time() - t:.0f} s); adapter files:", os.listdir(f"{TMPROOT}/working/voiceshield_lora_adapter"))
print("\nALL LOCAL CHECKS PASSED")
