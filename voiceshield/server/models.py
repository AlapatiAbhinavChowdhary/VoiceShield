"""The ONNX models: Silero VAD (speech or not) and NII AntiDeepfake (real or synthetic), optionally with the
Step 8c back-end on the detector's hidden layers."""
import logging
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .config import SR

log = logging.getLogger("voiceshield")


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, x))))


class VadModel:
    """Silero VAD v5. One shared session; each stream keeps its own state (see new_state)."""

    FRAME = 512      # samples per decision at 16 kHz (32 ms)
    CONTEXT = 64     # v5 expects the last 64 samples of the previous frame in front of each frame

    def __init__(self, path: str | Path):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        names = {i.name for i in self.sess.get_inputs()}
        if not {"input", "state", "sr"} <= names:
            raise RuntimeError(f"Unexpected Silero VAD inputs {names}; use the v5 model from setup_models.py")
        self._sr = np.array(SR, dtype=np.int64)

    def new_state(self) -> dict:
        return {"state": np.zeros((2, 1, 128), np.float32), "context": np.zeros((1, self.CONTEXT), np.float32)}

    def prob(self, frame: np.ndarray, st: dict) -> float:
        x = np.concatenate([st["context"], frame.reshape(1, -1).astype(np.float32)], axis=1)
        out, st["state"] = self.sess.run(None, {"input": x, "state": st["state"], "sr": self._sr})
        st["context"] = x[:, -self.CONTEXT:]
        return float(out.reshape(-1)[0])


def fit_window(wav: np.ndarray, n: int) -> np.ndarray:
    """Exactly n samples: the last n if longer, repeated (as in training) if shorter."""
    wav = np.asarray(wav, np.float32)
    if len(wav) == 0:
        return np.zeros(n, np.float32)
    if len(wav) < n:
        wav = np.tile(wav, n // len(wav) + 1)[:n]
    return wav[-n:]


def _softmax(a: np.ndarray) -> np.ndarray:
    e = np.exp(a - a.max())
    return e / e.sum()


# The notebook's "auto-gain + noise suppression" effect is auto-gain followed by SPECTRAL SUBTRACTION.
PROCESSING_LABELS = {"auto-gain + noise suppression": "auto-gain + spectral subtraction"}


class LayerBackend:
    """Step 8c back-end (notebooks/step8c_frozen_backend.ipynb), in NumPy. The detector itself is unchanged.

    Input: the detector's own score (llr) and the time-means of its 25 hidden layers (25 x 1024).
    - back-end score q: softmax-weighted sum of the standardised layers -> linear read-out (1,050 parameters);
    - fused score f = (1 - w) * (llr - mo) / so + w * (q - mp) / sp, with w chosen on the dev sets (0.85);
      it is reported on the original llr scale as mo + so * f, so that w = 0 would give back exactly the
      original llr and the line-profile rules (margins, clipping) keep their meaning;
    - processing classifier (9 types) and one calibrated threshold per type, for the "auto" line profile.
    """

    def __init__(self, path: str | Path):
        with np.load(path) as B:
            f64 = lambda k: np.asarray(B[k], np.float64)   # noqa: E731
            self.conds = [PROCESSING_LABELS.get(str(c), str(c)) for c in B["conds"]]
            self.mu, self.sd = f64("mu"), f64("sd")
            self.det_pi, self.det_w, self.det_b = _softmax(f64("det_a")), f64("det_w"), float(f64("det_b").reshape(-1)[0])
            self.cls_pi, self.cls_w, self.cls_b = _softmax(f64("cls_a")), f64("cls_w"), f64("cls_b")
            self.w = float(B["fusion_w"])
            self.mo, self.so, self.mp, self.sp = (float(B[k]) for k in ("mo", "so", "mp", "sp"))
            self.threshold = self.to_llr(float(B["thr_single_fused"]))          # pooled dev EER, fused score
            self.thresholds = {c: self.to_llr(float(t)) for c, t in zip(self.conds, B["thr_fused"])}
        n, d = self.mu.shape
        if self.sd.shape != (n, d) or self.det_w.shape != (d,) or self.cls_w.shape != (len(self.conds), d):
            raise ValueError(f"{path}: unexpected array shapes")
        self.layers = n

    def to_llr(self, f: float) -> float:
        return self.mo + self.so * f

    def __call__(self, llr: float, layers: np.ndarray) -> dict:
        z = (np.asarray(layers, np.float64) - self.mu) / self.sd               # (25, 1024)
        q = float(self.det_pi @ z @ self.det_w + self.det_b)
        f = (1 - self.w) * (llr - self.mo) / self.so + self.w * (q - self.mp) / self.sp
        c = self.conds[int(np.argmax(self.cls_w @ (self.cls_pi @ z) + self.cls_b))]
        return {"llr": self.to_llr(f), "llr_original": llr, "backend": q, "processing": c,
                "auto_threshold": self.thresholds[c]}


class Detector:
    """NII AntiDeepfake wav2vec2-Large (ONNX). Output index 0 = fake, 1 = real.

    With a LayerBackend (Step 8c) the ONNX file must also output "layers", and score() returns the fused
    score as "llr" (the detector's own score stays available as "llr_original").
    """

    def __init__(self, path: str | Path, threads: int = 0, providers: list[str] | None = None,
                 backend: LayerBackend | None = None):
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.log_severity_level = 3
        if threads:
            so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(path), so, providers=providers or ["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        self.window = inp.shape[1] if isinstance(inp.shape[1], int) else 64000
        self.provider = self.sess.get_providers()[0]
        outs = [o.name for o in self.sess.get_outputs()]
        if backend is not None and "layers" not in outs:
            raise RuntimeError(f"{Path(path).name} has no 'layers' output; the Step 8c back-end needs "
                               "voiceshield_layers.onnx from the Step 8c notebook")
        self.backend = backend
        self.outputs = [outs[0]] + (["layers"] if backend is not None else [])
        self.mode = "step8c" if backend is not None else "original"

    def score(self, wav: np.ndarray) -> dict:
        x = fit_window(wav, self.window)
        x = (x - x.mean()) / (x.std() + 1e-7)                      # per-utterance normalisation, as in training
        out = self.sess.run(self.outputs, {self.input_name: x[None, :].astype(np.float32)})
        logits = out[0][0]
        llr = float(logits[0] - logits[1])                           # log-odds fake vs real
        if self.backend is None:
            return {"llr": llr, "p_fake": sigmoid(llr)}
        r = self.backend(llr, out[1][0])
        r["p_fake"] = sigmoid(r["llr"])
        return r


def load_detector(cfg) -> Detector:
    """The detector chosen by config.json "detector_mode": "step8c" (frozen detector + back-end) or "original".

    If the Step 8c files are missing, falls back to the original detector (and says so in the log).
    """
    if cfg.detector_mode == "step8c":
        onnx, npz = cfg.models / cfg.layers_file, cfg.models / cfg.backend_file
        if onnx.exists() and npz.exists():
            return Detector(onnx, cfg.detector_threads, backend=LayerBackend(npz))
        log.warning("detector_mode is step8c but %s is missing; using the original detector",
                    " and ".join(p.name for p in (onnx, npz) if not p.exists()))
    path = cfg.models / cfg.detector_file
    if not path.exists():
        raise RuntimeError(f"Detector model not found: {path}. Run:  python setup_models.py")
    return Detector(path, cfg.detector_threads)
