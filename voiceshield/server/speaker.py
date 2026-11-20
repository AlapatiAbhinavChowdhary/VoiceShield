"""Speaker verification: is this the person they claim to be?

WeSpeaker ResNet34-LM (VoxCeleb, ONNX) turns speech into a 256-number voiceprint. A caller's
voiceprint is compared with the enrolled one by cosine similarity. Only voiceprints are stored
(data/voiceprints.json), never audio.

Note: a good voice *clone* of the enrolled person can match their voiceprint. This check is
aimed at human impersonators and wrong callers; the deepfake detector covers clones.
"""
import json
import math
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .config import SR

# ------------------------------------------------------------------ Kaldi-compatible fbank (numpy)
_FRAME, _SHIFT, _NFFT, _NMEL = 400, 160, 512, 80          # 25 ms / 10 ms at 16 kHz, as WeSpeaker expects


def _mel(f):
    return 1127.0 * np.log(1.0 + np.asarray(f) / 700.0)


def _mel_banks() -> np.ndarray:
    lo, hi = _mel(20.0), _mel(SR / 2)
    delta = (hi - lo) / (_NMEL + 1)
    mel = _mel(np.arange(_NFFT // 2) * SR / _NFFT)[None, :]
    left = lo + np.arange(_NMEL)[:, None] * delta
    center, right = left + delta, left + 2 * delta
    banks = np.maximum(0.0, np.minimum((mel - left) / (center - left), (right - mel) / (right - center)))
    return np.pad(banks, ((0, 0), (0, 1)))                   # (80, 257): Nyquist bin gets zero weight


_BANKS = _mel_banks()
_WINDOW = np.hamming(_FRAME)


def fbank(wav: np.ndarray) -> np.ndarray:
    """80-dim log-mel filterbank like torchaudio.compliance.kaldi.fbank(hamming, dither=0), with CMN."""
    x = np.asarray(wav, np.float64) * 32768.0
    n = 1 + (len(x) - _FRAME) // _SHIFT
    if n < 1:
        return np.zeros((0, _NMEL), np.float32)
    idx = np.arange(_FRAME)[None, :] + _SHIFT * np.arange(n)[:, None]
    fr = x[idx]
    fr = fr - fr.mean(axis=1, keepdims=True)                              # remove DC per frame
    fr = fr - 0.97 * np.concatenate([fr[:, :1], fr[:, :-1]], axis=1)      # pre-emphasis
    spec = np.abs(np.fft.rfft(fr * _WINDOW, n=_NFFT)) ** 2
    feats = np.log(np.maximum(spec @ _BANKS.T, np.finfo(np.float32).eps))
    return (feats - feats.mean(axis=0)).astype(np.float32)               # cepstral mean normalisation


# ------------------------------------------------------------------ model
class SpeakerModel:
    def __init__(self, path: str | Path, threads: int = 2):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name

    def embed(self, wav: np.ndarray) -> np.ndarray:
        """Unit-length voiceprint of a speech segment (needs about 1.5 s or more)."""
        f = fbank(wav)
        if len(f) < 50:
            raise ValueError("too little speech for a voiceprint")
        e = self.sess.run(None, {self.input_name: f[None]})[0][0].astype(np.float64)
        return e / (np.linalg.norm(e) + 1e-9)


def cosine(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


class CallVoice:
    """Running voiceprint of the caller during one call (mean of window voiceprints)."""

    def __init__(self, enrolled: np.ndarray | None, threshold: float, scale: float):
        self.enrolled = None if enrolled is None else np.asarray(enrolled, float)
        self.threshold, self.scale = threshold, scale
        self.sum = None
        self.n = 0

    def add(self, emb: np.ndarray) -> dict | None:
        self.sum = emb.copy() if self.sum is None else self.sum + emb
        self.n += 1
        if self.enrolled is None:
            return None
        sim_now = cosine(emb, self.enrolled)
        sim_call = cosine(self.sum, self.enrolled)
        # mismatch probability: 0.5 at the threshold, near 1 well below it
        z = (self.threshold - sim_call) / self.scale
        mismatch = 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, z))))
        return {"similarity": round(sim_call, 3), "similarity_window": round(sim_now, 3),
                "mismatch": mismatch, "windows": self.n}


class Voiceprints:
    """data/voiceprints.json: {name: {"embedding": [...256], "seconds": .., "created": ..}}"""

    def __init__(self, path: Path):
        self.path = Path(path)

    def all(self) -> dict:
        return json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}

    def names(self) -> list[str]:
        return sorted(self.all())

    def get(self, name: str | None) -> np.ndarray | None:
        rec = self.all().get(name or "")
        return None if rec is None else np.asarray(rec["embedding"], float)

    def save(self, name: str, embedding: np.ndarray, seconds: float, source: str = "") -> dict:
        name = name.strip()
        if not name or len(name) > 40 or not all(c.isalnum() or c in " -_." for c in name):
            raise ValueError("name must be 1-40 letters/digits/space/-/_/.")
        data = self.all()
        data[name] = {"embedding": [round(float(v), 6) for v in embedding], "seconds": round(seconds, 1),
                      "source": source, "created": time.strftime("%Y-%m-%d %H:%M")}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data), encoding="utf-8")
        return {k: v for k, v in data[name].items() if k != "embedding"}

    def delete(self, name: str) -> bool:
        data = self.all()
        if name not in data:
            return False
        del data[name]
        self.path.write_text(json.dumps(data), encoding="utf-8")
        return True
