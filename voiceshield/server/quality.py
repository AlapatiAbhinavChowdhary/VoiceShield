"""Audio quality of one analysis window, and how much to trust the detector's call on it.

Informational only: nothing here changes a score, a threshold or a decision. It tells the operator when the
audio itself is unreliable (very quiet, clipped, or noisy), which Steps 6b and 8 showed is when the detector
errs most, and marks windows whose score sits close to the line's cut-off as low-confidence.
"""
import numpy as np

from .config import SR

FRAME = int(0.02 * SR)          # 20 ms frames


def audio_quality(w: np.ndarray) -> dict:
    """Level (dBFS), clipping share, and an SNR estimate from the spread of frame energies.

    The SNR proxy is the gap between loud (90th percentile) and quiet (10th percentile) 20 ms frames: speech
    over a clean line shows a wide gap; speech buried in steady noise, or noise suppression artefacts, a narrow
    one. It is an estimate, not a measurement against a clean reference.
    """
    x = np.asarray(w, np.float32)
    if x.size < FRAME * 5:
        return {"level_dbfs": None, "snr_db": None, "clipping": 0.0, "label": "unknown"}
    rms = float(np.sqrt(np.mean(x ** 2)) + 1e-12)
    n = x.size // FRAME
    fr = np.sqrt(np.mean(x[: n * FRAME].reshape(n, FRAME) ** 2, axis=1)) + 1e-9
    lo, hi = np.percentile(fr, [10, 90])
    snr = float(min(60.0, 20 * np.log10(hi / lo)))     # capped: digital silence between clips would give > 100 dB
    clip = float(np.mean(np.abs(x) >= 0.985))
    level = 20 * np.log10(rms)
    # a raw laptop mic sits near -52 dBFS and still scores correctly (Step 5b), so level alone only flags near-silence
    if level < -60 or clip > 0.01 or snr < 10:
        label = "poor"
    elif level < -50 or clip > 0.001 or snr < 18:
        label = "fair"
    else:
        label = "good"
    return {"level_dbfs": round(level, 1), "snr_db": round(snr, 1), "clipping": round(clip, 4), "label": label}


def confidence(llr: float, cut: float, quality_label: str, margin: float = 1.0) -> str:
    """"low" when the window's score is within `margin` log-odds of the cut-off or the audio is poor, else "normal"."""
    return "low" if abs(llr - cut) < margin or quality_label == "poor" else "normal"
