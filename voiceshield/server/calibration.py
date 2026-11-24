"""Line profiles: the detector threshold that fits one audio channel.

The detector's raw score shifts with the channel: clean studio speech scores far more "real"
than phone or laptop-mic speech. A threshold tuned on mixed data (Step 4: llr -4.07) is a safe
default, but each deployment should calibrate on genuine speech from its own line.

- From genuine speech only (e.g. 20 s of your own voice through the demo mic):
  threshold = 95th percentile of genuine window scores + margin, clipped to a sane range.
- From genuine + fake dev recordings: the equal-error-rate threshold.
"""
import json
import time
from pathlib import Path

import numpy as np

from .config import write_text_atomic

GENUINE_MARGIN = 1.5            # log-odds above the genuine 95th percentile
CLIP = (-9.5, 0.0)              # never calibrate outside this range
MIN_WINDOWS = 5


def threshold_from_genuine(llrs, margin: float = GENUINE_MARGIN) -> float:
    llrs = np.asarray(llrs, float)
    if len(llrs) < MIN_WINDOWS:
        raise ValueError(f"need at least {MIN_WINDOWS} analysed windows of genuine speech, got {len(llrs)}")
    return float(np.clip(np.percentile(llrs, 95) + margin, *CLIP))


def threshold_eer(llr_real, llr_fake) -> tuple[float, float]:
    """Returns (threshold, EER %) where false-alarm rate ~= miss rate."""
    real, fake = np.asarray(llr_real, float), np.asarray(llr_fake, float)
    cands = np.unique(np.concatenate([real, fake]))
    far = np.array([(real >= t).mean() for t in cands])     # genuine flagged
    frr = np.array([(fake < t).mean() for t in cands])      # fakes missed
    i = int(np.argmin(np.abs(far - frr)))
    return float(np.clip(cands[i], *CLIP)), float(100 * (far[i] + frr[i]) / 2)


class Profiles:
    """data/profiles.json: {name: {"llr_threshold": ..., "source": ..., ...}}. "default" comes from config.json.

    builtin: fixed profiles that are not stored in the file (the Step 8c "auto" profile).
    """

    def __init__(self, path: Path, default_threshold: float, default_info: dict | None = None,
                 builtin: dict | None = None):
        self.path = Path(path)
        self.default = {"llr_threshold": default_threshold,
                        **(default_info or {"source": "Step 4 dev sets (mixed channels)", "method": "pooled dev EER"})}
        self.builtin = dict(builtin or {})
        self.reserved = {"default", *self.builtin}

    def all(self) -> dict:
        saved = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        return {"default": self.default, **self.builtin, **{k: v for k, v in saved.items() if k not in self.reserved}}

    def get(self, name: str | None) -> tuple[str, dict]:
        profiles = self.all()
        name = name if isinstance(name, str) and name in profiles else "default"
        return name, profiles[name]

    def save(self, name: str, threshold: float, **info) -> dict:
        if not name or name in self.reserved or not name.replace("-", "").replace("_", "").isalnum():
            raise ValueError(f"profile name must be letters/digits/-/_ and not one of {sorted(self.reserved)}")
        saved = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        saved[name] = {"llr_threshold": round(float(threshold), 3), "created": time.strftime("%Y-%m-%d %H:%M"), **info}
        write_text_atomic(self.path, json.dumps(saved, indent=2))
        return saved[name]


def profiles_for(cfg, detector) -> Profiles:
    """A threshold only means something for the score it was set on, so each score keeps its own profiles.

    Original detector: data/profiles.json. Step 8c fused score: data/profiles_step8c.json, with its own
    default (pooled dev EER of the fused score) and the built-in "auto" profile (threshold chosen per window
    from the processing type the back-end recognises).
    """
    backend = getattr(detector, "backend", None)
    path = cfg.path(cfg.profiles_file)
    if backend is None:
        return Profiles(path, cfg.risk.llr_threshold)
    auto = {"llr_threshold": round(backend.threshold, 3), "auto": True,
            "source": "Step 8c: processing type recognised in every window",
            "method": "per-processing EER thresholds on held-out speakers",
            "thresholds": {c: round(t, 3) for c, t in backend.thresholds.items()}}
    return Profiles(path.with_name(path.stem + "_step8c" + path.suffix), round(backend.threshold, 3),
                    {"source": "Step 8c dev sets (8 sets, clean and processed)", "method": "pooled dev EER of the fused score"},
                    {"auto": auto})
