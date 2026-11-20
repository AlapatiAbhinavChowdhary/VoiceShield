"""Calibrate a line profile from recordings of that line.

Genuine speech only (threshold = genuine 95th percentile + 1.5):
    python tools/calibrate.py --save-as mic --genuine my_voice_*.wav

Genuine + fake dev recordings (equal-error-rate threshold):
    python tools/calibrate.py --save-as studio --genuine dev_real/*.flac --fake dev_fake/*.flac

Never calibrate on the recordings you will test or demo with.
The profile is saved for the detector chosen in config.json ("detector_mode"): original scores go to
data/profiles.json, Step 8c fused scores to data/profiles_step8c.json.
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.calibration import profiles_for, threshold_eer, threshold_from_genuine  # noqa: E402
from server.config import load_config                                             # noqa: E402
from server.models import VadModel, load_detector                                 # noqa: E402
from server.offline import analyse, load_audio                                    # noqa: E402


def window_llrs(files, det, vad, cfg) -> list[float]:
    out = []
    for f in files:
        out += [w["llr"] for w in analyse(load_audio(f), det, vad, cfg)["windows"]]
    return out


def expand(pats):
    return [f for p in pats for f in (sorted(glob.glob(p)) or [p])]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--save-as", required=True)
    ap.add_argument("--genuine", nargs="+", required=True)
    ap.add_argument("--fake", nargs="*", default=[])
    ap.add_argument("--note", default="")
    a = ap.parse_args()

    cfg = load_config()
    det, vad = load_detector(cfg), VadModel(cfg.models / cfg.vad_file)
    gen_files, fake_files = expand(a.genuine), expand(a.fake)
    g = window_llrs(gen_files, det, vad, cfg)
    print(f"detector: {det.mode}")
    print(f"genuine: {len(gen_files)} files, {len(g)} windows, llr median {np.median(g):.2f}, 95th pct {np.percentile(g, 95):.2f}")
    info = {"source": a.note or f"{len(gen_files)} genuine" + (f" + {len(fake_files)} fake" if fake_files else "") + " files",
            "n_genuine_windows": len(g)}
    if fake_files:
        f = window_llrs(fake_files, det, vad, cfg)
        thr, eer = threshold_eer(g, f)
        print(f"fake:    {len(fake_files)} files, {len(f)} windows, llr median {np.median(f):.2f}")
        print(f"window-level EER {eer:.1f}% at llr {thr:.2f}")
        info.update(method="window-level EER on dev recordings", n_fake_windows=len(f), dev_window_eer=round(eer, 2))
    else:
        thr = threshold_from_genuine(g)
        info.update(method="genuine 95th percentile + 1.5", genuine_median_llr=round(float(np.median(g)), 3))
    profiles = profiles_for(cfg, det)
    saved = profiles.save(a.save_as, thr, **info)
    print(f"saved profile '{a.save_as}' in {profiles.path.name}: {saved}")


if __name__ == "__main__":
    main()
