"""Rebuild the demo calls in samples/ and the 'studio' line profile from public data.

    python tools/build_demo.py
    python tools/build_demo.py --profile-only     (only re-calibrate 'studio', e.g. after switching detector_mode)

The 'studio' profile is saved for the detector chosen in config.json ("detector_mode").

Source: garystafford/deepfake-audio-detection (CC BY 4.0), split exactly as in Step 4:
  calibration (dev) only : fake hu_* (Hume)           real yt_0013, yt_0001, yt_0003, yt_0007
  demo calls (test) only : fake el_* (ElevenLabs), hg_* (Kokoro)    real yt_0009, yt_0010
No demo recording is ever used to set a threshold. Each demo call is the first N clips of
one recording in file order (no cherry-picking), joined with short pauses.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.calibration import profiles_for, threshold_eer  # noqa: E402
from server.config import ROOT, SR, load_config          # noqa: E402
from server.models import VadModel, load_detector        # noqa: E402
from server.offline import analyse, load_audio           # noqa: E402

REPO = "garystafford/deepfake-audio-detection"
DEV_FAKE = ("fake/hu_",)
DEV_REAL = ("real/yt_0013", "real/yt_0001", "real/yt_0003", "real/yt_0007")
N_PARTS = 8
GAP_S = 0.4
DEMO = [  # file, label, description, kind, clip prefix
    ("genuine_call_1.flac", "Genuine caller 1", "Real recording, speaker not used in calibration", "genuine",
     "real/yt_0009_p2_part_2"),
    ("genuine_call_2.flac", "Genuine caller 2", "Real recording, speaker not used in calibration", "genuine",
     "real/yt_0010_part_0"),
    ("cloned_call_elevenlabs.flac", "Cloned voice: ElevenLabs", "Commercial voice clone, service not used in calibration",
     "cloned", "fake/el_0001_c_part_0"),
    ("cloned_call_kokoro.flac", "Cloned voice: Kokoro TTS", "Open-source TTS, not used in calibration", "cloned",
     "fake/hg_0001_part_0"),
]

SAMPLES_README = """# Demo calls

Built by `tools/build_demo.py` from **garystafford/deepfake-audio-detection** (CC BY 4.0,
https://huggingface.co/datasets/garystafford/deepfake-audio-detection). Each file joins the first
{n} clips of one recording (16 kHz mono, {gap} s pauses). Real speakers yt_0009 / yt_0010 and fake
services ElevenLabs / Kokoro are the Step 4 *test* split: they were never used to train or calibrate
anything. The `studio` line profile was calibrated on the *dev* split (Hume fakes, real yt_0013,
yt_0001, yt_0003, yt_0007).
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=str(ROOT / "data" / "demo_source"), help="download folder")
    ap.add_argument("--profile-only", action="store_true", help="keep samples/ as it is; only calibrate 'studio'")
    a = ap.parse_args()
    cache = Path(a.cache)

    files = [s.rfilename for s in HfApi().repo_info(REPO, repo_type="dataset").siblings]
    dev_real = sorted(f for f in files if f.startswith(DEV_REAL) and f.endswith(".flac"))
    dev_fake = sorted(f for f in files if f.startswith(DEV_FAKE) and f.endswith(".flac"))
    demo_parts = {d[0]: sorted(f for f in files if f.startswith(d[4]) and f.endswith(".flac"))[:N_PARTS] for d in DEMO}

    def get(f: str) -> Path:
        p = cache / f
        if not p.exists():
            hf_hub_download(REPO, f, repo_type="dataset", local_dir=cache)
        return p

    print(f"downloading {len(dev_real)} + {len(dev_fake)} dev clips"
          + ("" if a.profile_only else f" and {sum(map(len, demo_parts.values()))} demo clips") + " ...")
    dev_real_p = [get(f) for f in dev_real]
    dev_fake_p = [get(f) for f in dev_fake]

    out = ROOT / "samples"
    out.mkdir(exist_ok=True)
    meta = []
    for fname, label, desc, kind, _ in ([] if a.profile_only else DEMO):
        gap = np.zeros(int(GAP_S * SR), np.float32)
        parts = [load_audio(get(f)) for f in demo_parts[fname]]
        wav = np.concatenate([x for p in parts for x in (p, gap)])
        sf.write(out / fname, wav, SR, format="FLAC")
        meta.append({"file": fname, "label": label, "description": f"{desc} · {len(wav) / SR:.0f} s",
                     "kind": kind, "profile": "studio", "parts": demo_parts[fname]})
        print(f"  {fname}: {len(parts)} clips, {len(wav) / SR:.1f} s")
    if not a.profile_only:
        old = json.loads((out / "samples.json").read_text(encoding="utf-8")) if (out / "samples.json").exists() else []
        ours = {d[0] for d in DEMO}
        meta += [s for s in old if s["file"] not in ours]      # keep samples added by hand (e.g. scam_call_tts.wav)
        (out / "samples.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        (out / "README.md").write_text(SAMPLES_README.format(n=N_PARTS, gap=GAP_S), encoding="utf-8")

    cfg = load_config()
    det, vad = load_detector(cfg), VadModel(cfg.models / cfg.vad_file)
    print(f"scoring dev clips with the live pipeline (detector {det.mode}) ...")
    g = [w["llr"] for p in dev_real_p for w in analyse(load_audio(p), det, vad, cfg)["windows"]]
    f = [w["llr"] for p in dev_fake_p for w in analyse(load_audio(p), det, vad, cfg)["windows"]]
    thr, eer = threshold_eer(g, f)
    saved = profiles_for(cfg, det).save(
        "studio", thr, source=f"dev split of {REPO}: {len(dev_real)} real clips (4 speakers), {len(dev_fake)} Hume fakes",
        method="window-level EER on dev recordings", n_genuine_windows=len(g), n_fake_windows=len(f),
        dev_window_eer=round(eer, 2), genuine_median_llr=round(float(np.median(g)), 2),
        fake_median_llr=round(float(np.median(f)), 2))
    print(f"studio profile: cut-off {thr:.2f} (dev window EER {eer:.1f}%) -> {saved}")


if __name__ == "__main__":
    main()
