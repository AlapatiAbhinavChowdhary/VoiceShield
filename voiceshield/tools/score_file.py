"""Score audio files with the live pipeline (no server needed).

    python tools/score_file.py samples/genuine_call_1.flac samples/cloned_call_elevenlabs.flac
    python tools/score_file.py --profile studio --timeline my_call.wav
    python tools/score_file.py --csv results.csv folder_of_clips/*.wav
"""
import argparse
import csv
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.calibration import profiles_for      # noqa: E402
from server.config import load_config            # noqa: E402
from server.models import VadModel, load_detector  # noqa: E402
from server.offline import analyse, load_audio   # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--profile", default="default", help="line profile from data/profiles.json")
    ap.add_argument("--timeline", action="store_true", help="print every analysed window")
    ap.add_argument("--csv", help="write one row per file")
    a = ap.parse_args()

    cfg = load_config()
    det, vad = load_detector(cfg), VadModel(cfg.models / cfg.vad_file)
    name, prof = profiles_for(cfg, det).get(a.profile)
    thr, auto = prof["llr_threshold"], bool(prof.get("auto"))
    files = [f for pat in a.files for f in (sorted(glob.glob(pat)) or [pat])]
    print(f"detector {det.mode}, profile '{name}' (llr threshold {'auto' if auto else f'{thr:.2f}'}), {len(files)} file(s)\n")
    rows = []
    for f in files:
        res = analyse(load_audio(f), det, vad, cfg, thr, auto=auto)
        if a.timeline:
            for w in res["windows"]:
                print(f"   t={w['t']:6.1f}s  speech={w['speech_s']:5.1f}s  llr={w['llr']:6.2f}  "
                      f"risk={w['risk']:5.1f}  {w['level']}")
        llrs = [w["llr"] for w in res["windows"]]
        row = {"file": f, "verdict": res["verdict"], "max_risk": round(res["max_risk"], 1),
               "windows": len(llrs), "mean_llr": round(sum(llrs) / len(llrs), 2) if llrs else None,
               "audio_s": round(res["audio_s"], 1), "speech_s": round(res["speech_s"], 1),
               "first_alert_s": res["first_alert_s"] and round(res["first_alert_s"], 1),
               "first_hold_s": res["first_hold_s"] and round(res["first_hold_s"], 1)}
        rows.append(row)
        print(f"{row['verdict']:6s}  max risk {row['max_risk']:5.1f}  windows {row['windows']:3d}  "
              f"mean llr {row['mean_llr']}  first alert {row['first_alert_s']}s  HOLD at {row['first_hold_s']}s  {f}")
    if a.csv and rows:
        with open(a.csv, "w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
        print("\nwrote", a.csv)


if __name__ == "__main__":
    main()
