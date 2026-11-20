"""Original detector vs Step 8c (frozen detector + back-end) on real recordings, window by window.

    python tools/compare_step8c.py ../recordings/laptop_mic_enhancements_*.wav samples/*.flac samples/*.wav

Prints, per file: median and 95th percentile of each score, windows above each detector's default
threshold, the processing type the Step 8c classifier sees, and how far the two models' own (head)
scores differ (they share the same weights, so this should be ~0).
"""
import argparse
import collections
import csv
import glob
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.config import load_config                                   # noqa: E402
from server.models import Detector, LayerBackend, VadModel              # noqa: E402
from server.offline import load_audio, make_processor                   # noqa: E402


def windows(path, vad, cfg):
    proc, wav, out = make_processor(vad, cfg), load_audio(path), []
    for i in range(0, len(wav), 1600):
        out += proc.feed(wav[i:i + 1600])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--csv", help="write one row per window")
    a = ap.parse_args()
    cfg = load_config()
    vad = VadModel(cfg.models / cfg.vad_file)
    old = Detector(cfg.models / cfg.detector_file, cfg.detector_threads)
    new = Detector(cfg.models / cfg.layers_file, cfg.detector_threads, backend=LayerBackend(cfg.models / cfg.backend_file))
    t_old, t_new = cfg.risk.llr_threshold, new.backend.threshold
    print(f"default thresholds: original {t_old:.2f} | Step 8c {t_new:.2f} (llr scale)\n")
    rows, lat_o, lat_n, head_diff = [], [], [], []
    files = [f for p in a.files for f in (sorted(glob.glob(p)) or [p])]
    for f in files:
        ws = windows(f, vad, cfg)
        if not ws:
            print(f"{Path(f).name}: no speech windows")
            continue
        o, n = [], []
        for w in ws:
            t0 = time.perf_counter(); ro = old.score(w); lat_o.append(time.perf_counter() - t0)
            t0 = time.perf_counter(); rn = new.score(w); lat_n.append(time.perf_counter() - t0)
            head_diff.append(abs(ro["llr"] - rn["llr_original"]))
            o.append(ro["llr"]); n.append(rn)
            rows.append({"file": Path(f).name, "llr_original": round(ro["llr"], 3), "llr_step8c": round(rn["llr"], 3),
                         "processing": rn["processing"], "auto_threshold": round(rn["auto_threshold"], 3)})
        g = [r["llr"] for r in n]
        proc = collections.Counter(r["processing"] for r in n).most_common(2)
        print(f"{Path(f).name}  ({len(ws)} windows)")
        print(f"   original: median {np.median(o):6.2f}  95th {np.percentile(o, 95):6.2f}  "
              f"above {t_old:.2f}: {sum(x >= t_old for x in o):3d}/{len(o)}")
        print(f"   Step 8c : median {np.median(g):6.2f}  95th {np.percentile(g, 95):6.2f}  "
              f"above {t_new:.2f}: {sum(x >= t_new for x in g):3d}/{len(g)}  "
              f"above auto: {sum(r['llr'] >= r['auto_threshold'] for r in n):3d}/{len(n)}  "
              f"audio looks like: {', '.join(f'{k} ({v})' for k, v in proc)}")
    print(f"\nsame detector head in both files: max |llr difference| {max(head_diff):.4f} over {len(head_diff)} windows")
    print(f"CPU time per window: original {1000 * np.median(lat_o):.0f} ms, Step 8c {1000 * np.median(lat_n):.0f} ms (median)")
    if a.csv and rows:
        with open(a.csv, "w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
        print("wrote", a.csv)


if __name__ == "__main__":
    main()
