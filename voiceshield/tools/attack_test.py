"""Step 9 attack test, offline: run the full live pipeline (detector + voiceprint + transcript + scam rules) on
recordings while the caller CLAIMS to be an enrolled person, with the original detector and with Step 8c.

    python tools/attack_test.py --claim "Abhinav Chowdhary Alapati" ../recordings/step9_clone_of_enrolled_user.wav ...

The interesting case is a clone OF THE ENROLLED PERSON: the voiceprint matches, so the decision has to come
from the AI-voice detector and the scam language.
"""
import argparse
import gc
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.asr import Transcriber                               # noqa: E402
from server.calibration import Profiles, profiles_for             # noqa: E402
from server.config import load_config                             # noqa: E402
from server.models import Detector, LayerBackend, VadModel        # noqa: E402
from server.offline import analyse, load_audio                    # noqa: E402
from server.speaker import SpeakerModel, Voiceprints              # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--claim", required=True, help="enrolled person the caller claims to be")
    ap.add_argument("--language", default="en")
    a = ap.parse_args()
    cfg = load_config()
    enrolled = Voiceprints(cfg.path(cfg.voiceprints_file)).get(a.claim)
    if enrolled is None:
        sys.exit(f"{a.claim!r} is not enrolled")
    vad, spk = VadModel(cfg.models / cfg.vad_file), SpeakerModel(cfg.models / cfg.speaker.model_file)
    asr = Transcriber(cfg.asr, cfg.models / "whisper")
    wavs = {f: load_audio(f) for f in a.files}
    setups = [("original", lambda: Detector(cfg.models / cfg.detector_file, cfg.detector_threads),
               lambda d: Profiles(cfg.path(cfg.profiles_file), cfg.risk.llr_threshold)),
              ("step8c", lambda: Detector(cfg.models / cfg.layers_file, cfg.detector_threads, backend=LayerBackend(cfg.models / cfg.backend_file)),
               lambda d: profiles_for(cfg, d))]
    for name, make, profs in setups:
        det = make()                                   # one 1.26 GB detector in memory at a time
        thr = profs(det).get("default")[1]["llr_threshold"]
        print(f"\n=== detector {name} · profile default (cut-off {thr:.2f}) · caller claims to be {a.claim}")
        for f, wav in wavs.items():
            r = analyse(wav, det, vad, cfg, thr, speaker=spk, enrolled=enrolled, transcriber=asr, language=a.language)
            w = r["windows"]
            sims = [x["speaker"]["similarity"] for x in w if x.get("speaker")]
            flagged = sum(x["window_score"] >= 0.5 for x in w)
            tactics = sorted({h["label"] for t in r["transcript"] for h in t["hits"]})
            print(f"{Path(f).name}: {r['verdict']} (max risk {r['max_risk']:.0f})"
                  + (f", first alert {r['first_alert_s']:.1f} s" if r["first_alert_s"] else "")
                  + (f", HOLD at {r['first_hold_s']:.1f} s" if r["first_hold_s"] else ""))
            print(f"   voiceprint similarity mean {np.mean(sims):.2f} (min {min(sims):.2f}) -> "
                  f"{'MATCHES' if np.mean(sims) >= cfg.speaker.threshold else 'does not match'} the enrolled voice")
            print(f"   detector: median score {np.median([x['llr'] for x in w]):.2f}, windows flagged {flagged}/{len(w)}")
            print(f"   scam tactics: {', '.join(tactics) or 'none'} · signals at the end {r['signals']}")
            print(f"   transcript: {' '.join(t['text'] for t in r['transcript'])[:160]}")
        del det
        gc.collect()


if __name__ == "__main__":
    main()
