"""Enroll a person's voiceprint from recordings instead of the live microphone.

    python tools/enroll_file.py "Abhinav" recordings/my_voice.wav [more.wav ...]

Uses the same windows and averaging as the console's "Enroll voice". Replaces an existing
voiceprint with the same name. Only the 256-number voiceprint is stored, not the audio.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.config import load_config                 # noqa: E402
from server.models import VadModel                    # noqa: E402
from server.offline import load_audio, make_processor  # noqa: E402
from server.speaker import SpeakerModel, Voiceprints, cosine   # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name")
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    cfg = load_config()
    vad, spk = VadModel(cfg.models / cfg.vad_file), SpeakerModel(cfg.models / cfg.speaker.model_file)
    embs, speech = [], 0.0
    for f in a.files:
        proc = make_processor(vad, cfg)
        wav = load_audio(f)
        for i in range(0, len(wav), 1600):
            for w in proc.feed(wav[i:i + 1600]):
                embs.append(spk.embed(w))
        speech += proc.speech_seconds
    if len(embs) < cfg.speaker.min_enroll_windows:
        sys.exit(f"only {len(embs)} windows of speech; need {cfg.speaker.min_enroll_windows} (about 7 s)")
    emb = np.mean(embs, axis=0)
    emb /= np.linalg.norm(emb)
    store = Voiceprints(cfg.path(cfg.voiceprints_file))
    old = store.get(a.name)
    saved = store.save(a.name, emb, speech, "file:" + ", ".join(Path(f).name for f in a.files))
    print(f"saved voiceprint '{a.name}' from {len(embs)} windows ({speech:.1f} s of speech): {saved}")
    if old is not None:
        print(f"similarity to the previous voiceprint: {cosine(old, emb):.2f}")


if __name__ == "__main__":
    main()
