"""Why does a recording look fake to the detector? Checks the channel, not the voice.

    python tools/diagnose_audio.py voiceshield_mic_2026-11-10-00-10-00.wav

Prints level/clipping, how much of the spectrum carries sound (Bluetooth hands-free mics cut
everything above 4 kHz, noise-suppression DSP leaves tell-tale gaps), and the detector's
median score on the file as-is and after a few channel changes.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt, welch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.config import SR, load_config            # noqa: E402
from server.models import Detector, VadModel         # noqa: E402
from server.offline import analyse, load_audio       # noqa: E402


def bandwidth(wav: np.ndarray, floor_db: float = 60.0) -> tuple[float, float]:
    """(highest frequency still within floor_db of the spectral peak, dB of energy above 4 kHz)."""
    f, p = welch(wav, SR, nperseg=1024)
    rel = 10 * np.log10(p / p.max() + 1e-20)
    edge = float(f[np.nonzero(rel > -floor_db)[0].max()])
    hi_db = 10 * np.log10(p[f > 4000].sum() / p.sum() + 1e-20)
    return edge, hi_db


def phone(wav):
    from scipy.signal import resample_poly
    x = sosfilt(butter(4, [300, 3400], btype="band", fs=SR, output="sos"), wav)
    x = resample_poly(x, 1, 2)
    x = x / (np.abs(x).max() + 1e-9) * 0.9
    y = np.round(np.sign(x) * np.log1p(255 * np.abs(x)) / np.log1p(255) * 127) / 127
    x = np.sign(y) * ((1 + 255) ** np.abs(y) - 1) / 255
    return resample_poly(x, 2, 1).astype(np.float32)


def main() -> None:
    cfg = load_config()
    det, vad = Detector(cfg.models / cfg.detector_file, cfg.detector_threads), VadModel(cfg.models / cfg.vad_file)
    for path in sys.argv[1:]:
        wav = load_audio(path)
        peak = 20 * np.log10(np.abs(wav).max() + 1e-9)
        rms = 20 * np.log10(np.sqrt(np.mean(wav ** 2)) + 1e-9)
        clip = 100 * np.mean(np.abs(wav) > 0.99)
        edge, hi_db = bandwidth(wav)
        print(f"\n{path}\n  {len(wav) / SR:.1f} s | peak {peak:.1f} dBFS | rms {rms:.1f} dBFS | clipped {clip:.2f}% of samples")
        print(f"  sound present up to {edge:.0f} Hz | energy above 4 kHz: {hi_db:.0f} dB of total")
        if edge < 4500 or hi_db < -40:
            print("  -> NARROWBAND: nothing above ~4 kHz. Typical of a Bluetooth hands-free mic or an 8 kHz phone line")
        if clip > 0.05:
            print("  -> CLIPPING: the mic is overloaded; speak further away or lower the input volume")
        if rms < -45:
            print("  -> VERY QUIET: raise the mic input volume or speak closer")
        for name, x in [("as recorded", wav), ("low-pass 7 kHz", sosfilt(butter(8, 7000, fs=SR, output="sos"), wav)),
                        ("phone line (300-3400 Hz, mu-law)", phone(wav))]:
            r = analyse(np.asarray(x, np.float32), det, vad, cfg)
            ll = [w["llr"] for w in r["windows"]]
            if ll:
                print(f"  {name:34s} windows {len(ll):3d}  median llr {np.median(ll):6.2f}  "
                      f"(real speech is usually about -10 to -6)")
            else:
                print(f"  {name:34s} no speech windows (speech {r['speech_s']:.1f} s)")


if __name__ == "__main__":
    main()
