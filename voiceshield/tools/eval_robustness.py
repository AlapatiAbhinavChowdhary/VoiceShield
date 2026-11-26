"""Robustness suite: genuine and spoofed calls under noise / phone conditions, scored once by the real detector.

    python tools/eval_robustness.py --build     score every recording x condition (needs the ONNX models, ~15 min CPU)
    python tools/eval_robustness.py --report    replay the stored windows through decision rules and compare them

--build writes eval/robustness_windows.json (scores and quality per window; no audio). --report needs no models.

Recordings (labels are known by construction):
  genuine  demo genuine callers 1-2 (test-split speakers), 4 dev-split speakers (compilations of their clips),
           2 raw laptop-microphone sessions of a team member (local only, not in the repository)
  spoof    ElevenLabs and Kokoro demo clones, Windows-TTS scam call, kNN-VC clone of an enrolled member (Step 9),
           4 compilations of Hume dev-split fakes
Conditions: clean; white noise at 20 / 10 / 5 dB SNR; babble (3 other speakers) at 10 dB; phone line (G.711);
            phone line + white noise at 15 dB. Synthetic conditions on real recordings; dev-split audio was used
            to calibrate the `studio` profile, so this is a controlled suite, not an independent benchmark.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from server.config import SR, load_config  # noqa: E402

OUT = ROOT / "eval" / "robustness_windows.json"
CONDITIONS = ["clean", "white20", "white10", "white5", "babble10", "phone", "phone_white15"]


def compile_clips(files, seconds=40.0):
    from server.offline import load_audio
    parts, n = [], 0
    for f in files:
        x = load_audio(f)
        x = x / max(1e-6, np.abs(x).max()) * 0.5
        parts += [x, np.zeros(int(0.3 * SR), np.float32)]
        n += len(x)
        if n >= seconds * SR:
            break
    return np.concatenate(parts).astype(np.float32)


def recordings():
    from server.offline import load_audio
    s, rec, src = ROOT / "samples", ROOT.parent / "recordings", ROOT / "data" / "demo_source"
    out = []
    for f, lab, kind in [(s / "genuine_call_1.flac", "genuine", "demo caller (test split)"),
                         (s / "genuine_call_2.flac", "genuine", "demo caller (test split)"),
                         (rec / "laptop_mic_enhancements_OFF_1.wav", "genuine", "raw laptop mic (local)"),
                         (rec / "laptop_mic_enhancements_OFF_2.wav", "genuine", "raw laptop mic (local)"),
                         (s / "cloned_call_elevenlabs.flac", "spoof", "ElevenLabs clone"),
                         (s / "cloned_call_kokoro.flac", "spoof", "Kokoro clone"),
                         (s / "scam_call_tts.wav", "spoof", "Windows TTS scam call"),
                         (rec / "step9_clone_of_enrolled_user.wav", "spoof", "kNN-VC clone of a member (local)")]:
        if f.exists():
            out.append({"id": f.stem, "label": lab, "kind": kind, "audio": load_audio(f)})
    real = sorted((src / "real").glob("*.flac"))
    for spk in sorted({f.name[:7] for f in real}):
        out.append({"id": f"dev_{spk}", "label": "genuine", "kind": "dev-split speaker",
                    "audio": compile_clips([f for f in real if f.name.startswith(spk)])})
    fake = sorted((src / "fake").glob("*.flac"))
    groups = sorted({f.name[:7] for f in fake})
    for g in groups[:4]:
        out.append({"id": f"hume_{g}", "label": "spoof", "kind": "Hume dev-split fake",
                    "audio": compile_clips([f for f in fake if f.name.startswith(g)])})
    babble_pool = [compile_clips([f for f in real if f.name.startswith(spk)], 20) for spk in sorted({f.name[:7] for f in real})]
    return out, babble_pool


def condition(x, cond, babble_pool, rid):
    from server.lab import add_noise, phone_line
    if cond == "clean":
        return x
    if cond.startswith("white") and cond[5:].isdigit():
        return add_noise(x, float(cond[5:]))
    if cond == "babble10":
        rng = np.random.default_rng(abs(hash(rid)) % 2 ** 32)
        pool = [b for i, b in enumerate(babble_pool) if f"dev_yt_{i}" != rid]
        b = sum(np.resize(pool[i], x.size) for i in rng.choice(len(pool), size=min(3, len(pool)), replace=False))
        px, pb = np.mean(x[np.abs(x) > 0.02] ** 2) if np.any(np.abs(x) > 0.02) else 1e-6, np.mean(b ** 2) + 1e-12
        return np.clip(x + b * np.sqrt(px / pb / 10 ** (10 / 10)), -1, 1).astype(np.float32)
    if cond == "phone":
        return phone_line(x)
    if cond == "phone_white15":
        return add_noise(phone_line(x), 15)
    raise ValueError(cond)


def build():
    from server.models import VadModel, load_detector
    from server.offline import make_processor
    from server.quality import audio_quality
    cfg = load_config()
    det, vad = load_detector(cfg), VadModel(cfg.models / cfg.vad_file)
    recs, babble = recordings()
    res = {"built": time.strftime("%Y-%m-%d %H:%M"), "detector_mode": getattr(det, "mode", "original"),
           "conditions": CONDITIONS, "recordings": []}
    t0 = time.time()
    for r in recs:
        entry = {k: r[k] for k in ("id", "label", "kind")} | {"conditions": {}}
        for cond in CONDITIONS:
            x = condition(r["audio"], cond, babble, r["id"])
            proc, wins = make_processor(vad, cfg), []
            for w in proc.feed(x):
                s = det.score(w)
                q = audio_quality(w)
                wins.append({"t": round(proc.seconds, 2), "llr": round(float(s["llr"]), 3), "q": q["label"],
                             "snr": q["snr_db"], "proc": s.get("processing")})
            entry["conditions"][cond] = wins
            print(f"{time.time() - t0:7.0f}s  {r['id']:38s} {cond:14s} {len(wins):3d} windows", flush=True)
        res["recordings"].append(entry)
        OUT.parent.mkdir(exist_ok=True)
        OUT.write_text(json.dumps(res), encoding="utf-8")            # partial results survive an interruption
    print("done", OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.build:
        build()
    if a.report:
        from eval_rules import report  # noqa: E402  (tools/eval_rules.py)
        report(OUT)
