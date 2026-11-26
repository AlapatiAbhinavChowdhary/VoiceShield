"""Scenario lab: repeatable demonstrations of the whole decision chain, isolated from live calls.

Every scenario states what is real and what is simulated:
  - "real"       a real recording through the real detector (+ real speech-to-text where marked);
  - "simulated"  a real recording with a simulated audio condition added (noise, phone line), real detector;
  - "transcript" a scripted transcript through the real scam-language engine and risk engine; no audio, the
                 caller's voice is taken as a genuine human (this is how a human scammer looks to the system).
Results never touch the audit log, and the server refuses lab runs while a live call is being analysed.
"""
import time
from pathlib import Path

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

from .config import ROOT, SR
from .offline import analyse, load_audio
from .quality import audio_quality
from .risk import LEVELS, RiskEngine
from .scam import CallScam

SAMPLES = ROOT / "samples"

SCENARIOS = [
    {"id": "genuine-call", "title": "Legitimate customer call", "truth": "genuine", "mode": "real",
     "file": "genuine_call_1.flac", "story": "A real person calls about their account. Nothing should interrupt the payment."},
    {"id": "clone-elevenlabs", "title": "AI-cloned voice (ElevenLabs)", "truth": "fraud", "mode": "real",
     "file": "cloned_call_elevenlabs.flac", "story": "A premium commercial voice clone, never seen in training or calibration."},
    {"id": "clone-kokoro", "title": "AI-cloned voice (Kokoro)", "truth": "fraud", "mode": "real",
     "file": "cloned_call_kokoro.flac", "story": "An open-weights TTS voice that sounds natural for its first seconds."},
    {"id": "fake-officer-tts", "title": "Fake bank officer: synthetic voice + scam script", "truth": "fraud", "mode": "real",
     "file": "scam_call_tts.wav", "asr": True,
     "story": "A synthetic voice claims to be from the bank, threatens a block and asks for the OTP (real speech-to-text)."},
    {"id": "noisy-genuine", "title": "Genuine caller in a noisy place", "truth": "genuine", "mode": "simulated",
     "file": "genuine_call_2.flac", "effect": ["noise", 10],
     "story": "The same kind of real caller with steady background noise added at 10 dB SNR."},
    {"id": "noisy-clone", "title": "Voice clone in a noisy place", "truth": "fraud", "mode": "simulated",
     "file": "cloned_call_elevenlabs.flac", "effect": ["noise", 10],
     "story": "The ElevenLabs clone with the same noise: noise can hide synthesis artefacts."},
    {"id": "phone-genuine", "title": "Genuine caller over a phone line", "truth": "genuine", "mode": "simulated",
     "file": "genuine_call_1.flac", "effect": ["phone"],
     "story": "A real caller through a simulated narrow-band line (300-3400 Hz, 8 kHz, G.711 mu-law)."},
    {"id": "human-bank-script", "title": "Human fraudster: fake bank KYC script", "truth": "fraud", "mode": "transcript",
     "utterances": ["Hello, I am calling from SBI customer care.", "Your KYC is not updated, your account will be blocked today.",
                    "Please share the OTP you just received."],
     "story": "A real human voice, so the voice check passes; only the words give the scam away."},
    {"id": "human-family", "title": "Human fraudster: family emergency (Hinglish)", "truth": "fraud", "mode": "transcript",
     "utterances": ["Papa main accident mein hoon, hospital mein hoon.", "Mujhe turant paise bhej do, Google Pay kar do.",
                    "Mummy ko mat batana."],
     "story": "An urgent request for money in the voice of a relative, in Hinglish."},
    {"id": "human-arrest", "title": "Human fraudster: digital arrest", "truth": "fraud", "mode": "transcript",
     "utterances": ["This is the CBI crime branch.", "A parcel in your name has drugs, a case has been registered.",
                    "To avoid arrest you must transfer money for verification immediately.", "Do not tell anyone about this call."],
     "story": "A fake officer threatens arrest and demands a transfer and secrecy."},
    {"id": "bank-warning", "title": "Bank's genuine fraud warning", "truth": "genuine", "mode": "transcript",
     "utterances": ["Hello, this is your bank's customer care.", "We will never ask for your OTP or PIN.",
                    "Please do not share your OTP with anyone, even if they say they are from the bank."],
     "story": "A real bank reminding the customer not to share OTPs: it should not be treated as a scam."},
]
BY_ID = {s["id"]: s for s in SCENARIOS}

PAYMENT = {"PASS": "Allowed", "VERIFY": "Held for OTP or call-back", "HOLD": "Blocked · security team alerted"}


def public(s: dict) -> dict:
    return {k: v for k, v in s.items() if k not in ("utterances",)} | {
        "inference": {"real": "Real recording · real model inference",
                      "simulated": "Real recording + simulated audio condition · real model inference",
                      "transcript": "Scripted transcript · real scam-language and risk engines (no audio)"}[s["mode"]],
        "expected": "PASS" if s["truth"] == "genuine" else "VERIFY or HOLD"}


def add_noise(x: np.ndarray, snr_db: float) -> np.ndarray:
    speech = x[np.abs(x) > 0.02 * np.abs(x).max()] if np.abs(x).max() > 0 else x
    p = float(np.mean(speech ** 2)) if speech.size else 1e-6
    noise = np.random.default_rng(7).standard_normal(x.size).astype(np.float32) * np.sqrt(p / 10 ** (snr_db / 10))
    return np.clip(x + noise, -1, 1).astype(np.float32)


def phone_line(x: np.ndarray) -> np.ndarray:
    y = sosfiltfilt(butter(6, [300, 3400], btype="band", fs=SR, output="sos"), x)
    y = resample_poly(y, 1, 2)                                        # 8 kHz
    mu = 255.0
    c = np.sign(y) * np.log1p(mu * np.clip(np.abs(y), 0, 1)) / np.log1p(mu)
    c = np.round(c * 127) / 127                                       # 8-bit companded
    y = np.sign(c) * np.expm1(np.abs(c) * np.log1p(mu)) / mu
    return resample_poly(y, 2, 1).astype(np.float32)


def run(s: dict, detector, vad, cfg, threshold: float, transcriber=None) -> dict:
    t0 = time.perf_counter()
    if s["mode"] == "transcript":
        risk, scam = RiskEngine(cfg.risk, threshold), CallScam(context=cfg.scam_context)
        timeline, transcript, first = [], [], {}
        for i, u in enumerate(s["utterances"]):
            t = 3.0 * (i + 1)
            r = scam.add(u)
            risk.set_scam(r["score"], r["tactics"])
            st = risk.state()
            first.setdefault(st["level"], t)
            timeline.append({"t": t, "risk": st["risk"], "level": st["level"]})
            transcript.append({"t": t, "text": u, "tactics": [h["label"] for h in r["hits"]],
                               "suppressed": [x["phrase"] for x in r["suppressed"]]})
        verdict, st = LEVELS[risk.verdict], risk.state()
        out = {"verdict": verdict, "max_risk": round(risk.max_risk, 1), "signals": st["signals"], "reasons": st["reasons"],
               "timeline": timeline, "transcript": transcript, "patterns": r["patterns"], "quality": None,
               "first_verify_s": first.get("VERIFY", first.get("HOLD")), "first_hold_s": first.get("HOLD"), "windows": 0}
    else:
        x = load_audio(SAMPLES / s["file"])
        eff = s.get("effect")
        if eff and eff[0] == "noise":
            x = add_noise(x, eff[1])
        elif eff and eff[0] == "phone":
            x = phone_line(x)
        a = analyse(x, detector, vad, cfg, threshold, transcriber=transcriber if s.get("asr") else None,
                    scam_context=cfg.scam_context, keep_windows=True)
        W = a["windows"]
        qs = [audio_quality(w) for w in a.get("audio_windows", [])]
        labels = [q["label"] for q in qs]
        # explain the decision at the moment it was reached, not the state at the end of the call
        st = next((w for w in W if w["level"] == a["verdict"]), W[-1]) if W else \
            {"signals": {"voice": 0, "identity": None, "scam": 0}, "reasons": []}
        out = {"verdict": a["verdict"], "max_risk": round(a["max_risk"], 1), "signals": st.get("signals", a["signals"]),
               "reasons": st.get("reasons", []), "windows": len(W), "decided_at_s": round(st["t"], 2) if W else None,
               "timeline": [{"t": round(w["t"], 2), "risk": w["risk"], "level": w["level"], "llr": round(w["llr"], 2),
                             "window_score": round(w["window_score"], 3)} for w in W],
               "transcript": [{"t": round(u["t1"], 2), "text": u["text"], "tactics": [h["label"] for h in u["hits"]]}
                              for u in a["transcript"]],
               "patterns": a.get("patterns", []),
               "quality": {"good": labels.count("good"), "fair": labels.count("fair"), "poor": labels.count("poor"),
                           "median_snr_db": float(np.median([q["snr_db"] for q in qs])) if qs else None},
               "flagged": sum(w["window_score"] >= 0.5 for w in W),
               "median_llr": round(float(np.median([w["llr"] for w in W])), 2) if W else None,
               "first_verify_s": a["first_alert_s"], "first_hold_s": a["first_hold_s"], "audio_s": round(a["audio_s"], 1)}
    out["correct"] = (out["verdict"] == "PASS") if s["truth"] == "genuine" else (out["verdict"] in ("VERIFY", "HOLD"))
    out["payment"] = PAYMENT[out["verdict"]]
    out["threshold"] = round(threshold, 3)
    out["elapsed_s"] = round(time.perf_counter() - t0, 2)
    return out
