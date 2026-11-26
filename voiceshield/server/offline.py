"""Run the exact live pipeline on a whole file, without a server (for tests, calibration and evaluation)."""
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .config import SR, Config
from .risk import LEVELS, RiskEngine
from .scam import CallScam
from .speaker import CallVoice
from .stream import StreamProcessor


def load_audio(path: str | Path) -> np.ndarray:
    """Any soundfile-readable file -> 16 kHz mono float32."""
    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != SR:
        g = gcd(sr, SR)
        wav = resample_poly(wav, SR // g, sr // g).astype(np.float32)
    return wav


def make_processor(vad, cfg: Config) -> StreamProcessor:
    return StreamProcessor(vad, cfg.window_seconds, cfg.hop_seconds, cfg.min_speech_seconds, cfg.vad_threshold,
                           cfg.vad_hangover_ms, cfg.asr.min_utterance_seconds, cfg.asr.max_utterance_seconds)


def analyse(wav: np.ndarray, detector, vad, cfg: Config, threshold: float | None = None, chunk_s: float = 0.1,
            speaker=None, enrolled: np.ndarray | None = None, transcriber=None, language: str | None = None,
            auto: bool = False, scam_context: bool = False, keep_windows: bool = False) -> dict:
    """Feeds the audio in live-sized chunks and runs every check the live server would run.

    speaker + enrolled -> voiceprint check; transcriber -> transcript + scam language. Offline, the
    transcript of an utterance is available as soon as the utterance ends (live adds ASR latency).
    auto -> Step 8c "auto" line profile: each window uses the threshold of the processing type it shows.
    """
    proc = make_processor(vad, cfg)
    risk = RiskEngine(cfg.risk, threshold)
    voice = CallVoice(enrolled, cfg.speaker.threshold, cfg.speaker.scale) if speaker is not None else None
    scam = CallScam(context=scam_context)
    step = max(1, int(chunk_s * SR))
    windows, transcript, first_hold, first_verify = [], [], None, None
    audio_windows, patterns = [], []

    def note(state):
        nonlocal first_hold, first_verify
        if state["level"] == "HOLD" and first_hold is None:
            first_hold = proc.seconds
        if state["level"] != "PASS" and first_verify is None:
            first_verify = proc.seconds

    def utterances(utts):
        for u, t0, t1 in utts:
            r = transcriber.transcribe(u, language)
            s = scam.add(r["text"])
            patterns[:] = s.get("patterns", [])
            risk.set_scam(s["score"], s["tactics"])
            st = risk.state()
            transcript.append({"t0": t0, "t1": t1, **r, "hits": s["hits"], "scam": s["score"], **st})
            note(st)

    for i in range(0, len(wav), step):
        for w in proc.feed(wav[i:i + step]):
            if keep_windows:
                audio_windows.append(w)
            r = detector.score(w)
            ws = risk.update(r["llr"], r.get("auto_threshold") if auto else None)
            spk = None
            if voice is not None:
                spk = voice.add(speaker.embed(w))
                risk.set_identity(spk and spk["mismatch"])
            st = risk.state()
            windows.append({"t": proc.seconds, "speech_s": proc.speech_seconds, **r, "window_score": ws,
                            "speaker": spk, **st})
            note(st)
        if transcriber is not None:
            utterances(proc.pop_utterances())
    if transcriber is not None:
        utterances(proc.flush())
    return {"windows": windows, "transcript": transcript, "verdict": LEVELS[risk.verdict], "max_risk": risk.max_risk,
            "audio_s": proc.seconds, "speech_s": proc.speech_seconds, "signals": risk.state()["signals"],
            "first_alert_s": first_verify, "first_hold_s": first_hold, "audio_windows": audio_windows, "patterns": patterns}
