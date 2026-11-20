"""Speech-to-text for the live transcript (faster-whisper, multilingual Whisper).

GPU (CUDA) is used when NVIDIA's cuBLAS/cuDNN runtime is installed (pip packages
nvidia-cublas-cu12 and nvidia-cudnn-cu12); otherwise it falls back to CPU with a smaller model.
Only speech found by the VAD is transcribed, which avoids most of Whisper's made-up text on silence.
"""
import logging
import os
import site
import sys
import time
from pathlib import Path

import numpy as np

from .config import AsrConfig

log = logging.getLogger("voiceshield.asr")
for _noisy in ("faster_whisper", "httpx", "huggingface_hub"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


def _load(size: str, **kw):
    """Use the downloaded model without touching the internet; download only if it is missing."""
    from faster_whisper import WhisperModel
    try:
        return WhisperModel(size, local_files_only=True, **kw)
    except Exception:
        return WhisperModel(size, **kw)

# phrases Whisper is known to invent on noise or very short audio
HALLUCINATIONS = {"", "you", "thank you", "thank you.", "thanks for watching!", "thank you for watching.",
                  "thank you for watching!", "bye.", "bye", "please subscribe", "subtitles by the amara.org community"}
LANGUAGES = {"en": "English", "hi": "Hindi", "te": "Telugu"}


def _add_nvidia_dlls() -> None:
    """pip's nvidia-* wheels put their DLLs in site-packages/nvidia/*/bin; make Windows find them."""
    if sys.platform != "win32":
        return
    for sp in set(site.getsitepackages() + [site.getusersitepackages()]):
        for b in (Path(sp) / "nvidia").glob("*/bin"):
            os.add_dll_directory(str(b))
            os.environ["PATH"] = str(b) + os.pathsep + os.environ.get("PATH", "")


class Transcriber:
    def __init__(self, cfg: AsrConfig, download_root: Path):
        self.device = None
        if cfg.device in ("auto", "cuda"):
            _add_nvidia_dlls()
            try:
                m = _load(cfg.model, device="cuda", compute_type="float16", download_root=str(download_root))
                m.transcribe(np.zeros(8000, np.float32), beam_size=1, language="en")   # loads the CUDA libraries now
                self.model, self.device, self.size = m, "cuda", cfg.model
            except Exception as e:
                if cfg.device == "cuda":
                    raise
                log.info("speech-to-text: GPU not usable (%s); using CPU", str(e)[:120])
        if self.device is None:
            size = cfg.cpu_model or cfg.model
            self.model = _load(size, device="cpu", compute_type="int8", download_root=str(download_root),
                               cpu_threads=cfg.cpu_threads)
            self.device, self.size = "cpu", size
        log.info("speech-to-text: Whisper %s on %s", self.size, self.device)

    def transcribe(self, wav: np.ndarray, language: str | None = None) -> dict:
        t0 = time.perf_counter()
        segs, info = self.model.transcribe(np.asarray(wav, np.float32), language=language or None, beam_size=1,
                                           condition_on_previous_text=False, vad_filter=False,
                                           without_timestamps=True, temperature=0.0)
        parts = [s.text.strip() for s in segs if s.no_speech_prob < 0.6 and s.avg_logprob > -1.2]
        text = " ".join(p for p in parts if p)
        if text.lower().strip() in HALLUCINATIONS:
            text = ""
        return {"text": text, "language": info.language, "language_prob": round(info.language_probability, 2),
                "latency_ms": round((time.perf_counter() - t0) * 1000)}
