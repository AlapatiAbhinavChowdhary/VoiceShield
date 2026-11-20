"""Live audio -> speech-only analysis windows.

Audio arrives in small chunks of any size. Silero VAD marks each 32 ms frame as speech or not.
Only speech frames go into a rolling buffer, and every `hop` seconds of new speech the last
`window` seconds of speech are handed to the detector. Silence never reaches the detector,
which matches how the model was evaluated (silence trimmed).
"""
import numpy as np
from scipy.signal import ellip, sosfilt, sosfilt_zi

from .config import SR


class StreamResampler:
    """Chunk-by-chunk resampler to 16 kHz (anti-alias low-pass + interpolation, state kept between chunks).

    Browsers that support AudioContext({sampleRate: 16000}) never need this; it is the fallback.
    """

    def __init__(self, sr_in: int, sr_out: int = SR):
        self.step = sr_in / sr_out
        # elliptic low-pass: flat to 7.2 kHz, >= 60 dB down soon after, so almost nothing aliases
        self.sos = ellip(8, 0.1, 60, 0.45 * sr_out, btype="low", fs=sr_in, output="sos")
        self.zi = sosfilt_zi(self.sos) * 0.0
        self.buf = np.zeros(0, np.float64)
        self.pos = 0.0          # position (in input samples, relative to buf[0]) of the next output sample

    def __call__(self, x: np.ndarray) -> np.ndarray:
        y, self.zi = sosfilt(self.sos, np.asarray(x, np.float64), zi=self.zi)
        buf = np.concatenate([self.buf, y])
        if len(buf) < 2 or self.pos > len(buf) - 1:
            self.buf = buf
            return np.zeros(0, np.float32)
        n = int((len(buf) - 1 - self.pos) // self.step) + 1
        idx = self.pos + np.arange(n) * self.step
        out = np.interp(idx, np.arange(len(buf)), buf)
        self.pos = idx[-1] + self.step
        drop = min(max(int(self.pos) - 1, 0), len(buf))     # keep one sample before pos for interpolation
        self.buf, self.pos = buf[drop:], self.pos - drop
        return out.astype(np.float32)


class StreamProcessor:
    def __init__(self, vad, window_s: float = 4.0, hop_s: float = 1.0, min_speech_s: float = 2.0,
                 vad_threshold: float = 0.5, hangover_ms: int = 300, min_utt_s: float = 0.6, max_utt_s: float = 8.0):
        self.vad = vad
        self.vad_state = vad.new_state()
        self.F = vad.FRAME
        self.W = int(window_s * SR)
        self.HOP = int(hop_s * SR)
        self.MIN = int(min_speech_s * SR)
        self.thr = vad_threshold
        self.hang_frames = max(0, round(hangover_ms / 1000 * SR / self.F))
        self.pending = np.zeros(0, np.float32)
        self.speech_buf = np.zeros(0, np.float32)
        self.hang = 0
        self.new_speech = 0
        self.total = 0           # samples received (all audio)
        self.speech_total = 0    # speech samples received
        self.in_speech = False
        # utterances for speech-to-text: speech between pauses, cut at max_utt_s
        self.MIN_UTT = int(min_utt_s * SR)
        self.MAX_UTT = int(max_utt_s * SR)
        self.utt: list[np.ndarray] = []
        self.utt_start = 0.0
        self.utterances: list[tuple[np.ndarray, float, float]] = []

    @property
    def seconds(self) -> float:
        return self.total / SR

    @property
    def speech_seconds(self) -> float:
        return self.speech_total / SR

    def feed(self, samples: np.ndarray) -> list[np.ndarray]:
        """Add audio; returns the analysis windows that became ready (usually none or one)."""
        x = np.concatenate([self.pending, np.asarray(samples, np.float32)])
        n = len(x) // self.F * self.F
        self.pending = x[n:]
        ready = []
        for fr in x[:n].reshape(-1, self.F):
            self.total += self.F
            if self.vad.prob(fr, self.vad_state) >= self.thr:
                self.hang, speech = self.hang_frames, True
            elif self.hang > 0:
                self.hang, speech = self.hang - 1, True
            else:
                speech = False
            self.in_speech = speech
            if not speech:
                self._end_utterance()
                continue
            if not self.utt:
                self.utt_start = (self.total - self.F) / SR
            self.utt.append(fr)
            if len(self.utt) * self.F >= self.MAX_UTT:
                self._end_utterance()
            self.speech_buf = np.concatenate([self.speech_buf, fr])[-self.W:]
            self.speech_total += self.F
            self.new_speech += self.F
            if self.new_speech >= self.HOP and self.speech_total >= self.MIN:
                ready.append(self.speech_buf.copy())
                self.new_speech = 0
        return ready

    def _end_utterance(self) -> None:
        if self.utt and len(self.utt) * self.F >= self.MIN_UTT:
            self.utterances.append((np.concatenate(self.utt), self.utt_start, self.total / SR))
        self.utt = []

    def pop_utterances(self) -> list[tuple[np.ndarray, float, float]]:
        """Finished utterances since the last call: (audio, start s, end s)."""
        out, self.utterances = self.utterances, []
        return out

    def flush(self) -> list[tuple[np.ndarray, float, float]]:
        """End of stream: also return the utterance still in progress."""
        self._end_utterance()
        return self.pop_utterances()
