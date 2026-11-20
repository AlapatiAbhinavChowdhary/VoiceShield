"""Risk engine: three independent signals -> PASS / VERIFY / HOLD, with the reasons.

Signals (each 0-100):
  V  synthetic voice  - EMA of detector window scores (starting from 0 = genuine);
                        window score = sigmoid((llr - line cut-off) / scale)
  I  identity         - probability the caller is NOT the person they claim to be (voiceprint mismatch);
                        only when a claimed identity is set
  C  scam language    - scam-tactic score of the transcript so far (asks for OTP, threatens arrest, ...)

Decision rules (explainable, in this order):
  HOLD    V >= hold_at and k of the last n windows look fake            (strong synthetic voice)
  HOLD    suspicious voice or identity (ever in this call) AND C >= scam_at   (fused evidence)
  VERIFY  V >= verify_at, or 2 suspicious windows, or I >= identity_at, or C >= scam_at
  PASS    otherwise
The call verdict only escalates: once HOLD, the call stays HOLD.
The single "risk" number shown is the noisy-OR of the three signals.
"""
import math
from collections import deque

from .config import RiskConfig

LEVELS = ("PASS", "VERIFY", "HOLD")


class RiskEngine:
    def __init__(self, cfg: RiskConfig, llr_threshold: float | None = None):
        self.cfg = cfg
        self.threshold = cfg.llr_threshold if llr_threshold is None else llr_threshold
        self.ema = 0.0                        # starts at "genuine": one odd first window cannot trigger an alert
        self.recent: deque[bool] = deque(maxlen=cfg.hold_n)
        self.identity: float | None = None   # 0-100 or None
        self.scam = 0.0                       # 0-100
        self.scam_tactics: list[str] = []
        self.voice_flag = False               # synthetic voice was suspicious at some point in this call
        self.identity_flag = False
        self.verdict = 0
        self.max_risk = 0.0
        self.windows = 0

    # ---------------------------------------------------------------- signal inputs
    def window_score(self, llr: float, threshold: float | None = None) -> float:
        z = (llr - (self.threshold if threshold is None else threshold)) / self.cfg.llr_scale
        return 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, z))))

    # Inputs only record evidence; call state() afterwards (once) to get the decision.
    def update(self, llr: float, threshold: float | None = None) -> float:
        """New detector window; returns its window score (0-1).

        threshold: this window's own cut-off (the "auto" line profile); None = the call's line profile.
        """
        c = self.cfg
        s = self.window_score(llr, threshold)
        self.ema = (1 - c.ema_alpha) * self.ema + c.ema_alpha * s
        self.recent.append(s >= 0.5)
        self.windows += 1
        return s

    def set_identity(self, mismatch: float | None) -> None:
        """Voiceprint mismatch probability 0-1 (None = no claimed identity)."""
        self.identity = None if mismatch is None else 100.0 * mismatch

    def set_scam(self, score: float, tactics: list[str]) -> None:
        self.scam = 100.0 * score
        self.scam_tactics = list(tactics)

    # ---------------------------------------------------------------- decision
    @property
    def voice(self) -> float:
        return 100.0 * self.ema

    def state(self) -> dict:
        c = self.cfg
        V, I, C = self.voice, self.identity, self.scam
        k = sum(self.recent)
        voice_now = V >= c.verify_at or k >= 2
        identity_now = I is not None and I >= c.identity_at
        scam_now = C >= c.scam_at
        self.voice_flag |= voice_now
        self.identity_flag |= identity_now
        reasons = []
        if voice_now:
            reasons.append(f"synthetic-voice score {V:.0f}/100")
        if identity_now:
            reasons.append(f"voice does not match the claimed person ({I:.0f}/100)")
        if scam_now:
            reasons.append("scam language: " + ", ".join(self.scam_tactics))
        if V >= c.hold_at and k >= c.hold_k:
            level = 2
        elif scam_now and (self.voice_flag or self.identity_flag):
            level = 2
            if not voice_now and self.voice_flag:
                reasons.append("synthetic voice flagged earlier in the call")
            if not identity_now and self.identity_flag:
                reasons.append("voice mismatch flagged earlier in the call")
        elif voice_now or identity_now or scam_now:
            level = 1
        else:
            level = 0
        risk = 100.0 * (1.0 - (1 - V / 100) * (1 - (I or 0.0) / 100) * (1 - C / 100))
        escalated = level > self.verdict
        self.verdict = max(self.verdict, level)
        self.max_risk = max(self.max_risk, risk)
        return {"risk": risk, "level": LEVELS[level], "verdict": LEVELS[self.verdict], "escalated": escalated,
                "signals": {"voice": round(V, 1), "identity": None if I is None else round(I, 1), "scam": round(C, 1)},
                "suspicious_recent": k, "reasons": reasons}
