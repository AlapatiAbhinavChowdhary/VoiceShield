"""Settings from config.json (paths are relative to the project folder)."""
import json
import os
from dataclasses import dataclass, field, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SR = 16000                       # every model here works on 16 kHz mono


def write_text_atomic(path: str | os.PathLike, text: str) -> None:
    """Write to a temporary file, then rename it: a crash never leaves a half-written JSON file behind."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


@dataclass
class RiskConfig:
    llr_threshold: float = -4.07  # log-odds of p_fake = 0.0168, the dev-set EER threshold from Step 4
    llr_scale: float = 1.0        # log-odds per unit of the window score's sigmoid
    ema_alpha: float = 0.4        # weight of the newest window in the running risk
    verify_at: float = 45         # running synthetic-voice risk (0-100) that triggers VERIFY
    hold_at: float = 75           # running synthetic-voice risk that can trigger HOLD ...
    hold_k: int = 3               # ... if at least k of the last n windows look fake
    hold_n: int = 5
    identity_at: float = 50       # voice-mismatch probability (0-100) that counts as "not the claimed person"
    scam_at: float = 50           # scam-language score (0-100) that counts as a scam script


@dataclass
class SpeakerConfig:
    enabled: bool = True
    model_file: str = "voxceleb_resnet34_LM.onnx"
    threshold: float = 0.45       # cosine similarity where mismatch probability = 0.5
    scale: float = 0.08           # how sharply mismatch rises below the threshold
    min_enroll_windows: int = 5   # ~7 s of speech


@dataclass
class AsrConfig:
    enabled: bool = True
    model: str = "small"          # faster-whisper size on GPU: tiny / base / small / medium
    cpu_model: str = "base"       # size used when falling back to CPU (small is ~3x slower there)
    device: str = "auto"          # auto = GPU if it works, else CPU; or "cpu" / "cuda"
    cpu_threads: int = 4
    min_utterance_seconds: float = 0.6
    max_utterance_seconds: float = 8.0


@dataclass
class Config:
    models_dir: str = "models"
    detector_file: str = "wav2vec2-large-antideepfake.onnx"
    # "original": the detector's own score. "step8c": the same frozen detector + the Step 8c back-end on its
    # hidden layers (more robust to noise suppression, Bluetooth and Opus); needs the two files below.
    detector_mode: str = "original"
    layers_file: str = "voiceshield_layers.onnx"
    backend_file: str = "step8c_backend.npz"
    vad_file: str = "silero_vad.onnx"
    detector_threads: int = 0     # 0 = let ONNX Runtime decide
    window_seconds: float = 4.0
    hop_seconds: float = 1.0
    min_speech_seconds: float = 2.0
    vad_threshold: float = 0.5
    vad_hangover_ms: int = 300
    risk: RiskConfig = field(default_factory=RiskConfig)
    speaker: SpeakerConfig = field(default_factory=SpeakerConfig)
    asr: AsrConfig = field(default_factory=AsrConfig)
    audit_db: str = "data/audit.db"
    profiles_file: str = "data/profiles.json"
    voiceprints_file: str = "data/voiceprints.json"
    webhook_url: str | None = None
    max_streams: int = 4          # live analyses at once; more would make every call fall behind real time

    def path(self, p: str) -> Path:
        p = Path(p)
        return p if p.is_absolute() else ROOT / p

    @property
    def models(self) -> Path:
        return Path(os.environ["VOICESHIELD_MODELS"]) if os.environ.get("VOICESHIELD_MODELS") else self.path(self.models_dir)


_SECTIONS = {"risk": RiskConfig, "speaker": SpeakerConfig, "asr": AsrConfig}


def load_config(path: str | os.PathLike | None = None) -> Config:
    path = Path(path or os.environ.get("VOICESHIELD_CONFIG") or ROOT / "config.json")
    raw = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    unknown = set(raw) - {f.name for f in fields(Config)}
    if unknown:
        raise ValueError(f"Unknown settings in {path}: {sorted(unknown)}")
    sections = {}
    for name, cls in _SECTIONS.items():
        sub = raw.pop(name, {}) or {}
        bad = set(sub) - {f.name for f in fields(cls)}
        if bad:
            raise ValueError(f"Unknown settings in {path} -> {name}: {sorted(bad)}")
        sections[name] = cls(**sub)
    cfg = Config(**raw, **sections)
    if cfg.detector_mode not in ("original", "step8c"):
        raise ValueError(f'{path}: detector_mode must be "original" or "step8c", not {cfg.detector_mode!r}')
    return cfg
