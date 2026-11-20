"""Tests with the real ONNX models and demo recordings (skipped until setup_models.py has run)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.calibration import Profiles          # noqa: E402
from server.config import ROOT, load_config      # noqa: E402

cfg = load_config()
HAVE_MODELS = (cfg.models / cfg.detector_file).exists() and (cfg.models / cfg.vad_file).exists()
pytestmark = pytest.mark.skipif(not HAVE_MODELS, reason="run setup_models.py first")
SAMPLES = ROOT / "samples"


@pytest.fixture(scope="module")
def models():
    from server.models import Detector, VadModel
    return Detector(cfg.models / cfg.detector_file, cfg.detector_threads), VadModel(cfg.models / cfg.vad_file)


def demo(kind):
    meta = SAMPLES / "samples.json"
    items = json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else []
    return [s for s in items if s["kind"] == kind and (SAMPLES / s["file"]).exists()]


def test_vad_ignores_noise_and_silence(models):
    _, vad = models
    rng = np.random.default_rng(0)
    for x in (np.zeros(16000 * 2, np.float32), (rng.standard_normal(16000 * 2) * 0.01).astype(np.float32)):
        st = vad.new_state()
        assert max(vad.prob(x[i:i + 512], st) for i in range(0, len(x) - 511, 512)) < 0.5


def test_detector_output_is_finite(models):
    det, _ = models
    r = det.score(np.zeros(1000, np.float32))
    assert det.window == 64000 and np.isfinite(r["llr"]) and 0 <= r["p_fake"] <= 1


def test_voiceprints_separate_two_real_speakers():
    """genuine_call_1 and genuine_call_2 are different people (Step 4 test split)."""
    from server.offline import load_audio
    from server.speaker import SpeakerModel, cosine
    p = cfg.models / cfg.speaker.model_file
    if not p.exists() or not (SAMPLES / "genuine_call_2.flac").exists():
        pytest.skip("voiceprint model or demo calls missing")
    m = SpeakerModel(p)
    b = load_audio(SAMPLES / "genuine_call_2.flac")
    a = load_audio(SAMPLES / "genuine_call_1.flac")
    b1, b2, a1 = m.embed(b[:4 * 16000]), m.embed(b[5 * 16000:9 * 16000]), m.embed(a[:4 * 16000])
    assert cosine(b1, b2) > 0.5 > cosine(a1, b1)


def test_scam_call_full_pipeline_holds():
    from server.asr import Transcriber
    from server.offline import analyse, load_audio
    from server.models import Detector, VadModel
    clip = SAMPLES / "scam_call_tts.wav"
    if not clip.exists() or not (cfg.models / "whisper").exists():
        pytest.skip("scam demo clip or Whisper model missing")
    det, vad = Detector(cfg.models / cfg.detector_file), VadModel(cfg.models / cfg.vad_file)
    res = analyse(load_audio(clip), det, vad, cfg, transcriber=Transcriber(cfg.asr, cfg.models / "whisper"),
                  language="en")
    tactics = {h["tactic"] for t in res["transcript"] for h in t["hits"]}
    assert res["verdict"] == "HOLD" and {"credentials", "account_threat"} <= tactics
    assert res["signals"]["scam"] >= 50


@pytest.mark.parametrize("kind,expect", [("genuine", {"PASS"}), ("cloned", {"VERIFY", "HOLD"})])
def test_demo_calls(models, kind, expect):
    from server.offline import analyse, load_audio
    items = demo(kind)
    if not items:
        pytest.skip("no demo recordings in samples/")
    profiles = Profiles(cfg.path(cfg.profiles_file), cfg.risk.llr_threshold)
    for s in items:
        _, prof = profiles.get(s.get("profile"))
        res = analyse(load_audio(SAMPLES / s["file"]), *models, cfg, prof["llr_threshold"])
        assert res["speech_s"] > 0.6 * res["audio_s"], s["file"]
        assert res["verdict"] in expect, (s["file"], res["verdict"], res["max_risk"])


# ---------------------------------------------------------------- Step 8c (frozen detector + back-end)
HAVE_8C = (cfg.models / cfg.layers_file).exists() and (cfg.models / cfg.backend_file).exists()
need_8c = pytest.mark.skipif(not HAVE_8C, reason="Step 8c files (voiceshield_layers.onnx, step8c_backend.npz) missing")


# These tests load one 1.26 GB detector at a time (the demo laptop has little free memory). Run them on their
# own if memory is short:  python -m pytest -q tests/test_models.py -k step8c
def load_step8c():
    from server.models import Detector, LayerBackend
    return Detector(cfg.models / cfg.layers_file, cfg.detector_threads, backend=LayerBackend(cfg.models / cfg.backend_file))


@need_8c
def test_step8c_export_is_the_same_detector():
    """voiceshield_layers.onnx must give the original detector's own score (same weights, extra output)."""
    import gc
    from server.models import Detector
    from server.offline import load_audio
    wav = load_audio(SAMPLES / "genuine_call_1.flac") if (SAMPLES / "genuine_call_1.flac").exists() \
        else (np.random.default_rng(0).standard_normal(160000) * 0.1).astype(np.float32)
    windows = [wav[i * 32000: i * 32000 + 64000] for i in range(3)]
    det = Detector(cfg.models / cfg.detector_file, cfg.detector_threads)
    ref = [det.score(w)["llr"] for w in windows]
    del det
    gc.collect()
    det = load_step8c()
    for w, r0 in zip(windows, ref):
        r = det.score(w)
        assert abs(r["llr_original"] - r0) < 0.01
        assert np.isfinite(r["llr"]) and r["processing"] in det.backend.conds


@pytest.fixture(scope="module")
def step8c():
    return load_step8c()


@need_8c
@pytest.mark.parametrize("kind,expect", [("genuine", {"PASS"}), ("cloned", {"VERIFY", "HOLD"})])
def test_step8c_demo_calls(step8c, kind, expect):
    from server.calibration import profiles_for
    from server.models import VadModel
    from server.offline import analyse, load_audio
    items = demo(kind)
    if not items:
        pytest.skip("no demo recordings in samples/")
    profiles, vad = profiles_for(cfg, step8c), VadModel(cfg.models / cfg.vad_file)
    for s in items:
        _, prof = profiles.get(s.get("profile"))
        res = analyse(load_audio(SAMPLES / s["file"]), step8c, vad, cfg, prof["llr_threshold"])
        assert res["verdict"] in expect, (s["file"], res["verdict"], res["max_risk"])
