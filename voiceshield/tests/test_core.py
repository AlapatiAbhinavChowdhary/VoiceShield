"""Fast tests with stub models (no downloads needed):  python -m pytest -q"""
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.calibration import Profiles, profiles_for, threshold_eer, threshold_from_genuine  # noqa: E402
from server.config import SR, Config, RiskConfig, load_config                     # noqa: E402
from server.models import LayerBackend                                            # noqa: E402
from server.risk import RiskEngine                                                # noqa: E402
from server.scam import CallScam, find_tactics                                    # noqa: E402
from server.speaker import CallVoice, Voiceprints, fbank                          # noqa: E402
from server.stream import StreamProcessor, StreamResampler                        # noqa: E402

rng = np.random.default_rng(0)


class StubVad:
    """'Speech' = any frame with RMS above 0.01."""
    FRAME = 512

    def new_state(self):
        return {}

    def prob(self, frame, st):
        return 1.0 if np.sqrt(np.mean(frame ** 2)) > 0.01 else 0.0


class StubDetector:
    """'Fake' = loud audio (RMS > 0.2). Returns the same keys as the real Detector."""
    provider = "stub"

    def score(self, w):
        llr = 3.0 if np.sqrt(np.mean(np.asarray(w) ** 2)) > 0.2 else -10.0
        return {"llr": llr, "p_fake": float(1 / (1 + np.exp(-llr)))}


class StubSpeaker:
    """Voiceprint = direction depends on loudness: quiet (<0.06 RMS) is 'person A', louder is 'person B'."""
    A, B = np.eye(4)[0], np.eye(4)[1]

    def embed(self, w):
        return self.A if np.sqrt(np.mean(np.asarray(w) ** 2)) < 0.06 else self.B


class StubTranscriber:
    """Loud utterances 'say' a scam script, others say something harmless."""
    size, device = "stub", "cpu"

    def transcribe(self, wav, language=None):
        loud = np.sqrt(np.mean(np.asarray(wav) ** 2)) > 0.08
        text = "I am calling from your bank, your account will be blocked, share the OTP now" if loud \
            else "hello, how are you doing today"
        return {"text": text, "language": "en", "language_prob": 0.99, "latency_ms": 1}


def speech(sec, amp=0.1):
    return (rng.standard_normal(int(sec * SR)) * amp).astype(np.float32)


def silence(sec):
    return np.zeros(int(sec * SR), np.float32)


def feed_in_chunks(proc, wav, sizes):
    out, i, k = [], 0, 0
    while i < len(wav):
        n = sizes[k % len(sizes)]
        out += proc.feed(wav[i:i + n])
        i, k = i + n, k + 1
    return out


def level(eng, llr):
    eng.update(llr)
    return eng.state()


# ---------------------------------------------------------------- config
def test_shipped_config_loads():
    cfg = load_config()
    assert cfg.speaker.threshold > 0 and cfg.asr.model and cfg.risk.scam_at == 50


# ---------------------------------------------------------------- resampler
def test_resampler_passband_and_antialias():
    sr_in = 48000
    t = np.arange(sr_in * 2) / sr_in
    for f, expect_pass in [(1000, True), (3000, True), (10000, False), (15000, False)]:
        r = StreamResampler(sr_in)
        y = np.concatenate([r(c) for c in np.array_split(np.sin(2 * np.pi * f * t).astype(np.float32), 37)])
        assert abs(len(y) - 2 * SR) <= 2
        amp = np.sqrt(2) * np.sqrt(np.mean(y[SR // 2:] ** 2))
        if expect_pass:
            assert 0.95 < amp < 1.05, (f, amp)
        else:
            assert amp < 0.01, (f, amp)          # > 40 dB rejection of what would alias


def test_resampler_chunking_invariant():
    x = speech(1.0)
    x44 = np.interp(np.arange(0, len(x), 16000 / 44100), np.arange(len(x)), x).astype(np.float32)
    whole = StreamResampler(44100)(x44)
    r = StreamResampler(44100)
    parts = np.concatenate([r(c) for c in np.array_split(x44, 23)])
    assert len(parts) == len(whole) and np.allclose(parts, whole, atol=1e-5)


# ---------------------------------------------------------------- stream processor
def make_proc(**kw):
    args = dict(window_s=4.0, hop_s=1.0, min_speech_s=2.0, vad_threshold=0.5, hangover_ms=0)
    args.update(kw)
    return StreamProcessor(StubVad(), **args)


def test_windows_are_speech_only_and_scheduled():
    wav = np.concatenate([silence(2), speech(6), silence(3), speech(2)])
    wins = feed_in_chunks(make_proc(), wav, [1600])
    # 8 s of speech (the silence gap adds nothing); windows every ~1 s of speech once 2 s are in.
    # VAD frames are 32 ms, so "2 s" is 63 frames (2.016 s) and a hop is 32 frames (1.024 s).
    lens = [len(w) / SR for w in wins]
    assert len(wins) == 6 and abs(lens[0] - 2.0) < 0.05 and abs(lens[1] - 3.0) < 0.07
    assert lens[2:] == [4.0] * 4
    for w in wins:
        frames = w.reshape(-1, 512)
        assert (np.sqrt((frames ** 2).mean(1)) > 0.01).all(), "silence leaked into a window"


def test_chunk_size_does_not_change_windows():
    wav = np.concatenate([speech(3), silence(1), speech(4)])
    a = feed_in_chunks(make_proc(), wav, [len(wav)])
    for sizes in ([160], [1600, 7, 4097], [333, 51200]):
        b = feed_in_chunks(make_proc(), wav, sizes)
        assert len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b))


def test_hangover_bridges_short_pauses():
    wav = np.concatenate([speech(1.0), silence(0.2), speech(1.0)])
    p = make_proc(hangover_ms=300)
    feed_in_chunks(p, wav, [1600])
    assert p.speech_seconds >= 2.15             # the 0.2 s pause counted as speech


def test_utterances_split_on_pauses_and_length():
    wav = np.concatenate([speech(2), silence(1), speech(10), silence(1), speech(0.3), silence(1), speech(1.5)])
    p = make_proc(min_utt_s=0.6, max_utt_s=8.0)
    feed_in_chunks(p, wav, [1600])
    utts = p.pop_utterances() + p.flush()
    lens = [round(len(u) / SR, 1) for u, _, _ in utts]
    assert lens == [2.0, 8.0, 2.0, 1.5], lens                # 10 s cut at 8 s; the 0.3 s blip is dropped
    t0s = [round(t0, 1) for _, t0, _ in utts]
    assert t0s[0] == 0.0 and t0s[1] == 3.0


# ---------------------------------------------------------------- risk engine
def test_risk_escalates_to_hold_and_stays():
    eng = RiskEngine(RiskConfig())
    assert [level(eng, -10.0)["level"] for _ in range(5)] == ["PASS"] * 5
    seq = [level(eng, 3.0) for _ in range(4)]
    assert [s["level"] for s in seq] == ["PASS", "VERIFY", "HOLD", "HOLD"]
    assert [s["escalated"] for s in seq] == [False, True, True, False]
    later = [level(eng, -10.0) for _ in range(10)]
    assert later[-1]["level"] == "PASS" and later[-1]["verdict"] == "HOLD"   # verdict never de-escalates


def test_one_odd_window_does_not_hold():
    eng = RiskEngine(RiskConfig())
    assert "HOLD" not in [level(eng, x)["level"] for x in [-10, -10, 5, -10, -10, -10]]


def test_odd_first_window_does_not_alert():
    """Real case (10 Nov, call 13): a genuine caller's first window scored just above the cut-off."""
    eng = RiskEngine(RiskConfig(), llr_threshold=-6.01)
    levels = [level(eng, x)["level"] for x in [-4.2, -8.3, -9.2, -8.9, -8.8, -9.6]]
    assert levels == ["PASS"] * 6


def test_clone_from_first_window_still_holds_by_third():
    eng = RiskEngine(RiskConfig(), llr_threshold=-6.01)
    assert [level(eng, x)["level"] for x in [-2.9, -3.1, -1.7]] == ["PASS", "VERIFY", "HOLD"]


def test_window_score_is_half_at_threshold():
    eng = RiskEngine(RiskConfig(), llr_threshold=-8.0)
    assert abs(eng.window_score(-8.0) - 0.5) < 1e-9 and eng.window_score(1e9) == pytest.approx(1.0)


def test_window_threshold_override():
    """The Step 8c "auto" profile gives each window its own cut-off."""
    eng = RiskEngine(RiskConfig(), llr_threshold=-4.0)
    assert eng.update(-3.0) > 0.5 and eng.update(-3.0, threshold=-2.0) < 0.5


# ---------------------------------------------------------------- Step 8c back-end (NumPy)
def make_backend_file(path, w=0.85, n=3, d=4):
    """A small back-end file with the same keys and layout as the Step 8c notebook writes."""
    r = np.random.default_rng(1)
    conds = np.array(["clean", "auto-gain + noise suppression", "Opus"])
    np.savez(path, conds=conds, mu=r.normal(size=(n, d)).astype(np.float32),
             sd=r.uniform(0.5, 2, (n, d)).astype(np.float32), det_a=r.normal(size=n).astype(np.float32),
             det_w=r.normal(size=d).astype(np.float32), det_b=np.array([0.3], np.float32),
             cls_a=r.normal(size=n).astype(np.float32), cls_w=r.normal(size=(3, d)).astype(np.float32),
             cls_b=r.normal(size=3).astype(np.float32), fusion_w=w, mo=-1.0, so=8.0, mp=-0.5, sp=7.0,
             thr_original=np.array([-4.0, -5.0, 0.2]), thr_fused=np.array([-0.2, 0.0, 0.1]),
             thr_single_original=-4.0, thr_single_fused=-0.1)
    return path


def test_layer_backend_matches_notebook_formula(tmp_path):
    B = LayerBackend(make_backend_file(tmp_path / "b.npz"))
    raw = np.load(tmp_path / "b.npz")
    layers = np.random.default_rng(2).normal(size=(3, 4))

    def softmax(a):
        e = np.exp(a - a.max())
        return e / e.sum()
    # written the way the notebook's PyTorch code computes it (test_step8c.py does the same check)
    z = (layers - raw["mu"]) / raw["sd"]
    q = (softmax(raw["det_a"])[:, None] * z).sum(0) @ raw["det_w"] + raw["det_b"][0]
    fused = 0.15 * (2.5 - raw["mo"]) / raw["so"] + 0.85 * (q - raw["mp"]) / raw["sp"]
    cond = int(np.argmax(raw["cls_w"] @ (softmax(raw["cls_a"])[:, None] * z).sum(0) + raw["cls_b"]))
    r = B(2.5, layers)
    assert r["llr"] == pytest.approx(-1.0 + 8.0 * fused) and r["backend"] == pytest.approx(q)
    assert r["llr_original"] == 2.5 and r["processing"] == B.conds[cond]
    assert r["auto_threshold"] == pytest.approx(B.thresholds[r["processing"]])
    assert B.threshold == pytest.approx(-1.0 + 8.0 * -0.1)                  # fused thresholds on the llr scale
    assert "auto-gain + spectral subtraction" in B.conds                     # what that notebook effect really was
    # fusion weight 0 gives back the detector's own score exactly
    B0 = LayerBackend(make_backend_file(tmp_path / "b0.npz", w=0.0))
    assert B0(-7.25, layers)["llr"] == pytest.approx(-7.25)


def test_profiles_are_kept_per_score(tmp_path):
    cfg = replace(Config(), profiles_file=str(tmp_path / "profiles.json"))

    class Orig:
        pass

    class Fused:
        backend = LayerBackend(make_backend_file(tmp_path / "b.npz"))

    p_orig, p_new = profiles_for(cfg, Orig()), profiles_for(cfg, Fused())
    assert p_orig.path.name == "profiles.json" and p_new.path.name == "profiles_step8c.json"
    p_orig.save("mic", -6.0)
    assert "mic" in p_orig.all() and "mic" not in p_new.all()               # never mix thresholds of two scores
    assert p_new.all()["default"]["llr_threshold"] == pytest.approx(Fused.backend.threshold, abs=1e-3)
    auto = p_new.all()["auto"]
    assert auto["auto"] and set(auto["thresholds"]) == set(Fused.backend.conds) and "auto" not in p_orig.all()
    with pytest.raises(ValueError):
        p_new.save("auto", -1.0)


def test_fusion_rules():
    cfg = RiskConfig()
    # scam language alone -> VERIFY, never HOLD
    e = RiskEngine(cfg); level(e, -10); e.set_scam(0.84, ["asks for OTP / PIN / password"])
    st = e.state()
    assert st["level"] == "VERIFY" and "scam language" in st["reasons"][0]
    # voice suspicious earlier (VERIFY only) + scam later -> HOLD
    e = RiskEngine(cfg)
    for x in [-10, 3, 3, -10, -10, -10, -10]:
        level(e, x)
    assert e.state()["verdict"] == "VERIFY"
    e.set_scam(0.6, ["asks for OTP / PIN / password"])
    st = e.state()
    assert st["level"] == "HOLD" and st["escalated"] and any("earlier" in r for r in st["reasons"])
    # identity mismatch alone -> VERIFY; with scam -> HOLD
    e = RiskEngine(cfg); level(e, -10); e.set_identity(0.97)
    assert e.state()["level"] == "VERIFY"
    e.set_scam(0.55, ["threatens police / arrest / legal action"])
    assert e.state()["level"] == "HOLD"
    # genuine voice, matching identity, harmless talk -> PASS; combined risk is a noisy-OR
    e = RiskEngine(cfg); level(e, -10); e.set_identity(0.02); e.set_scam(0.4, ["asks to send / transfer money"])
    st = e.state()
    assert st["level"] == "PASS" and 40 < st["risk"] < 45


# ---------------------------------------------------------------- scam language
@pytest.mark.parametrize("text,expect", [
    ("Hello sir, I am calling from SBI. Your account will be blocked. Share the OTP immediately.",
     {"credentials", "account_threat", "urgency_secrecy", "impersonation"}),
    ("Sir aapka KYC pending hai, turant OTP batao warna account band ho jayega",
     {"credentials", "account_threat", "urgency_secrecy"}),
    ("नमस्ते, मैं बैंक से बोल रहा हूँ, आपका अकाउंट ब्लॉक हो जाएगा, ओटीपी बताइए",
     {"credentials", "account_threat", "impersonation"}),
    ("నమస్కారం, నేను బ్యాంక్ నుంచి మాట్లాడుతున్నాను, మీ ఓటీపీ చెప్పండి, వెంటనే",
     {"credentials", "urgency_secrecy", "impersonation"}),
    ("This is CBI, a parcel with drugs in your name, you are under digital arrest, do not tell anyone",
     {"authority_threat", "urgency_secrecy"}),
    ("Please install AnyDesk so I can fix your refund", {"remote_access", "payment"}),
    ("The weather is nice and we are planning a trip to the hills", set()),
    ("I opened the app and the pin icon was missing", {"credentials"}),     # known false positive: 'pin'
])
def test_scam_tactics(text, expect):
    assert {h["tactic"] for h in find_tactics(text)} == expect


def test_scam_score_accumulates_once_per_tactic():
    c = CallScam()
    assert c.add("I want to transfer money to my brother")["score"] == pytest.approx(0.4)
    assert c.add("transfer it today")["score"] == pytest.approx(0.4)              # same tactic: no increase
    s = c.add("and tell me the OTP")["score"]
    assert s == pytest.approx(1 - 0.6 * 0.4) and c.score >= 0.5


# ---------------------------------------------------------------- voiceprints
def test_fbank_shape_and_mean_normalised():
    f = fbank(speech(1.0))
    assert f.shape == (98, 80) and np.abs(f.mean(axis=0)).max() < 1e-3
    assert fbank(speech(0.01)).shape[0] == 0


def test_call_voice_mismatch(tmp_path):
    enrolled = np.eye(4)[0]
    v = CallVoice(enrolled, threshold=0.45, scale=0.08)
    same = v.add(np.array([0.9, 0.1, 0, 0]))
    assert same["similarity"] > 0.9 and same["mismatch"] < 0.01
    v2 = CallVoice(enrolled, 0.45, 0.08)
    other = v2.add(np.eye(4)[1])
    assert other["mismatch"] > 0.99
    assert CallVoice(None, 0.45, 0.08).add(np.eye(4)[0]) is None            # no claimed identity
    vp = Voiceprints(tmp_path / "v.json")
    vp.save("Abhinav", enrolled, 21.0, "mic")
    assert vp.names() == ["Abhinav"] and np.allclose(vp.get("Abhinav"), enrolled)
    with pytest.raises(ValueError):
        vp.save("../evil", enrolled, 1.0)
    assert vp.delete("Abhinav") and vp.get("Abhinav") is None


# ---------------------------------------------------------------- calibration / profiles
def test_calibration_rules(tmp_path):
    assert threshold_from_genuine([-10.0] * 20) == pytest.approx(-8.5)
    assert threshold_from_genuine([-20.0] * 20) == pytest.approx(-9.5)          # clipped
    with pytest.raises(ValueError):
        threshold_from_genuine([-10.0] * 3)
    thr, eer = threshold_eer([-10, -9.5, -9.8, -9.9], [-4, -3, 0, 2])
    assert eer == 0 and -9.5 < thr <= -4
    prof = Profiles(tmp_path / "p.json", -4.07)
    prof.save("mic", -6.2, source="test")
    assert prof.get("mic")[1]["llr_threshold"] == -6.2 and prof.get("nope")[0] == "default"
    with pytest.raises(ValueError):
        prof.save("default", -1)
    with pytest.raises(ValueError):
        prof.save("../x", -1)


# ---------------------------------------------------------------- server end to end (stub models)
@pytest.fixture()
def client(tmp_path):
    from fastapi.testclient import TestClient
    from server.app import create_app
    cfg = replace(Config(), audit_db=str(tmp_path / "audit.db"), profiles_file=str(tmp_path / "profiles.json"),
                  voiceprints_file=str(tmp_path / "voiceprints.json"), vad_hangover_ms=0)
    app = create_app(cfg, detector=StubDetector(), vad=StubVad(), speaker=StubSpeaker(), transcriber=StubTranscriber())
    with TestClient(app) as c:
        yield c


def run_call(client, wav, sr=SR, **start):
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_json({"type": "start", "sample_rate": sr, "source": "test", **start})
        ready = ws.receive_json()
        assert ready["type"] == "ready"
        pcm = (np.clip(wav, -1, 1) * 32767).astype("<i2")
        step = sr // 10
        for i in range(0, len(pcm), step):
            ws.send_bytes(pcm[i:i + step].tobytes())
        ws.send_json({"type": "stop"})
        msgs = []
        while True:
            m = ws.receive_json()
            msgs.append(m)
            if m["type"] in ("summary", "calibrated", "enrolled", "error"):
                return ready, msgs


def test_ws_genuine_call_passes(client):
    _, msgs = run_call(client, np.concatenate([silence(1), speech(8, amp=0.05)]))   # quiet = harmless talk
    assert msgs[-1]["type"] == "summary" and msgs[-1]["verdict"] == "PASS"
    assert any(m["type"] == "score" for m in msgs) and not any(m["type"] == "action" for m in msgs)
    assert any(m["type"] == "transcript" and m["text"].startswith("hello") for m in msgs)


def test_ws_cloned_call_holds_and_is_audited(client):
    _, msgs = run_call(client, speech(10, amp=0.4))
    actions = [m["level"] for m in msgs if m["type"] == "action"]
    assert actions[:2] == ["VERIFY", "HOLD"] and msgs[-1]["verdict"] == "HOLD"
    cid = msgs[-1]["call_id"]
    call = client.get(f"/api/calls/{cid}").json()
    assert call["verdict"] == "HOLD" and len(call["window_log"]) == msgs[-1]["windows"] > 0
    kinds = [e["kind"] for e in call["events"]]
    assert "HOLD" in kinds and "SCAM" in kinds
    blob = json.dumps(call).lower()
    assert "otp now" not in blob and "embedding" not in blob     # no transcript text or voiceprint stored


def test_ws_scam_script_with_genuine_voice_is_verify_not_hold(client):
    # stub detector: amp 0.12 is "real" (RMS < 0.2) but the stub transcriber hears a scam script (RMS > 0.08)
    _, msgs = run_call(client, speech(10, amp=0.12))
    assert msgs[-1]["verdict"] == "VERIFY"
    assert any("scam language" in r for r in msgs[-1]["reasons"])


def test_ws_enroll_then_claimed_identity(client):
    _, msgs = run_call(client, speech(10, amp=0.05), mode="enroll", save_as="Abhinav")      # quiet = person A
    assert msgs[-1]["type"] == "enrolled" and msgs[-1]["name"] == "Abhinav"
    assert client.get("/api/status").json()["voiceprints"] == ["Abhinav"]
    ready, msgs = run_call(client, speech(8, amp=0.05), claimed="Abhinav")                  # same person
    assert ready["speaker_check"] and msgs[-1]["verdict"] == "PASS"
    sims = [m["speaker"]["similarity"] for m in msgs if m["type"] == "score"]
    assert sims and min(sims) > 0.99
    _, msgs = run_call(client, speech(8, amp=0.07), claimed="Abhinav")                      # person B claims A
    assert msgs[-1]["verdict"] == "VERIFY"
    assert any("does not match" in r for r in msgs[-1]["reasons"])
    assert client.delete("/api/voiceprints/Abhinav").status_code == 200


def test_ws_enroll_too_short(client):
    _, msgs = run_call(client, speech(3, amp=0.05), mode="enroll", save_as="X")
    assert msgs[-1]["type"] == "error"


def test_ws_resamples_44k_input(client):
    x = speech(8, amp=0.4)
    x44 = np.interp(np.arange(0, len(x), 16000 / 44100), np.arange(len(x)), x).astype(np.float32)
    _, msgs = run_call(client, x44, sr=44100)
    assert msgs[-1]["verdict"] == "HOLD" and 7.5 < msgs[-1]["audio_s"] < 8.5


class StubLayerDetector(StubDetector):
    """Step 8c-like stub: quiet speech scores -10 with audio type 'clean' (its own cut-off -12), loud speech
    scores +3 with audio type 'Opus' (cut-off +5). So "auto" flags quiet speech and passes loud speech,
    the opposite of a fixed cut-off of -4."""
    mode = "step8c"

    def __init__(self, backend):
        self.backend = backend

    def score(self, w):
        loud = np.sqrt(np.mean(np.asarray(w) ** 2)) > 0.2
        llr = 3.0 if loud else -10.0
        return {"llr": llr, "p_fake": float(1 / (1 + np.exp(-llr))), "llr_original": llr - 1,
                "processing": "Opus" if loud else "clean", "auto_threshold": 5.0 if loud else -12.0}


def test_ws_step8c_detector_and_auto_profile(tmp_path):
    from fastapi.testclient import TestClient
    from server.app import create_app
    backend = LayerBackend(make_backend_file(tmp_path / "b.npz"))
    backend.threshold = -4.0
    cfg = replace(Config(), audit_db=str(tmp_path / "audit.db"), profiles_file=str(tmp_path / "profiles.json"),
                  voiceprints_file=str(tmp_path / "voiceprints.json"), vad_hangover_ms=0)
    app = create_app(cfg, detector=StubLayerDetector(backend), vad=StubVad(), speaker=StubSpeaker(),
                     transcriber=StubTranscriber())
    with TestClient(app) as c:
        st = c.get("/api/status").json()
        assert st["detector_mode"] == "step8c" and {"default", "auto"} <= set(st["profiles"])
        ready, msgs = run_call(c, speech(8, amp=0.05), profile="default")
        assert ready["detector_mode"] == "step8c" and not ready["auto_threshold"] and msgs[-1]["verdict"] == "PASS"
        scores = [m for m in msgs if m["type"] == "score"]
        assert scores and scores[0]["processing"] == "clean" and scores[0]["window_threshold"] == -4.0
        ready, msgs = run_call(c, speech(8, amp=0.05), profile="auto")
        assert ready["auto_threshold"] and msgs[-1]["verdict"] == "HOLD"
        assert {m["window_threshold"] for m in msgs if m["type"] == "score"} == {-12.0}
        _, msgs = run_call(c, speech(12), mode="calibrate", save_as="mic")
        assert msgs[-1]["type"] == "calibrated" and (tmp_path / "profiles_step8c.json").exists()
        assert not (tmp_path / "profiles.json").exists()


def test_ws_calibration_saves_profile(client):
    _, msgs = run_call(client, speech(12), mode="calibrate", save_as="mic")
    assert msgs[-1]["type"] == "calibrated" and msgs[-1]["llr_threshold"] == pytest.approx(-8.5)
    ready, _ = run_call(client, speech(3), profile="mic")
    assert ready["profile"] == "mic" and ready["threshold"] == pytest.approx(-8.5)
    assert "mic" in client.get("/api/status").json()["profiles"]


def test_ws_calibration_too_short(client):
    _, msgs = run_call(client, speech(3), mode="calibrate", save_as="mic")
    assert msgs[-1]["type"] == "error"


def test_call_pages_are_served(client):
    for page in ("/customer", "/caller", "/security", "/static/vs-client.js", "/static/ui.css", "/static/shell.js"):
        assert client.get(page).status_code == 200, page


def test_signalling_relay(client):
    with client.websocket_connect("/ws/signal/demo?role=customer") as cust:
        assert cust.receive_json() == {"type": "joined", "role": "customer", "peer": False}
        with client.websocket_connect("/ws/signal/demo?role=caller") as caller:
            assert caller.receive_json() == {"type": "joined", "role": "caller", "peer": True}
            assert cust.receive_json() == {"type": "peer-joined", "role": "caller"}
            caller.send_json({"type": "nonsense", "x": 1})                 # ignored, not relayed
            caller.send_json({"type": "offer", "sdp": "v=0..."})
            assert cust.receive_json() == {"type": "offer", "sdp": "v=0..."}
            cust.send_json({"type": "answer", "sdp": "v=0 answer"})
            assert caller.receive_json()["type"] == "answer"
            caller.send_json({"type": "app", "action": "transfer_request", "amount": 49999, "payee": "X"})
            assert cust.receive_json()["amount"] == 49999
        assert cust.receive_json() == {"type": "peer-left", "role": "caller"}
    with pytest.raises(Exception):                                        # bad role / room are refused
        with client.websocket_connect("/ws/signal/demo?role=hacker") as ws:
            ws.receive_json()


def test_prevention_events_and_incidents(client):
    _, msgs = run_call(client, speech(10, amp=0.4), source="call:demo")
    cid = msgs[-1]["call_id"]
    assert client.post(f"/api/calls/{cid}/events", json={"kind": "transfer_blocked", "t": 5.0,
                                                          "detail": "₹49,999 to X"}).status_code == 200
    assert client.post(f"/api/calls/{cid}/events", json={"kind": "DROP_TABLE"}).status_code == 400
    assert client.post("/api/calls/99999/events", json={"kind": "OTP_SENT"}).status_code == 404
    inc = client.get("/api/incidents").json()
    mine = next(r for r in inc if r["id"] == cid)
    kinds = [e["kind"] for e in mine["events"]]
    assert mine["verdict"] == "HOLD" and "TRANSFER_BLOCKED" in kinds and "HOLD" in kinds


def test_ws_rejects_bad_start(client):
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_json({"hello": 1})
        assert ws.receive_json()["type"] == "error"


def test_ws_odd_start_fields_and_garbage_text_still_end_the_call(client):
    """A list as profile, a dict as claimed person and non-JSON text mid-call must not break the call or the audit."""
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_json({"type": "start", "sample_rate": SR, "source": "test", "profile": ["x"], "claimed": {"a": 1}})
        ready = ws.receive_json()
        assert ready["type"] == "ready" and ready["profile"] == "default" and ready["claimed"] is None
        pcm = (speech(6, amp=0.05) * 32767).astype("<i2")
        for i in range(0, len(pcm), SR // 10):
            ws.send_bytes(pcm[i:i + SR // 10].tobytes())
            if i == SR:
                ws.send_text("not json")
                ws.send_text("[1, 2]")
        ws.send_json({"type": "stop"})
        while (m := ws.receive_json())["type"] != "summary":
            pass
    assert m["verdict"] == "PASS"
    assert client.get(f"/api/calls/{ready['call_id']}").json()["verdict"] == "PASS"


def test_health_compression_and_cache_headers(client):
    h = client.get("/api/health").json()
    assert h["ok"] and h["active_streams"] == 0 and h["max_streams"] >= 1
    js = client.get("/static/app.js?v=test", headers={"Accept-Encoding": "gzip"})
    assert js.headers.get("content-encoding") == "gzip" and "immutable" in js.headers["cache-control"]
    assert client.get("/").headers["cache-control"] == "no-store"
    assert client.get("/api/status").headers["x-content-type-options"] == "nosniff"


def test_evidence_file_is_tamper_evident(client):
    import hashlib
    import json as _json
    ready, _ = run_call(client, speech(6, amp=0.3))
    r = client.get(f"/api/calls/{ready['call_id']}/evidence")
    assert r.status_code == 200 and "attachment" in r.headers["content-disposition"]
    ev = r.json()
    canon = _json.dumps(ev["call"], sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert ev["sha256"] == hashlib.sha256(canon.encode()).hexdigest() and ev["call"]["verdict"] == "HOLD"
    ev["call"]["verdict"] = "PASS"                                     # any edit changes the digest
    assert hashlib.sha256(_json.dumps(ev["call"], sort_keys=True, separators=(",", ":"),
                                      ensure_ascii=False).encode()).hexdigest() != ev["sha256"]
    assert client.get("/api/calls/999999/evidence").status_code == 404


def test_ws_busy_when_too_many_streams(tmp_path):
    from fastapi.testclient import TestClient
    from server.app import create_app
    cfg = replace(Config(), audit_db=str(tmp_path / "audit.db"), profiles_file=str(tmp_path / "profiles.json"),
                  voiceprints_file=str(tmp_path / "voiceprints.json"), max_streams=1)
    app = create_app(cfg, detector=StubDetector(), vad=StubVad(), speaker=StubSpeaker(), transcriber=StubTranscriber())
    with TestClient(app) as c:
        with c.websocket_connect("/ws/stream") as first:
            first.send_json({"type": "start", "sample_rate": SR, "source": "test"})
            assert first.receive_json()["type"] == "ready"
            with c.websocket_connect("/ws/stream") as second:
                m = second.receive_json()
                assert m["type"] == "error" and "busy" in m["message"].lower()
            first.send_json({"type": "stop"})
            while first.receive_json()["type"] != "summary":
                pass
        assert c.get("/api/health").json()["active_streams"] == 0


class FlakyDetector(StubDetector):
    """Fails on every other window, like a transient runtime error."""
    def __init__(self):
        self.n = 0

    def score(self, w):
        self.n += 1
        if self.n % 2 == 0:
            raise RuntimeError("boom")
        return super().score(w)


def test_ws_detector_error_warns_and_keeps_scoring(tmp_path):
    from fastapi.testclient import TestClient
    from server.app import create_app
    cfg = replace(Config(), audit_db=str(tmp_path / "audit.db"), profiles_file=str(tmp_path / "profiles.json"),
                  voiceprints_file=str(tmp_path / "voiceprints.json"), vad_hangover_ms=0)
    det = FlakyDetector()
    app = create_app(cfg, detector=det, vad=StubVad(), speaker=StubSpeaker(), transcriber=StubTranscriber())
    with TestClient(app) as c:
        det.n = 0                                             # the warm-up call at start-up does not count
        _, msgs = run_call(c, speech(14, amp=0.3))            # loud = fake for the stub
    kinds = [m["type"] for m in msgs]
    assert kinds.count("warning") == 1 and kinds.count("score") >= 3
    assert msgs[-1]["type"] == "summary" and msgs[-1]["verdict"] == "HOLD"
