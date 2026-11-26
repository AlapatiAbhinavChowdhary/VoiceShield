"""VoiceShield server.

    WebSocket /ws/stream     live audio in (16-bit PCM), scores and decisions out (JSON)
    GET  /                   test console (static/index.html)
    GET  /api/status         models, settings, line profiles, enrolled voiceprints
    GET  /api/samples        demo recordings in samples/
    GET  /api/health         liveness: detector mode, live analyses (limit: config max_streams), uptime
    GET  /api/calls          audit log of recent calls;  /api/calls/{id} for one call
    GET  /api/calls/{id}/evidence   downloadable record + SHA-256 of its canonical JSON (tamper-evident)
    GET  /api/voiceprints    enrolled people;  DELETE /api/voiceprints/{name}
    GET  /api/incidents      monitored calls with alerts and prevention actions (security screen)
    POST /api/calls/{id}/events   prevention action from the customer app (transfer blocked, OTP verified, ...)
    GET  /customer /caller /security   call demo pages
    WebSocket /ws/signal/{room}?role=customer|caller   WebRTC signalling relay between the two call pages

Protocol (client -> server):
    1. text  {"type": "start", "sample_rate": 16000, "source": "mic", "profile": "default",
              "mode": "monitor" | "calibrate" | "enroll", "save_as": "mic" | "<person name>",
              "claimed": "<enrolled person the caller claims to be>", "language": "en" | "hi" | "te" | null}
    2. binary  little-endian int16 mono PCM chunks (any size, ~100 ms is good)
    3. text  {"type": "stop"}   -> server replies with "summary" ("calibrated" / "enrolled") and closes
Server -> client: "ready", "vad", "score", "transcript", "action", "calib_window", "enroll_window",
                  "summary", "calibrated", "enrolled", "error", "warning" (a window failed; the call goes on).
"""
import asyncio
import hashlib
import json
import logging
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict

import numpy as np
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .audit import Audit
from .calibration import profiles_for, threshold_from_genuine
from .config import ROOT, SR, Config, load_config
from .models import VadModel, load_detector
from .offline import make_processor
from .quality import audio_quality, confidence
from . import lab
from .risk import LEVELS, RiskEngine
from .scam import CallScam
from .speaker import CallVoice, SpeakerModel, Voiceprints
from .stream import StreamResampler

log = logging.getLogger("voiceshield")

ACTIONS = {
    "VERIFY": "Verify before acting: call back on the registered number or confirm with an OTP the bank sends.",
    "HOLD": "Hold every transaction requested on this call and alert the security team.",
}
# prevention actions the customer app may record against a call (audit trail for the security screen)
APP_EVENTS = {"TRANSFER_REQUESTED", "TRANSFER_APPROVED", "TRANSFER_BLOCKED", "TRANSFER_DECLINED", "OTP_SENT",
              "OTP_VERIFIED", "OTP_FAILED", "CALLBACK_REQUESTED", "CALL_ENDED"}
SIGNAL_TYPES = {"offer", "answer", "ice", "bye", "app"}


class AppEvent(BaseModel):
    kind: str
    t: float = 0.0
    detail: str = Field(default="", max_length=300)


def post_webhook(url: str, payload: dict) -> None:
    try:
        req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=3).close()
    except Exception as e:                                   # an alert hook must never break the call
        log.warning("webhook failed: %s", e)


def create_app(cfg: Config | None = None, detector=None, vad=None, speaker="auto", transcriber="auto") -> FastAPI:
    """detector / vad / speaker / transcriber can be injected (tests). "auto" loads the real model if enabled."""
    cfg = cfg or load_config()
    S: dict = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if vad is None and not (cfg.models / cfg.vad_file).exists():
            raise RuntimeError(f"Model file not found: {cfg.models / cfg.vad_file}. Run:  python setup_models.py")
        S["detector"] = detector or load_detector(cfg)       # original, or Step 8c (see config "detector_mode")
        S["vad"] = vad or VadModel(cfg.models / cfg.vad_file)
        S["speaker"] = speaker
        if speaker == "auto":
            p = cfg.models / cfg.speaker.model_file
            S["speaker"] = SpeakerModel(p) if cfg.speaker.enabled and p.exists() else None
        S["asr"] = transcriber
        if transcriber == "auto":
            S["asr"] = None
            if cfg.asr.enabled:
                try:
                    from .asr import Transcriber
                    S["asr"] = Transcriber(cfg.asr, cfg.models / "whisper")
                except Exception as e:
                    log.warning("speech-to-text disabled: %s", e)
        S["pool"] = ThreadPoolExecutor(max_workers=1, thread_name_prefix="detector")
        S["spk_pool"] = ThreadPoolExecutor(max_workers=1, thread_name_prefix="speaker")
        S["asr_pool"] = ThreadPoolExecutor(max_workers=1, thread_name_prefix="asr")
        S["audit"] = Audit(cfg.path(cfg.audit_db))
        S["profiles"] = profiles_for(cfg, S["detector"])
        S["voiceprints"] = Voiceprints(cfg.path(cfg.voiceprints_file))
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(S["pool"], S["detector"].score, np.zeros(SR, np.float32))   # warm-up
        log.info("VoiceShield ready (detector %s on %s, voiceprints %s, speech-to-text %s)",
                 getattr(S["detector"], "mode", "original"), getattr(S["detector"], "provider", "?"),
                 "on" if S["speaker"] else "off",
                 f"Whisper {S['asr'].size} on {S['asr'].device}" if S["asr"] else "off")
        yield
        for k in ("pool", "spk_pool", "asr_pool"):
            S[k].shutdown(wait=False)

    app = FastAPI(title="VoiceShield", lifespan=lifespan)
    app.add_middleware(GZipMiddleware, minimum_size=1024)           # the console's JS/CSS shrink about 4x
    started = time.time()
    S["active"] = 0

    @app.middleware("http")
    async def headers(request: Request, call_next):
        resp = await call_next(request)
        path = request.url.path
        if path.startswith("/static/") and "v" in request.query_params:
            resp.headers["Cache-Control"] = "public, max-age=31536000, immutable"   # versioned: cache forever
        elif path.startswith("/api/") or path.endswith(".html") or "." not in path.rsplit("/", 1)[-1]:
            resp.headers["Cache-Control"] = "no-store"                # pages and live data always fresh
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("Referrer-Policy", "no-referrer")
        resp.headers.setdefault("X-Frame-Options", "DENY")
        return resp

    app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
    (ROOT / "samples").mkdir(exist_ok=True)
    app.mount("/samples", StaticFiles(directory=ROOT / "samples"), name="samples")

    @app.get("/")
    def index():
        return FileResponse(ROOT / "static" / "index.html")

    @app.get("/api/status")
    def status():
        asr = S["asr"]
        return {"detector": getattr(S["detector"], "provider", "?"),
                "detector_mode": getattr(S["detector"], "mode", "original"), "window_s": cfg.window_seconds,
                "hop_s": cfg.hop_seconds, "min_speech_s": cfg.min_speech_seconds,
                "risk": asdict(cfg.risk), "profiles": S["profiles"].all(),
                "speaker": bool(S["speaker"]), "voiceprints": S["voiceprints"].names(),
                "asr": f"Whisper {asr.size} on {asr.device.upper()}" if asr else None}

    @app.get("/api/health")
    def health():
        """Liveness for a monitor or the pre-flight check: models loaded, live analyses, uptime."""
        return {"ok": True, "detector_mode": getattr(S["detector"], "mode", "original"),
                "active_streams": S["active"], "max_streams": cfg.max_streams,
                "uptime_s": round(time.time() - started, 1)}

    # ------------------------------------------------------------------ scenario lab (isolated from live calls)
    lab_cache: dict = {}
    lab_lock = asyncio.Lock()

    @app.get("/api/lab/scenarios")
    def lab_scenarios():
        return [{**lab.public(s), "result": lab_cache.get(s["id"])} for s in lab.SCENARIOS
                if s["mode"] == "transcript" or (ROOT / "samples" / s["file"]).exists()]

    @app.post("/api/lab/run/{sid}")
    async def lab_run(sid: str, fresh: bool = False):
        s = lab.BY_ID.get(sid)
        if s is None:
            raise HTTPException(404, "no such scenario")
        if s["mode"] != "transcript" and not (ROOT / "samples" / s["file"]).exists():
            raise HTTPException(404, f"recording {s['file']} is missing from samples/")
        if S["active"] > 0:
            raise HTTPException(409, "A live call is being analysed. The lab waits so it cannot slow the call down.")
        if sid in lab_cache and not fresh:
            return lab_cache[sid]
        async with lab_lock:                                       # one lab run at a time, on the detector's thread
            _, prof = S["profiles"].get("default")
            res = await asyncio.get_running_loop().run_in_executor(
                S["pool"], lab.run, s, S["detector"], S["vad"], cfg, float(prof["llr_threshold"]), S["asr"])
        lab_cache[sid] = res
        return res

    @app.get("/api/samples")
    def samples():
        meta = ROOT / "samples" / "samples.json"
        items = json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else []
        return [s for s in items if (ROOT / "samples" / s["file"]).exists()]

    @app.get("/api/calls")
    def calls(limit: int = 20):
        return S["audit"].recent(min(max(limit, 1), 200))

    @app.get("/api/calls/{call_id}")
    def call(call_id: int):
        c = S["audit"].call(call_id)
        if c is None:
            raise HTTPException(404, "no such call")
        return c

    @app.get("/api/calls/{call_id}/evidence")
    def evidence(call_id: int):
        """Tamper-evident evidence file for a bank or police complaint: the full audit record (scores, decisions,
        prevention actions; never audio) plus the SHA-256 of its canonical JSON, so any later edit is detectable."""
        c = S["audit"].call(call_id)
        if c is None:
            raise HTTPException(404, "no such call")
        canon = json.dumps(c, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        body = {"format": "voiceshield-evidence/1",
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "server": {"detector_mode": getattr(S["detector"], "mode", "original"),
                           "detector": getattr(S["detector"], "provider", "?")},
                "verify": "sha256 of json.dumps(call, sort_keys=True, separators=(',', ':'), ensure_ascii=False)",
                "sha256": hashlib.sha256(canon.encode("utf-8")).hexdigest(), "call": c}
        return JSONResponse(body, headers={"Content-Disposition":
                                           f'attachment; filename="voiceshield-session-{call_id}-evidence.json"'})

    @app.get("/api/voiceprints")
    def voiceprints():
        return {k: {kk: vv for kk, vv in v.items() if kk != "embedding"} for k, v in S["voiceprints"].all().items()}

    @app.delete("/api/voiceprints/{name}")
    def delete_voiceprint(name: str):
        if not S["voiceprints"].delete(name):
            raise HTTPException(404, "no such voiceprint")
        return {"deleted": name}

    # ------------------------------------------------------------------ call demo (Step 7)
    for page in ("customer", "caller", "security"):
        app.add_api_route(f"/{page}", (lambda p=page: FileResponse(ROOT / "static" / f"{p}.html")),
                          methods=["GET"], include_in_schema=False)

    @app.get("/api/incidents")
    def incidents(limit: int = 30):
        return S["audit"].incidents(min(max(limit, 1), 200))

    @app.post("/api/calls/{call_id}/events")
    def add_event(call_id: int, ev: AppEvent):
        kind = ev.kind.upper()
        if kind not in APP_EVENTS:
            raise HTTPException(400, f"kind must be one of {sorted(APP_EVENTS)}")
        if S["audit"].call(call_id) is None:
            raise HTTPException(404, "no such call")
        S["audit"].event(call_id, ev.t, kind, ev.detail)
        if kind == "TRANSFER_BLOCKED" and cfg.webhook_url:
            threading.Thread(target=post_webhook, daemon=True, args=(
                cfg.webhook_url, {"call_id": call_id, "action": kind, "detail": ev.detail})).start()
        return {"ok": True}

    rooms: dict[str, dict[str, WebSocket]] = {}

    @app.websocket("/ws/signal/{room}")
    async def signal(ws: WebSocket, room: str):
        """Relays WebRTC offer/answer/ICE (and small app messages) between the customer and caller pages."""
        role = ws.query_params.get("role")
        if role not in ("customer", "caller") or not room.isalnum() or len(room) > 32:
            await ws.close(code=1008)
            return
        await ws.accept()
        peers = rooms.setdefault(room, {})
        old = peers.get(role)
        if old is not None:                                   # a reload replaces the old page
            try:
                await old.close(code=4000)
            except Exception:
                pass
        peers[role] = ws
        other = "caller" if role == "customer" else "customer"
        await ws.send_json({"type": "joined", "role": role, "peer": other in peers})
        if other in peers:
            try:
                await peers[other].send_json({"type": "peer-joined", "role": role})
            except Exception:
                pass
        try:
            while True:
                msg = await ws.receive_json()
                if not isinstance(msg, dict) or msg.get("type") not in SIGNAL_TYPES or len(json.dumps(msg)) > 20000:
                    continue
                if other in peers:
                    await peers[other].send_json(msg)
        except (WebSocketDisconnect, RuntimeError, json.JSONDecodeError):
            pass
        finally:
            if peers.get(role) is ws:
                del peers[role]
                if other in peers:
                    try:
                        await peers[other].send_json({"type": "peer-left", "role": role})
                    except Exception:
                        pass
            if not peers:
                rooms.pop(room, None)

    @app.websocket("/ws/stream")
    async def stream(ws: WebSocket):
        await ws.accept()
        if S["active"] >= cfg.max_streams:
            await ws.send_json({"type": "error", "message": f"Server busy: {S['active']} live analyses already "
                                                            "running. Stop one and try again."})
            await ws.close(code=1013)
            return
        S["active"] += 1
        try:
            await analyse_stream(ws)
        finally:
            S["active"] -= 1

    async def analyse_stream(ws: WebSocket):
        try:
            start = json.loads(await ws.receive_text())
            assert start.get("type") == "start"
            sr_in = int(start.get("sample_rate", SR))
            assert 8000 <= sr_in <= 192000
        except Exception:
            await ws.send_json({"type": "error", "message": 'first message must be {"type": "start", "sample_rate": ...}'})
            await ws.close(code=1003)
            return

        mode = start.get("mode") if start.get("mode") in ("calibrate", "enroll") else "monitor"
        if mode == "enroll" and not S["speaker"]:
            await ws.send_json({"type": "error", "message": "voiceprint model not available (run setup_models.py)"})
            await ws.close()
            return
        source = str(start.get("source", "unknown"))[:200]
        language = start.get("language") if start.get("language") in ("en", "hi", "te") else None
        profile_name, profile = S["profiles"].get(start.get("profile"))
        threshold = float(profile["llr_threshold"])
        auto = bool(profile.get("auto"))         # Step 8c "auto" profile: each window brings its own cut-off
        claimed = start.get("claimed") if isinstance(start.get("claimed"), str) else None
        claimed = (claimed or "").strip() or None
        enrolled = S["voiceprints"].get(claimed) if claimed else None
        if claimed and enrolled is None:
            claimed = None
        risk = RiskEngine(cfg.risk, threshold)
        voice = CallVoice(enrolled, cfg.speaker.threshold, cfg.speaker.scale) if S["speaker"] and mode == "monitor" else None
        scam = CallScam(context=cfg.scam_context)
        seen_patterns: set = set()
        proc = make_processor(S["vad"], cfg)
        resample = StreamResampler(sr_in) if sr_in != SR else None
        audit: Audit = S["audit"]
        call_id = audit.start_call(source + (f" | claims: {claimed}" if claimed else ""), mode, profile_name, threshold)
        loop = asyncio.get_running_loop()
        latest: dict = {"item": None, "done": False}
        wake = asyncio.Event()
        utt_q: asyncio.Queue = asyncio.Queue()
        send_lock = asyncio.Lock()
        collected: list = []                     # calibration llrs or enrollment voiceprints
        failures = {"n": 0}                      # windows the detector could not score

        async def send(msg: dict) -> None:
            async with send_lock:
                await ws.send_json(msg)

        async def decide(t: float) -> dict:
            st = risk.state()
            if st["escalated"]:
                level = st["level"]
                msg = f"{level}: " + "; ".join(st["reasons"]) + ". " + ACTIONS[level]
                audit.event(call_id, t, level, msg)
                await send({"type": "action", "level": level, "t": round(t, 2), "message": ACTIONS[level],
                            "reasons": st["reasons"]})
                if cfg.webhook_url:
                    loop.run_in_executor(None, post_webhook, cfg.webhook_url,
                                         {"call_id": call_id, "level": level, "risk": st["risk"], "t": t,
                                          "reasons": st["reasons"], "source": source, "claimed": claimed})
            return st

        async def embed(w):
            if not S["speaker"] or (mode == "monitor" and voice is None):
                return None
            try:
                return await loop.run_in_executor(S["spk_pool"], S["speaker"].embed, w)
            except ValueError:
                return None

        async def scorer() -> None:
            while True:
                if latest["item"] is None:
                    if latest["done"]:
                        return
                    await wake.wait()
                    wake.clear()
                    continue
                w, t, speech_s = latest["item"]
                latest["item"] = None
                t0 = time.perf_counter()
                if mode == "enroll":
                    emb = await embed(w)
                    if emb is not None:
                        collected.append(emb)
                    await send({"type": "enroll_window", "n": len(collected), "t": round(t, 2),
                                "speech_s": round(speech_s, 2)})
                    continue
                try:
                    r, emb = await asyncio.gather(loop.run_in_executor(S["pool"], S["detector"].score, w), embed(w))
                except Exception as e:           # one bad window must not silently end the analysis of the call
                    log.exception("scoring failed at %.1f s", t)
                    failures["n"] += 1
                    if failures["n"] == 1:
                        await send({"type": "warning", "t": round(t, 2),
                                    "message": f"A window could not be analysed ({type(e).__name__}); still listening."})
                    continue
                lat = (time.perf_counter() - t0) * 1000
                base = {"t": round(t, 2), "speech_s": round(speech_s, 2), "llr": r["llr"], "p_fake": r["p_fake"],
                        "latency_ms": round(lat, 1)}
                cut = r["auto_threshold"] if auto and "auto_threshold" in r else threshold
                q = audio_quality(w)                               # informational: never changes the decision
                extra = {"window_threshold": round(cut, 3),       # Step 8c adds the original score and the audio type
                         **{k: r[k] for k in ("llr_original", "processing") if k in r},
                         "quality": q, "confidence": confidence(r["llr"], cut, q["label"])}
                if mode == "calibrate":
                    collected.append(r["llr"])
                    audit.window(call_id, **base)
                    await send({"type": "calib_window", "n": len(collected), **base, **extra})
                    continue
                ws_score = risk.update(r["llr"], cut)
                spk = voice.add(emb) if (voice is not None and emb is not None) else None
                if spk is not None:
                    risk.set_identity(spk["mismatch"])
                st = await decide(t)
                audit.window(call_id, **base, window_score=ws_score, risk=st["risk"], level=st["level"],
                             similarity=spk and spk["similarity"])
                await send({"type": "score", **base, **extra, "window_score": ws_score, "speaker": spk, **st})

        async def transcriber() -> None:
            while True:
                item = await utt_q.get()
                if item is None:
                    return
                u, t0, t1 = item
                try:
                    r = await loop.run_in_executor(S["asr_pool"], S["asr"].transcribe, u, language)
                except Exception as e:
                    log.warning("transcription failed: %s", e)
                    continue
                if not r["text"]:
                    continue
                s = scam.add(r["text"])
                if s["new_tactics"]:
                    audit.event(call_id, t1, "SCAM", "; ".join(h["label"] for h in s["new_tactics"]))
                for p in s["patterns"]:                      # named fraud script, logged once per call
                    if p not in seen_patterns:
                        seen_patterns.add(p)
                        audit.event(call_id, t1, "SCAM_PATTERN", p)
                risk.set_scam(s["score"], s["tactics"])
                st = await decide(t1)
                await send({"type": "transcript", "t0": round(t0, 2), "t1": round(t1, 2), **r,
                            "hits": s["hits"], "scam_score": s["score"], "patterns": s["patterns"],
                            "suppressed": s["suppressed"], **st})

        tasks = [asyncio.create_task(scorer())]
        use_asr = S["asr"] is not None and mode == "monitor"
        if use_asr:
            tasks.append(asyncio.create_task(transcriber()))
        await send({"type": "ready", "call_id": call_id, "mode": mode, "profile": profile_name,
                    "threshold": threshold, "auto_threshold": auto, "window_s": cfg.window_seconds,
                    "hop_s": cfg.hop_seconds, "detector": getattr(S["detector"], "provider", "?"),
                    "detector_mode": getattr(S["detector"], "mode", "original"), "claimed": claimed,
                    "speaker_check": voice is not None and enrolled is not None, "transcript": use_asr,
                    "language": language})
        last_vad = -1.0
        try:
            while True:
                m = await ws.receive()
                if m["type"] == "websocket.disconnect":
                    break
                if m.get("bytes") is not None:
                    b = m["bytes"]
                    x = np.frombuffer(b[: len(b) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0
                    if resample is not None:
                        x = resample(x)
                    for w in proc.feed(x):
                        latest["item"] = (w, proc.seconds, proc.speech_seconds)
                        wake.set()
                    for u in proc.pop_utterances():
                        if use_asr:
                            utt_q.put_nowait(u)
                    if proc.seconds - last_vad >= 0.25:
                        last_vad = proc.seconds
                        await send({"type": "vad", "t": round(proc.seconds, 2), "speech": proc.in_speech,
                                    "speech_s": round(proc.speech_seconds, 2)})
                elif m.get("text"):
                    try:
                        msg = json.loads(m["text"])
                    except ValueError:                   # garbage text must not leave the call open in the audit log
                        continue
                    if isinstance(msg, dict) and msg.get("type") == "stop":
                        break
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            latest["done"] = True
            wake.set()
            if use_asr:
                for u in proc.flush():
                    utt_q.put_nowait(u)
                utt_q.put_nowait(None)
            for task in tasks:
                try:
                    await task
                except Exception as e:                       # e.g. client vanished mid-send
                    log.info("worker ended: %s", e)

        verdict = mode.upper() if mode != "monitor" else LEVELS[risk.verdict]
        summary = audit.end_call(call_id, "CALIBRATION" if mode == "calibrate" else verdict, risk.max_risk,
                                 proc.seconds, proc.speech_seconds)
        try:
            if mode == "calibrate":
                name = str(start.get("save_as") or "mic")
                try:
                    thr = threshold_from_genuine(collected)
                    saved = S["profiles"].save(name, thr, source=source, method="genuine 95th percentile + 1.5",
                                               n_windows=len(collected),
                                               genuine_median_llr=round(float(np.median(collected)), 3))
                    await send({"type": "calibrated", "profile": name, **saved})
                except ValueError as e:
                    await send({"type": "error", "message": f"Calibration not saved: {e}. Speak for longer (20 s)."})
            elif mode == "enroll":
                name = str(start.get("save_as") or "").strip()
                if len(collected) < cfg.speaker.min_enroll_windows:
                    await send({"type": "error", "message": f"Voiceprint not saved: only {len(collected)} windows of "
                                f"speech, need {cfg.speaker.min_enroll_windows}. Speak for about 20 s."})
                else:
                    emb = np.mean(collected, axis=0)
                    try:
                        saved = S["voiceprints"].save(name, emb / np.linalg.norm(emb), proc.speech_seconds, source)
                        await send({"type": "enrolled", "name": name, "windows": len(collected), **saved})
                    except ValueError as e:
                        await send({"type": "error", "message": f"Voiceprint not saved: {e}"})
            else:
                st = risk.state()
                await send({"type": "summary", "call_id": call_id, "verdict": summary["verdict"],
                            "max_risk": round(summary["max_risk"] or 0, 1), "windows": summary["windows"],
                            "audio_s": round(proc.seconds, 1), "speech_s": round(proc.speech_seconds, 1),
                            "signals": st["signals"], "reasons": st["reasons"]})
            await ws.close()
        except Exception:
            pass

    return app
