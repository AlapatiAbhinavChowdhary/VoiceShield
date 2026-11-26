"""Demo-day pre-flight check: is this laptop ready to run VoiceShield live?

    python tools/preflight.py          (or double-click start_demo.bat, which runs this first)

Checks, without changing anything: model files, settings, line profiles, enrolled voiceprints, free memory,
the server port, the GPU for speech-to-text, the demo recordings, and (Windows) which microphones are active
and whether audio enhancements are off on the built-in microphone. Exit code 1 if something blocks the demo.
"""
import ctypes
import json
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from server.config import load_config  # noqa: E402

OK, WARN, FAIL = "ok", "WARN", "FAIL"
results: list[tuple[str, str, str]] = []


def report(status: str, what: str, detail: str = "") -> None:
    results.append((status, what, detail))
    print(f"  [{status:>4}] {what}" + (f": {detail}" if detail else ""), flush=True)


def gb(n: float) -> str:
    return f"{n / 2**30:.1f} GB"


def check_models(cfg) -> None:
    m = cfg.models
    need = [(cfg.vad_file, 1e6, "voice activity detector")]
    if cfg.detector_mode == "step8c":
        need += [(cfg.layers_file, 1e9, "detector with layer outputs (Step 8c)"),
                 (cfg.backend_file, 1e3, "Step 8c back-end")]
    else:
        need += [(cfg.detector_file, 1e9, "AI-voice detector")]
    if cfg.speaker.enabled:
        need += [(cfg.speaker.model_file, 1e7, "voiceprint model")]
    for name, min_size, what in need:
        p = m / name
        if not p.exists():
            report(FAIL, what, f"missing {p.name}; run: python setup_models.py")
        elif p.stat().st_size < min_size:
            report(FAIL, what, f"{p.name} is only {p.stat().st_size} bytes (incomplete download?)")
        else:
            report(OK, what, f"{p.name} ({p.stat().st_size / 2**20:.0f} MB)")
    if cfg.asr.enabled:
        w = m / "whisper"
        has = w.exists() and any(f.is_file() and f.stat().st_size > 1e6 for f in w.rglob("*"))
        report(OK if has else WARN, "speech-to-text model",
               "cached" if has else "not cached yet: the first call downloads it (needs internet)")


def check_profiles(cfg) -> None:
    path = cfg.path(cfg.profiles_file)
    if cfg.detector_mode == "step8c":
        path = path.with_name(path.stem + "_step8c" + path.suffix)
    saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    names = ", ".join(f"{k} {v['llr_threshold']:+.2f}" for k, v in saved.items()) or "none"
    report(OK, f"line profiles for the {cfg.detector_mode} score", names + " (+ built-in default/auto)")
    if "mic" not in saved:
        report(WARN, "laptop-microphone profile", 'not calibrated: Settings > "Calibrate my line" with 20 s of '
                                                  "your own voice, before a live mic demo")


def check_voiceprints(cfg) -> None:
    p = cfg.path(cfg.voiceprints_file)
    names = sorted(json.loads(p.read_text(encoding="utf-8"))) if p.exists() else []
    if names:
        report(OK, "enrolled voiceprints", f"{len(names)} enrolled")
    else:
        report(WARN, "enrolled voiceprints", "none: enrol a team member to demo the identity check")


def check_memory() -> None:
    if sys.platform != "win32":
        return

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    s = MEMORYSTATUSEX()
    s.dwLength = ctypes.sizeof(s)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
    free = s.ullAvailPhys
    # the detector (1.2 GB file) needs about 2.5 GB while loading, Whisper and the rest about 1 GB more
    # a warning, never a blocker: Windows usually frees memory when the models load (a blocked viva is worse)
    status = OK if free > 4 * 2**30 else WARN
    report(status, "free memory", f"{gb(free)} of {gb(s.ullTotalPhys)}"
           + ("" if status == OK else ": close browser tabs, Kaggle and other servers before starting"))


def check_port(port: int = 8000) -> None:
    with socket.socket() as sck:
        busy = sck.connect_ex(("127.0.0.1", port)) == 0
    if not busy:
        report(OK, f"port {port}", "free")
        return
    try:
        from urllib.request import urlopen
        with urlopen(f"http://127.0.0.1:{port}/api/status", timeout=2) as r:
            st = json.load(r)
        report(OK, f"port {port}", f"VoiceShield already running (detector mode {st.get('detector_mode')})")
    except Exception:
        report(FAIL, f"port {port}", "used by another program: close it or run  python run.py --port 8010")


def check_gpu() -> None:
    try:
        import ctranslate2
        n = ctranslate2.get_cuda_device_count()
    except Exception:
        n = 0
    report(OK if n else WARN, "GPU for speech-to-text",
           f"{n} CUDA device(s): Whisper small" if n else "none: Whisper base on CPU (slower transcripts, still works)")


def check_samples() -> None:
    meta = ROOT / "samples" / "samples.json"
    items = json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else []
    have = [s for s in items if (ROOT / "samples" / s["file"]).exists()]
    report(OK if len(have) >= 4 else WARN, "demo recordings", f"{len(have)} of {len(items)} in samples/")


def windows_microphones() -> list[dict]:
    """Active capture endpoints from the registry (read only). enhancements_off: True/False/None (unknown)."""
    import winreg
    base = r"SOFTWARE\Microsoft\Windows\CurrentVersion\MMDevices\Audio\Capture"
    name_key, iface_key = "{a45c254e-df1c-4efd-8020-67d146a850e0},2", "{b3f8fa53-0004-438e-9003-51a46e139bfc},6"
    sysfx_key = "{1da5d803-d492-4edd-8c23-e0c0ffee7f0e},5"     # PKEY_AudioEndpoint_Disable_SysFx

    def value(path, name):
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path) as k:
                return winreg.QueryValueEx(k, name)[0]
        except OSError:
            return None

    mics = []
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base) as root:
        i = 0
        while True:
            try:
                dev = winreg.EnumKey(root, i)
            except OSError:
                break
            i += 1
            path = base + "\\" + dev
            if value(path, "DeviceState") != 1:                 # 1 = active
                continue
            off = value(path + r"\FxProperties", sysfx_key)
            if off is None:
                off = value(path + r"\Properties", sysfx_key)
            mics.append({"name": value(path + r"\Properties", name_key) or "?",
                         "interface": value(path + r"\Properties", iface_key) or "",
                         "enhancements_off": None if off is None else bool(off)})
    return mics


RISKY = ("bluetooth", "hands-free", "headset", "virtual", "voicemeeter", "vb-audio", "cable", "stereo mix")


def check_microphones() -> None:
    if sys.platform != "win32":
        return
    try:
        mics = windows_microphones()
    except Exception as e:
        report(WARN, "microphones", f"could not read the device list ({e})")
        return
    builtin = [m for m in mics if not any(r in (m["name"] + " " + m["interface"]).lower() for r in RISKY)]
    risky = [m for m in mics if m not in builtin]
    for m in builtin:
        st = {True: "enhancements OFF", False: "enhancements ON", None: "enhancements unknown"}[m["enhancements_off"]]
        report(OK if m["enhancements_off"] else WARN, f"microphone '{m['name']}'",
               st + ("" if m["enhancements_off"] else ": turn them off in Sound settings > this microphone; "
                                                        "with them on, a real voice can be flagged as AI"))
    if not builtin:
        report(WARN, "microphones", "no active built-in microphone found")
    if risky:
        report(WARN, "other active inputs", ", ".join(sorted({m["name"] for m in risky}))
               + ": in the browser choose the built-in microphone, not these (Bluetooth or virtual inputs "
                 "made a real voice look like AI in our tests)")


def main() -> int:
    print("VoiceShield pre-flight check\n", flush=True)
    try:
        cfg = load_config()
        report(OK, "settings (config.json)", f"detector mode {cfg.detector_mode}")
    except Exception as e:
        report(FAIL, "settings (config.json)", str(e))
        return 1
    for check in (lambda: check_models(cfg), lambda: check_profiles(cfg), lambda: check_voiceprints(cfg),
                  check_memory, check_port, check_gpu, check_samples, check_microphones):
        try:
            check()
        except Exception as e:                                   # a broken check must not hide the others
            report(WARN, getattr(check, "__name__", "check"), f"check failed: {e}")
    fails = sum(s == FAIL for s, _, _ in results)
    warns = sum(s == WARN for s, _, _ in results)
    print(f"\n{'NOT READY' if fails else 'READY'}: {fails} blocking problem(s), {warns} warning(s).")
    print("Demo rules: Windows audio enhancements OFF, no Bluetooth microphone, built-in mic selected in the browser.")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
