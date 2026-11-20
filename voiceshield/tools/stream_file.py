"""Stream an audio file to a running server as if it were a live call, and print what comes back.

    python tools/stream_file.py samples/cloned_call_elevenlabs.flac --profile studio
    python tools/stream_file.py my.wav --speed 4        (4x faster than real time)
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np
import websockets

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from server.offline import load_audio   # noqa: E402

CHUNK_S = 0.1


async def run(path: str, url: str, profile: str, speed: float, quiet: bool) -> dict:
    wav = load_audio(path)
    pcm = (np.clip(wav, -1, 1) * 32767).astype("<i2")
    step = int(16000 * CHUNK_S)
    summary: dict = {}
    async with websockets.connect(url, max_size=None) as ws:
        await ws.send(json.dumps({"type": "start", "sample_rate": 16000, "source": f"file:{Path(path).name}",
                                  "profile": profile}))
        ready = json.loads(await ws.recv())
        print(f"call {ready['call_id']}  profile '{ready['profile']}' (llr threshold {ready['threshold']:.2f})")

        async def reader():
            async for raw in ws:
                m = json.loads(raw)
                if m["type"] == "score" and not quiet:
                    print(f"  t={m['t']:5.1f}s  llr={m['llr']:6.2f}  risk={m['risk']:5.1f}  {m['level']:6s}  "
                          f"({m['latency_ms']:.0f} ms)")
                elif m["type"] == "action":
                    print(f"  >>> {m['level']} at {m['t']}s: {m['message']}")
                elif m["type"] in ("summary", "error"):
                    summary.update(m)

        rd = asyncio.create_task(reader())
        t0 = time.perf_counter()
        for k, i in enumerate(range(0, len(pcm), step)):
            await ws.send(pcm[i:i + step].tobytes())
            if speed > 0:
                await asyncio.sleep(max(0.0, t0 + (k + 1) * CHUNK_S / speed - time.perf_counter()))
        await ws.send(json.dumps({"type": "stop"}))
        await rd
    print(f"verdict {summary.get('verdict')}  max risk {summary.get('max_risk')}  windows {summary.get('windows')}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--url", default="ws://127.0.0.1:8000/ws/stream")
    ap.add_argument("--profile", default="default")
    ap.add_argument("--speed", type=float, default=1.0, help="1 = real time, 0 = as fast as possible")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    asyncio.run(run(a.file, a.url, a.profile, a.speed, a.quiet))
