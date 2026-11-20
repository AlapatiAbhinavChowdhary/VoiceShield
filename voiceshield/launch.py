"""Start the local server and open the browser once it is ready."""
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.request import urlopen
import webbrowser

ROOT = Path(__file__).resolve().parent
URL = "http://127.0.0.1:8000"


def ready():
    try:
        with urlopen(URL + "/api/status", timeout=2) as response:
            status = json.load(response)
        return all(key in status for key in ("detector", "speaker", "asr", "profiles"))
    except (OSError, ValueError):
        return False


def main():
    if ready():
        print("VoiceShield is already running. Opening " + URL, flush=True)
        webbrowser.open(URL)
        return 0

    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", 8000)) == 0:
            print("Port 8000 is already in use, but VoiceShield is not ready.")
            print("Wait for an existing startup to finish, or close the app using that port and retry.")
            return 1

    print("Starting VoiceShield. Loading the voice models may take a minute.", flush=True)
    print("Keep this window open. Press Ctrl+C to stop VoiceShield.", flush=True)
    server = subprocess.Popen([sys.executable, str(ROOT / "run.py")], cwd=ROOT)
    try:
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if server.poll() is not None:
                print("VoiceShield could not start. See the error above.")
                return 1
            if ready():
                print("VoiceShield is ready: " + URL, flush=True)
                webbrowser.open(URL)
                return server.wait()
            time.sleep(1)
        print("Startup timed out. See the server messages above.")
        return 1
    except KeyboardInterrupt:
        print("\nStopping VoiceShield...")
        return 0
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


if __name__ == "__main__":
    sys.exit(main())
