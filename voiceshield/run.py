"""Start the VoiceShield server:  python run.py   then open http://127.0.0.1:8000"""
import argparse
import logging
import os

import uvicorn

from server.app import create_app

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--models", help="folder with the .onnx models (default: ./models)")
    a = ap.parse_args()
    if a.models:
        os.environ["VOICESHIELD_MODELS"] = a.models
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    uvicorn.run(create_app(), host=a.host, port=a.port, log_level="info")
