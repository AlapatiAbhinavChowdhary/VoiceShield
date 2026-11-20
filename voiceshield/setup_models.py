"""Download the models VoiceShield needs into ./models (about 2 GB, only once).

    python setup_models.py

Set VOICESHIELD_MODELS to use another folder.
"""
import os
import sys
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download

MODELS = Path(os.environ.get("VOICESHIELD_MODELS", Path(__file__).resolve().parent / "models"))

# NII AntiDeepfake wav2vec2-Large, ONNX export (licence: CC BY-NC-SA 4.0, research/education use)
DETECTOR_REPO = "SpeechAntiSpoofingBenchmarks/Wav2Vec2-Large-AntiDeepfake"
DETECTOR_FILE = "wav2vec2-large-antideepfake.onnx"

# Silero VAD v5 (MIT licence), pinned to a release tag
VAD_URL = "https://raw.githubusercontent.com/snakers4/silero-vad/v5.1.2/src/silero_vad/data/silero_vad.onnx"
VAD_FILE = "silero_vad.onnx"

# WeSpeaker ResNet34-LM speaker embedding, VoxCeleb (Apache-2.0)
SPEAKER_REPO = "Wespeaker/wespeaker-voxceleb-resnet34-LM"
SPEAKER_FILE = "voxceleb_resnet34_LM.onnx"

# Whisper (MIT) converted for faster-whisper: small for GPU, base as the CPU fallback
WHISPER_SIZES = ("small", "base")


def main() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)

    vad = MODELS / VAD_FILE
    if not vad.exists():
        print("Downloading Silero VAD ...")
        tmp = vad.with_suffix(".part")
        urllib.request.urlretrieve(VAD_URL, tmp)
        tmp.replace(vad)
    print(f"VAD:        {vad} ({vad.stat().st_size / 1e6:.1f} MB)")

    det = MODELS / DETECTOR_FILE
    if not det.exists():
        print("Downloading the AntiDeepfake detector (about 1.26 GB) ...")
        hf_hub_download(DETECTOR_REPO, DETECTOR_FILE, local_dir=MODELS)
    print(f"Detector:   {det} ({det.stat().st_size / 1e9:.2f} GB)")

    spk = MODELS / SPEAKER_FILE
    if not spk.exists():
        print("Downloading the voiceprint model (26 MB) ...")
        hf_hub_download(SPEAKER_REPO, SPEAKER_FILE, local_dir=MODELS)
    print(f"Voiceprint: {spk} ({spk.stat().st_size / 1e6:.0f} MB)")

    try:
        from faster_whisper import download_model
        for size in WHISPER_SIZES:
            print(f"Whisper {size}: ", end="", flush=True)
            print(download_model(size, cache_dir=str(MODELS / "whisper")))
    except ImportError:
        print("faster-whisper not installed: run  pip install -r requirements.txt  first (speech-to-text is optional)")
    print("Models ready.")


if __name__ == "__main__":
    sys.exit(main())
