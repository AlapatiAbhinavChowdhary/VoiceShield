# Demo calls

Built by `tools/build_demo.py` from **garystafford/deepfake-audio-detection** (CC BY 4.0,
https://huggingface.co/datasets/garystafford/deepfake-audio-detection). Each file joins the first
8 clips of one recording (16 kHz mono, 0.4 s pauses). Real speakers yt_0009 / yt_0010 and fake
services ElevenLabs / Kokoro are the Step 4 *test* split: they were never used to train or calibrate
anything. The `studio` line profile was calibrated on the *dev* split (Hume fakes, real yt_0013,
yt_0001, yt_0003, yt_0007).
