# VoiceShield

**Real-time detection and prevention of voice-cloning impersonation in calls** (Smart India Hackathon 2026, problem SIH26104).

VoiceShield listens to a call and decides every second whether to **PASS** it, ask to **VERIFY** the caller another way, or **HOLD** it. It combines three checks:
- an AI-voice detector;
- a voiceprint check against the person the caller claims to be;
- scam-language rules on a live transcript.

On HOLD, the bank app blocks the payment automatically and the security team is alerted.

| Start here | |
|---|---|
| [explain.md](explain.md) | The whole project in plain words, from the problem to the final demo |
| [voiceshield/README.md](voiceshield/README.md) | How to install and run the system, the screens, the demo script |
| [docs/SIH26104_Research_and_Build_Plan.md](docs/SIH26104_Research_and_Build_Plan.md) | Research report and the full results log (Steps 2–8d) |
| [docs/paper/voiceshield_paper.tex](docs/paper/voiceshield_paper.tex) | The research paper (LaTeX, compile on Overleaf) |
| [notebooks/](notebooks/) | Kaggle notebooks for every experiment, with local CPU tests in `notebooks/build/` |

### Key results
- **Live demo:** a robot-voice scam call is held after about 4 s of speech, and the transfer is blocked automatically.
- **Real premium clone:** a premium ElevenLabs clone is now held after 5.6 s instead of only being flagged for verification.
- **Consumer audio processing:** noise suppression, Bluetooth and VoIP compression break deepfake detectors. Error on premium clones rose from 8% to 34–45%.
- **Step 8c:** our fix leaves the detector frozen and reads its internal layers. The mean error over 13 unseen test sets fell from 9.7% to 4.1%, with no forgetting.

### Not included in this repository
- **Model files:** the ONNX models are 1.26 GB each, so `voiceshield/setup_models.py` downloads them. The Step 8c layer model comes from the Step 8c notebook.
- **Personal data:** our own voice recordings, enrolled voiceprints and the local audit log are kept private.

### Licences
The detector (NII AntiDeepfake) is CC BY-NC-SA 4.0, IndicSynth is CC BY-NC 4.0, and the demo clips come from a CC BY 4.0 dataset. This project is for non-commercial research and education only.
