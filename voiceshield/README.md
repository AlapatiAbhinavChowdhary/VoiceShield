# VoiceShield: live voice-clone detection server (SIH26104, Steps 5–6)

VoiceShield listens to call audio as it arrives and gives a decision every second of speech: **PASS**, **VERIFY** or **HOLD**, with the reasons. It combines three independent checks:

```
browser / call audio ──16-bit PCM──► WebSocket /ws/stream
                                        │
                     Silero VAD (32 ms frames): only speech goes on
             ┌──────────────────────────┼───────────────────────────┐
   rolling 4 s of speech every 1 s      │              utterances (speech between pauses)
             │                          │                           │
  A. NII AntiDeepfake (ONNX)   B. WeSpeaker voiceprint     C. Whisper speech-to-text
     synthetic-voice score        vs enrolled person          → scam-tactic rules
     (V: 0-100)                   (I: mismatch 0-100)         (C: 0-100, en / hi / te)
             └──────────────────────────┼───────────────────────────┘
                     risk engine → PASS / VERIFY / HOLD + reasons
          JSON to the client · action messages · optional webhook · SQLite audit log
```

### Decision rules (server/risk.py)
| Result | When |
|---|---|
| **HOLD** | synthetic voice is strong (running score ≥ 75 and 3 of the last 5 windows look fake) |
| **HOLD** | scam language (C ≥ 50) **and** the voice was suspicious at any point (V flagged, or voice ≠ claimed person) |
| **VERIFY** | any single signal is suspicious: V ≥ 45, 2 suspicious windows, I ≥ 50, or C ≥ 50 |
| **PASS** | none of the above |

The call verdict never goes back down. Scam words alone (a real person talking about OTPs) give VERIFY, never HOLD. A premium clone that only reaches VERIFY on voice becomes HOLD as soon as it uses a scam script.

### The three checks
- **A. Synthetic voice:** NII AntiDeepfake wav2vec2-Large, kept frozen, plus our Step 8c back-end on its 25 internal layers; see *Step 8c detector* and *Line profiles* below.
- **B. Voiceprint:** WeSpeaker ResNet34 (VoxCeleb).
  - Enroll a person with 25 s of speech. Only a 256-number voiceprint is stored, in `data/voiceprints.json`.
  - During a call, the caller's running voiceprint is compared with the claimed person's: same person ≈ 0.6–0.9 cosine similarity, different person ≈ 0.0–0.25, cut-off 0.45.
  - This catches human impersonators and wrong callers. A good clone *of the enrolled person* may still match, which is why check A exists.
- **C. Scam language:** Whisper transcribes each utterance. Rules in `server/scam.py` find 9 tactics in English, romanised Hindi/Telugu, Devanagari and Telugu script:
  - asks for an OTP/PIN;
  - remote-access app;
  - send money;
  - police/arrest threat;
  - account/KYC block;
  - family emergency;
  - lottery;
  - urgency/secrecy;
  - claims to be a bank.

  Tactics combine (noisy-OR), so "transfer money" alone stays at 40 while "bank + account blocked + share OTP" reaches 84. Whisper runs on the GPU (`small`) when NVIDIA libraries are installed, otherwise on the CPU (`base`).

## Run it (Windows)

Keep the project in a **short path** such as `C:\projects\voiceshield`. Some libraries fail to load from very long paths on Windows.

Run these in a terminal opened in the project folder:

```bash
py -3.12 -m venv .venv
```
```bash
.venv\Scripts\python -m pip install -r requirements.txt
```
```bash
.venv\Scripts\python setup_models.py
```
```bash
.venv\Scripts\python run.py
```

`setup_models.py` downloads about 2 GB, once. Then open **http://127.0.0.1:8000** in Chrome or Edge, or run `python run.py --port 8010` if port 8000 is busy.

Optional, NVIDIA GPU only: install this to make speech-to-text about 10× faster (about 1 GB):

```bash
.venv\Scripts\python -m pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"
```

**Microphone rules:** these were the cause of every false alarm in our tests.
1. Turn **off Windows audio enhancements** for the mic: **Settings → System → Sound → (your mic) → Audio enhancements: Off**. With them on, the laptop's noise suppression made a real voice score -2 instead of -10, which caused a false HOLD.
2. Use the laptop's built-in mic or a wired mic, **never a Bluetooth headset mic**.
3. Calibrate the `mic` profile on the demo laptop, in the demo room, after steps 1–2.

Calling `.venv\Scripts\python` directly means you never need to "activate" the environment, which PowerShell often blocks.

## Call demo (Step 7): three screens
| Screen | URL | Who uses it |
|---|---|---|
| **Customer phone** | `/customer` | the person being called. VoiceShield analyses the *caller's* audio live; includes a mock bank app |
| **Caller** | `/caller` | the genuine caller, or the attacker: speaks by mic, or **plays a recording (e.g. a cloned voice) into the call** |
| **Security operations** | `/#/incidents` (or `/security`) | the bank's security team: live incidents, reasons, scam tactics, prevention actions |

The audio travels between Caller and Customer as a real WebRTC call: Opus at 128 kbit/s, no browser processing on the caller's mic. The server relays only the call setup, through `/ws/signal/<room>`. It runs on one laptop (two windows), or on two devices on the same Wi-Fi.

**Prevention in the customer's bank app**, applied when the caller asks for money:

| Call verdict | Payment request |
|---|---|
| PASS | "Pay now" works |
| VERIFY | payment needs a check that does not depend on the caller: an **OTP sent to the registered phone** (shown as an SMS that warns never to share it), or **hang up and call the bank back** on the official number |
| HOLD | **payment blocked automatically**; security team alerted (incident number), recorded as `TRANSFER_BLOCKED` |

### Demo script (about 5 minutes)
Setup, once:
- Laptop mic with Windows audio enhancements OFF.
- Enrol the presenter's voice in the console: **Settings & enrolment** (`/#/settings`).
- Open three windows: `/customer` (click **Go online**), `/caller`, `/#/incidents`.
- Same laptop for caller and customer: keep the customer's **Talk** off, or use wired earphones.

1. **Genuine call.** Customer: *Caller claims to be* = presenter. Caller: **Call the customer** → customer **Answer** → the presenter speaks → **PASS**, voice match about 0.8. The caller sends a transfer request; the customer pays; it goes through.
2. **Impersonator.** Same claim, but a different person speaks → **VERIFY: voice does not match**. Transfer → **OTP / call-back required**.
3. **Cloned voice + scam script.** Caller → *Attacker tools* → **Scam call: robot voice asks for OTP** → *Play into the call*, then *Send transfer request*. The customer screen goes **HOLD** within about 5 s of speech; the transcript highlights "account will be blocked", "share the OTP"; **the payment is blocked**; **Incidents** shows it with its reasons, and the investigation view (`/#/call/<id>`) shows the evidence.
4. *(Step 9)* A real clone of the presenter's voice played into the call.

## Step 6 results (10 Nov 2026)
| Call | Verdict | Synthetic voice | Scam language |
|---|---|---|---|
| Genuine callers 1 and 2, laptop-mic voice | PASS | 1–6 | 0–40 |
| ElevenLabs clone (no scam words) | VERIFY | 41 | 0 |
| Kokoro clone (no scam words) | HOLD | 82 | 0 |
| Scam call, robot voice asks for OTP | **HOLD at 4.4 s** | 100 | 84 |

Speech-to-text: Whisper small on the RTX 3050 handles 8 s of speech in 0.2–0.3 s. Voiceprints: same person 0.6–0.87 similarity, different people 0.0–0.25.

## Step 5 results (9 Nov 2026, laptop CPU i5-12500H)
- **Speed:** about 330 ms of detector time per 4 s window on CPU. The window is re-checked after every 1 s of speech, so it keeps up in real time with no GPU.
- **Demo calls:** none of these recordings were used for calibration. Profile `studio`, cut-off -5.63, from dev data with 2.1% window-level EER.

| Demo call | Verdict | First alert | HOLD at | Max risk |
|---|---|---|---|---|
| Genuine caller 1 (34 s) | PASS | none | none | 10 |
| Genuine caller 2 (35 s) | PASS | none | none | 5 |
| Cloned voice: ElevenLabs (16 s) | VERIFY | 3.2 s | none | 71 |
| Cloned voice: Kokoro TTS (18 s) | HOLD | 14.1 s | 16.5 s | 82 |

- **Honest reading:**
  - There were no false alarms on genuine callers, and both clones were caught.
  - Premium clones get **VERIFY** from the detector alone, not always HOLD.
  - The Kokoro clone looked genuine for its first 13 s.
  - Step 6 adds a speaker-voiceprint check and scam-language detection, which push these calls to HOLD.

### The console (`/`, redesigned 20 Nov 2026)
One dark interface with a sidebar. Every view shows real server data only.

| View | Address | What it shows |
|---|---|---|
| **Command center** | `/#/live` | Start the microphone, a demo call or a file, then watch it live:<br>- the verdict, with its reasons;<br>- the AI-voice probability of the latest window, and windows flagged out of the last 5;<br>- a waveform drawn from the exact audio sent to the detector, coloured by the decision at each moment;<br>- a timeline of suspicious segments, the risk chart and the three signals;<br>- the system response, the event feed, the transcript and the window log. |
| **Voice analysis** | `/#/analysis` | Load a demo call or a file:<br>- waveform, spectrogram and per-window timeline, with a playhead;<br>- an Authentic / Suspicious / Spoofed assessment;<br>- the model outputs for every window. |
| **Incidents** | `/#/incidents` (also `/security`) | The security team's live incident table. It refreshes every 2 s, shows a notice on each new HOLD, and adds a badge in the sidebar. |
| **Detection history** | `/#/history` | The audit log, with search, verdict and mode filters, sorting and pages. |
| **Investigation** | `/#/call/<id>` | One session, with the evidence kept apart in three columns:<br>1. the model's prediction;<br>2. the supporting signals (voiceprint, scam tactics);<br>3. the actions taken.<br>Also the risk chart and the stored window log. |
| **Analytics** | `/#/analytics` | Charts from the audit log: outcomes by day, risk distribution, detector latency, scam tactics and prevention actions.<br>Also the offline model evaluation, clearly labelled as such. |
| **Settings & enrolment** | `/#/settings` | Enrol voiceprints, calibrate the microphone, see the line profiles and the engine's rules. |

The design rules live in `static/ui.css`. Status is always shown with an icon and a word, never by colour alone. The chart colours were checked for colour-blind separation. The layout respects reduced motion and works down to phone width.

## Step 8c detector (default since 20 Nov 2026)
The NII detector is **never changed**. One pass of it now also returns the time-average of each of its 25 internal layers, and a small model trained in `notebooks/step8c_frozen_backend.ipynb` (1,050 numbers, in NumPy) reads them:
- **Fused score:** 0.15 × the detector's own score + 0.85 × the back-end score. It is reported on the same log-odds scale as before (`llr`); the detector's own score is still sent as `llr_original`.
- **Audio type:** a second small model guesses how the audio was processed (clean, phone line, noise suppression, spectral subtraction, auto-gain, auto-gain + spectral subtraction, Bluetooth CVSD, Bluetooth mSBC, Opus). The console shows it in the *Audio type* column. The `auto` line profile uses that type's own cut-off for every window.

| Setting in `config.json` | Meaning |
|---|---|
| `"detector_mode": "step8c"` | detector + Step 8c back-end (needs `models/voiceshield_layers.onnx` and `models/step8c_backend.npz` from the Step 8c Kaggle output) |
| `"detector_mode": "original"` | the detector's own score only (`models/wav2vec2-large-antideepfake.onnx`), as before 20 Nov |

If the Step 8c files are missing, the server logs a warning and uses the original detector. A cut-off only fits the score it was set on, so each mode keeps its own line profiles: `data/profiles.json` (original) and `data/profiles_step8c.json` (Step 8c). Switching modes never reuses the other mode's cut-offs.

The checks on real recordings are in `tools/compare_step8c.py`:
- `voiceshield_layers.onnx` gives the same detector score as the original file (largest difference 0.0003 over 190 windows).
- It costs about 5% more CPU time per window (about 340 ms in the live console).

| Recording (full risk engine) | Original `default` / `studio` | Step 8c `default` / `studio` |
|---|---|---|
| Genuine callers 1 and 2 | PASS / PASS | PASS / PASS (max risk ≤ 1.2) |
| ElevenLabs clone | VERIFY / VERIFY | **HOLD at 5.6 s** / **HOLD at 12.3 s** |
| Kokoro clone | VERIFY / HOLD at 16.5 s | **HOLD at 13.1 s** / VERIFY |
| Scam call, robot voice | HOLD at 4.4 s / HOLD at 4.4 s | HOLD at 4.4 s / HOLD at 4.4 s |
| Laptop mic, Windows enhancements ON (2 sessions, `mic`) | false HOLD, false HOLD | VERIFY, false HOLD |

The Step 8c `studio` profile (0.00) was calibrated by the same rule as before: window EER on Hume fakes and 4 dev speakers, never the demo files.

The audio-type model recognises the laptop's Windows noise suppression as "noise suppression", but the false alarm is only partly fixed. **Keep Windows audio enhancements OFF.**

## Line profiles (calibration)
The detector's raw score depends on the audio channel. Clean studio speech scores much more "real" than laptop-mic or phone speech, so one fixed cut-off cannot fit every line. A **line profile** holds the cut-off for one channel:

| Profile | Cut-off from | Use for |
|---|---|---|
| `default` | Step 4 dev sets: Indian-language clones, phone line, In-the-Wild, commercial TTS | anything not yet calibrated |
| `studio` | dev recordings of clean commercial-TTS lines: Hume fakes and 4 dev speakers, never the demo files | the bundled demo calls |
| `mic` | 25 s of **your** voice, recorded with *Calibrate with my voice* | live microphone demos |
| `auto` (Step 8c only) | one cut-off per audio type (held-out speakers), chosen in every window from the type the back-end recognises | lines whose processing is unknown; trades some misses for fewer Opus false alarms |

In Step 8c mode:
- `default` is the pooled development-set EER cut-off of the fused score (−1.91).
- `studio` was recalibrated on the same dev clips as before (0.00; dev window EER 0.4%, against 2.1% for the original detector).
- `mic` was recalibrated on the same two enhancements-OFF sessions as the original `mic` profile (−1.64).

From the command line (profiles are saved for the current `detector_mode`):

```bash
python tools/calibrate.py --save-as mic --genuine my_voice_*.wav
```

```bash
python tools/calibrate.py --save-as studio --genuine dev_real/*.flac --fake dev_fake/*.flac
```

Rules:
- With genuine speech only, the cut-off is the 95th percentile of the genuine window scores plus 1.5.
- With genuine and fake recordings, the cut-off is the equal-error-rate point.
- Never calibrate on the recordings you demo with.

## Tools
| Command | What it does |
|---|---|
| `python tools/score_file.py --profile studio samples/*.flac` | runs the live pipeline on files without a server and prints the verdict, max risk and time to alert |
| `python tools/score_file.py --timeline my.wav` | prints every analysed window |
| `python tools/stream_file.py samples/cloned_call_elevenlabs.flac --profile studio` | streams a file to a running server in real time, like a call |
| `python tools/build_demo.py` | rebuilds `samples/` and the `studio` profile from the public dataset: dev split for calibration, test split for the demo calls |
| `python tools/build_demo.py --profile-only` | only recalibrates `studio` for the current `detector_mode` |
| `python tools/compare_step8c.py files...` | original detector vs Step 8c on the same speech windows: scores, audio type, CPU time |
| `python -m pytest -q` | runs the tests: fast stub tests always, real-model tests once the models are downloaded |

## WebSocket protocol (for the Step 7 call app)
1. Client sends text `{"type":"start","sample_rate":16000,"source":"...","profile":"studio","mode":"monitor"}`.
   - `mode: "calibrate"` with `"save_as": "mic"` records a line profile instead.
2. Client sends binary little-endian int16 mono PCM, in about 100 ms chunks. Any sample rate is accepted; the server resamples.
3. Client sends text `{"type":"stop"}`.

Server messages:
- `ready`
- `vad`: speech yes/no, about every 0.25 s
- `score`: per window. Contains `llr`, `p_fake`, `window_score`, `window_threshold`, `risk` (0–100), `level`, `verdict` and `latency_ms`. In Step 8c mode it also has `llr_original` (the detector's own score) and `processing` (the recognised audio type).
- `action`: when the level first rises to VERIFY or HOLD, with the prevention message
- `summary`: at the end. Calibration mode sends `calibrated` instead.

## Files
```
run.py               start the server
setup_models.py      download the ONNX models
config.json          window/hop, VAD and risk-engine settings, audit DB, optional webhook_url
server/              app.py (FastAPI + WebSocket), stream.py (VAD windows, resampler), models.py,
                     risk.py, calibration.py (line profiles), audit.py (SQLite), offline.py
static/              ui.css (design system, dark) + shell.js (sidebar, icons, toasts, gauge, risk chart, live waveform,
                     segment lane, spectrogram, SVG charts); console (index.html/app.js, 7 views) + call demo:
                     customer.html/js, caller.html/js; security.html redirects to the Incidents view;
                     vs-client.js (shared WebRTC + analysis client). No external libraries or fonts; works offline
samples/             demo calls + samples.json
tools/               score_file.py, calibrate.py, stream_file.py, build_demo.py, compare_step8c.py
tests/               test_core.py (stub models), test_models.py (real models)
data/                profiles.json / profiles_step8c.json (line profiles per detector mode), audit.db (run time)
models/              ONNX models; Step 8c adds voiceshield_layers.onnx + step8c_backend.npz (Kaggle output)
```

## Licences
- **NII AntiDeepfake** detector (`SpeechAntiSpoofingBenchmarks/Wav2Vec2-Large-AntiDeepfake`): CC BY-NC-SA 4.0. Fine for research and education; not for commercial use.
- **Silero VAD:** MIT.
- **Demo clips:** cut from `garystafford/deepfake-audio-detection`, CC BY 4.0. See `samples/README.md`.
