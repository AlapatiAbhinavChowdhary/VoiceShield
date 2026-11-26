<div align="center">

<img src="docs/assets/banner.svg" alt="VoiceShield: real-time detection and prevention of voice-cloning scams in phone calls" width="100%">

<br>

![SIH 2026](https://img.shields.io/badge/SIH_2026-SIH26104-22d3ee?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-WebSocket-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-CPU-005CED?style=for-the-badge&logo=onnx&logoColor=white)
![WebRTC](https://img.shields.io/badge/WebRTC-live_call-333333?style=for-the-badge&logo=webrtc&logoColor=white)
![Tests](https://img.shields.io/badge/tests-50_passing-10b981?style=for-the-badge)
![Use](https://img.shields.io/badge/use-non--commercial-f59e0b?style=for-the-badge)

**[📖 Plain-language explainer](explain.md)** &nbsp;·&nbsp; **[🚀 Quick start](#-quick-start)** &nbsp;·&nbsp; **[🔬 Research report](docs/SIH26104_Research_and_Build_Plan.md)** &nbsp;·&nbsp; **[📄 Paper](docs/paper/voiceshield_paper.tex)** &nbsp;·&nbsp; **[📓 Notebooks](notebooks/)**

</div>

<img src="docs/assets/stats.svg" alt="Headline results: HOLD in 4.4 s, error 9.7% to 4.1%, ElevenLabs clone VERIFY to HOLD, no audio stored" width="100%">

## 🛡️ What VoiceShield does

Voice cloning lets a scammer speak in the voice of a bank officer, a police officer or your own family member. VoiceShield **listens to the call while it happens** and decides **every second**:

<div align="center">

| | Verdict | What happens |
|:---:|---|---|
| 🟢 | **PASS** · authentic | Payments allowed, keep listening |
| 🟡 | **VERIFY** · suspicious | Confirm another way: an OTP to the customer's *own* phone, or a call-back on the official number |
| 🔴 | **HOLD** · likely spoofed | The bank app **blocks the payment automatically** and the security team is alerted |

</div>

It does this with **three independent checks** that cover for each other:

| | Check | Model | Catches |
|:---:|---|---|---|
| 🤖 | **AI-generated voice** | NII AntiDeepfake wav2vec 2.0 Large (frozen) + our **Step 8c** back-end on its 25 layers | Cloned and synthetic voices, even after noise suppression, Bluetooth or VoIP compression |
| 🧬 | **Voice ≠ claimed person** | WeSpeaker ResNet34 voiceprint vs. an enrolled customer | Human impersonators and clones of someone else |
| 💬 | **Scam language** | Whisper speech-to-text + 9 scam tactics in English, Hindi and Telugu | "Share the OTP", "account will be blocked", "digital arrest", "don't tell anyone"… |

> A matching voice never *lowers* the risk, and scam words from a genuine voice only ever reach VERIFY. HOLD needs a suspicious voice. Every alert explains itself in plain words.

---

## 🎬 See it in action

<div align="center">
<img src="docs/assets/ui-command-center.png" alt="Command center during a real scam call: HOLD at 4.4 s, live waveform turning red, scam tactics recognised" width="100%">
<sub><b>Command center</b>, a real run of the demo scam call. The robot voice reads a bank-fraud script and is <b>held after 4.4 s</b>. All four scam tactics are recognised, and the system response switches to “Hold transactions, alert security”.</sub>
</div>

<br>

<div align="center">
<img src="docs/assets/ui-command-center-wave.png" alt="Live waveform coloured by verdict, suspicious-segment timeline, risk chart and event feed" width="100%">
<sub>The <b>live waveform</b> is drawn from the exact audio sent to the detector. It turns cyan → amber → red as the verdict changes. Below it: every analysed 4-second window, the risk curve, and the event feed.</sub>
</div>

<br>

<table>
<tr>
<td width="50%"><img src="docs/assets/ui-voice-analysis.png" alt="Voice analysis: waveform, spectrogram and per-window timeline of an ElevenLabs clone"><br><sub><b>🔍 Voice analysis</b>: waveform, spectrogram and per-window verdicts for a premium ElevenLabs clone. Assessment: <b>Spoofed</b>.</sub></td>
<td width="50%"><img src="docs/assets/ui-investigation.png" alt="Investigation view separating model prediction, supporting signals and security actions"><br><sub><b>🕵️ Investigation</b>: the evidence kept apart in three columns: <b>(1)</b> what the detector predicted, <b>(2)</b> supporting signals, <b>(3)</b> actions actually taken.</sub></td>
</tr>
<tr>
<td width="50%"><img src="docs/assets/ui-analytics.png" alt="Analytics from the audit log"><br><sub><b>📊 Analytics</b> from the real audit log (outcomes, risk distribution, latency, scam tactics), plus the offline model evaluation.</sub></td>
<td width="50%"><img src="docs/assets/ui-incidents.png" alt="Incidents table for the security team"><br><sub><b>🚨 Incidents</b>: the bank security team's live view. It refreshes every 2 s and pops up a notice on every new HOLD.</sub></td>
</tr>
<tr>
<td width="50%"><img src="docs/assets/ui-customer-hold.png" alt="Customer phone during the Step 9 attack: HOLD and the 49,999 rupee payment blocked"><br><sub><b>📱 Customer phone</b> during the Step 9 attack: <b>HOLD</b>, and the bank app <b>blocks the ₹49,999 payment</b> automatically.</sub></td>
<td width="50%"><img src="docs/assets/ui-caller-attack.png" alt="Caller page playing a cloned voice into the call"><br><sub><b>☎️ Caller</b>: the attacker plays a clone of the enrolled person's own voice <i>into</i> the live call.</sub></td>
</tr>
</table>

<details>
<summary><b>📱 It also works at phone width</b></summary>
<br>
<div align="center"><img src="docs/assets/ui-mobile.png" alt="Command center on a phone" width="300"></div>
</details>

---

## ⚙️ How it works

```mermaid
flowchart LR
    A["Caller<br/>genuine or attacker"] -->|"WebRTC call, Opus"| B["Customer phone page"]
    B -->|"received audio, 16-bit PCM<br/>WebSocket, every 100 ms"| C["Resample to 16 kHz<br/>Silero VAD keeps speech only"]
    C --> D["4 s window<br/>every 1 s of speech"]
    C --> U["Utterances<br/>split at pauses, up to 8 s"]
    D --> G1["AI-voice detector<br/>wav2vec 2.0 + Step 8c back-end"]
    D --> G2["Voiceprint check<br/>WeSpeaker ResNet34"]
    U --> G3["Whisper transcript<br/>9 scam tactics"]
    G1 --> R{"Risk engine"}
    G2 --> R
    G3 --> R
    R -->|PASS| P["Payments allowed"]
    R -->|VERIFY| V["OTP to own phone<br/>or call-back"]
    R -->|HOLD| H["Payment blocked<br/>security alerted"]
    R --> L[("Audit log<br/>scores and decisions only")]
    style R fill:#0e7490,stroke:#22d3ee,color:#fff
    style H fill:#9f1239,stroke:#f43f5e,color:#fff
    style V fill:#92400e,stroke:#f59e0b,color:#fff
    style P fill:#065f46,stroke:#10b981,color:#fff
```

**One scam call, step by step**, as it happened in our end-to-end test:

```mermaid
sequenceDiagram
    autonumber
    participant S as Scammer (caller page)
    participant C as Customer phone
    participant V as VoiceShield server
    participant B as Bank app
    participant O as Security team
    S->>C: Plays an AI voice into the WebRTC call
    C->>V: Streams the audio it receives (never stored)
    V-->>C: Every second: PASS, risk 0-100, reasons
    Note over V: AI-voice windows flagged, "share the OTP" heard
    V-->>C: VERIFY at 3.2 s, HOLD at about 4 s of speech
    S->>B: Asks for a Rs 49,999 transfer
    B-->>S: Blocked automatically, the call is on HOLD
    V->>O: Incident with reasons, tactics and TRANSFER_BLOCKED
```

<details>
<summary><b>The decision rules (risk engine)</b></summary>

| Result | When |
|---|---|
| 🔴 **HOLD** | AI-voice risk ≥ 75 **and** 3 of the last 5 windows look fake |
| 🔴 **HOLD** | Scam language ≥ 50 **and** the voice or the identity was suspicious at any point in the call |
| 🟡 **VERIFY** | AI-voice risk ≥ 45, or 2 suspicious windows, or voiceprint mismatch ≥ 50, or scam language ≥ 50 |
| 🟢 **PASS** | None of the above |

- **Window score:** `s = σ(score − line cut-off)`. The AI-voice risk is a running average that starts at 0, so one odd window cannot raise an alarm.
- **Escalation only:** the verdict only goes up during a call, never back down.
- **Displayed risk:** the noisy-OR of the three signals.

</details>

---

## 🔬 Research highlights

### 1 · Everyday audio processing breaks deepfake detectors, and we repaired it without retraining

Phones, laptops, Bluetooth headsets and calling apps change the sound before anyone hears it. Noise suppression raised the detector's error on premium clones from **8% to 34%**, and WhatsApp-style **Opus** compression raised it to **45%**, close to a coin toss. **A scammer gets this disguise for free.**

Retraining made the detector *forget* real-world fakes, three times over. So in **Step 8c** we kept the detector **frozen** and trained a 1,050-number back-end that reads its internal layers instead:

<img src="docs/assets/chart-step8c.svg" alt="EER before and after the Step 8c back-end on 10 test conditions" width="100%">

### 2 · The evidence is still inside the model

Simple probes on each of the detector's 25 layers show that the **middle layers keep the clues** that the final read-out loses after processing. The back-end learned to weight exactly those layers.

<img src="docs/assets/chart-layers.svg" alt="Probe EER per layer: middle layers are most robust" width="100%">

### 3 · Fine-tuning versus leaving the detector frozen

| Approach | Trained parameters | Dev mean EER (8 sets) | In-the-Wild dev | Outcome |
|---|---:|---:|---:|:---:|
| Original detector | – | 4.17% | 4.0% | baseline |
| LoRA fine-tuning (Step 8) | 3.54 M | 5.28% | **9.5%** | ❌ forgot |
| LoRA + teacher anchor (Step 8b) | 3.54 M | 5.79% | **8.3%** | ❌ forgot |
| **Frozen detector + layer back-end (Step 8c)** | **1,050** | **1.39%** | **1.75%** | ✅ adopted |

<sub>Development values at each method's best checkpoint. The unseen test sets tell the same story: the In-the-Wild test EER is unchanged at 4.67% with Step 8c.</sub>

### 4 · On real recordings, in the live server

| Recording | Original detector | With Step 8c |
|---|:---:|:---:|
| 🙋 Genuine callers 1 and 2 | 🟢 PASS | 🟢 PASS (lower risk) |
| 🤖 ElevenLabs clone | 🟡 VERIFY | 🔴 **HOLD at 5.6 s** |
| 🤖 Kokoro clone | 🟡 VERIFY | 🔴 **HOLD at 13.1 s** |
| 📞 Robot-voice scam call | 🔴 HOLD at 4.4 s | 🔴 HOLD at 4.4 s |
| 💻 Laptop mic, Windows enhancements ON | ❌ 2 false HOLDs | 🟡 1 VERIFY · ❌ 1 false HOLD |

<sub>Both detectors use their own `default` cut-off. The exported model reproduces the original detector's score to within 0.0003 and adds about 5% CPU time (about 340 ms per window). With the stricter <code>studio</code> profile the Kokoro clone stays at VERIFY. Windows audio enhancements must stay <b>off</b>.</sub>

### 5 · The hardest attack: a clone of the person the caller claims to be

We cloned the enrolled team member's **own voice** locally, with consent, from 2 minutes of that member's recordings (kNN-VC), made it read the bank-fraud script, and attacked the system while claiming to be that member:

| Caller (claims to be the enrolled person) | 🧬 Voiceprint (cut-off 0.45) | 🤖 AI-voice detector | Verdict |
|---|:---:|:---:|:---:|
| **Clone of the member's voice + scam script** | 0.41, *almost* accepted | **18 / 18** windows flagged | 🔴 **HOLD at 4.4 s** |
| … the same clone **through the live WebRTC call** (2 runs) | 0.32 | median score 13 | 🟡 VERIFY 3.8–4.5 s → 🔴 **HOLD 7.8–10 s**, ₹49,999 **blocked** |
| The member's real voice | 0.95, matches | 0 / 19 | 🟢 PASS |

> The voiceprint alone would nearly have let the clone through. The AI-voice detector caught every window, and that is exactly why VoiceShield never relies on one check. Still open: the same attack with a *premium* clone.

---

## 🗺️ The journey

| Step | What we did | Result |
|:---:|---|---|
| 0 | Our own spectrogram CNN and wav2vec 2.0 classifier | ❌ 33–37% error on new audio, so we start from a post-trained detector |
| 2 | Zero-shot test of two detectors | ✅ AntiDeepfake **4.0%** EER on In-the-Wild; a popular Hugging Face model **32–38%** |
| 3 | Same-speaker Hindi / Telugu clones (kNN-VC) | ✅ **0.0–0.1%** EER clean · 🔬 a phone line raises it to 5.2% |
| 4 | Full fine-tuning on 3,760 clips | ❌ forgot real-world fakes, so the original was kept |
| 5 | Live streaming server | ✅ a 4 s window every 1 s of speech, **0.33 s** per window on a laptop CPU |
| 6 | Voiceprint and scam-language fusion | ✅ robot-voice scam call **HOLD at 4.4 s** |
| 6b–d | Live microphone tests | 🔬 Windows enhancements cause false alarms · ChatGPT's voice **HOLD at 5.3 s**, the human reading the same script **PASS** |
| 7 | Two-party WebRTC call with a bank app | ✅ HOLD after about 4 s of speech, ₹49,999 transfer **blocked** |
| 8 | Consumer audio processing study | 🔬 noise suppression / Opus: premium-clone EER **8% → 34% / 45%** |
| 8 · 8b | LoRA fine-tuning, alone and with a teacher anchor | ❌ learned the processing, forgot In-the-Wild (4.0% → 9.5%) |
| **8c** | **Frozen detector + 25-layer back-end** | ✅ mean EER **9.7% → 4.1%**, nothing forgotten |
| **8d** | **Step 8c in the live server** | ✅ ElevenLabs clone **VERIFY → HOLD** |
| UI | Security-console redesign | ✅ live waveform, forensics, investigation, analytics |
| **9** | **Attack: a clone of the enrolled person's own voice** | ✅ the voiceprint nearly accepted it (0.41 vs 0.45), but the detector flagged **18/18** windows: **HOLD**, and ₹49,999 **blocked** through the live call |

<sub>✅ worked &nbsp;·&nbsp; ❌ tried and rejected (reported honestly) &nbsp;·&nbsp; 🔬 research finding. Every number comes from the <a href="docs/SIH26104_Research_and_Build_Plan.md">results log</a>.</sub>

---

## 🧰 Tech stack

| Layer | Technology |
|---|---|
| 🤖 Detector | NII AntiDeepfake wav2vec 2.0 Large (ONNX, CPU) + Step 8c back-end in NumPy |
| 🎙️ Speech gate | Silero VAD v5 |
| 🧬 Voiceprint | WeSpeaker ResNet34-LM (ONNX) with a NumPy filterbank |
| 💬 Speech-to-text | faster-whisper: `small` on the GPU, `base` on the CPU |
| 🖥️ Server | Python 3.12, FastAPI, WebSockets, ONNX Runtime, SQLite audit log |
| 🎨 Frontend | HTML, CSS and JavaScript, Web Audio and AudioWorklet, WebRTC, Canvas and SVG. No frameworks, works **offline** |
| 🧪 Training and research | Kaggle T4 GPUs, PyTorch, Transformers, PEFT. Every notebook is first tested on the CPU |
| ✅ Quality | 41 fast tests (stub models) + 9 real-model tests (pytest) |

---

## 🚀 Quick start

**Already set up? Double-click `START_VOICESHIELD.bat`** in this folder. It checks the laptop, starts the server and opens http://127.0.0.1:8000. Keep its window open; closing it stops VoiceShield.

First-time setup:

```bash
cd voiceshield
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python setup_models.py        # downloads the ONNX models (~2 GB, once)
.venv\Scripts\python run.py                 # then open http://127.0.0.1:8000
```

| Open | For |
|---|---|
| `http://127.0.0.1:8000/#/live` | 🎛️ Command center (microphone, demo calls, files) |
| `/customer` + `/caller` | 📞 The two-window call demo |
| `/#/incidents` | 🚨 Security team view |

> [!TIP]
> Before a live demo: turn **off** Windows audio enhancements, use the built-in or a wired microphone (not Bluetooth), and calibrate once from **Settings & enrolment**.

<sub>The Step 8c layer model (<code>voiceshield_layers.onnx</code>, 1.26 GB) comes from the Step 8c Kaggle notebook. Without it the server falls back to the original detector automatically. Full setup: <a href="voiceshield/README.md">voiceshield/README.md</a>.</sub>

---

## 📁 Repository

```text
VoiceShield/
├── explain.md                 the whole project in plain words
├── voiceshield/               the working system
│   ├── server/                FastAPI + WebSocket, models, risk engine, calibration, audit log
│   ├── static/                the console (7 views) and the call-demo pages
│   ├── tools/                 scoring, calibration, Step 8c comparison
│   ├── samples/               demo calls (CC BY 4.0)
│   └── tests/                 41 fast + 9 real-model tests
├── notebooks/                 Kaggle experiments, Steps 2–8c (+ CPU tests in build/)
├── docs/
│   ├── SIH26104_Research_and_Build_Plan.md   research report and full results log
│   ├── paper/voiceshield_paper.tex           the research paper
│   └── assets/                README graphics (make_readme_figures.py rebuilds the charts)
└── recordings/                result tables (CSV) from the experiments
```

---

## 🔒 Privacy and responsible use

- 🎧 **What is never stored:** call audio is analysed in memory and never written to disk. Transcripts are never logged either.
- 🗄️ **What is stored:** the audit log keeps only scores, decisions, the *names* of scam tactics and prevention events.
- 🧬 **Voiceprints:** stored as 256 numbers and deletable from the console.
- 💻 **Where it runs:** everything runs on the laptop, and nothing is sent to an outside service.
- 🧑‍⚖️ **A human decides:** VoiceShield warns and triggers bank-side checks.
- 📦 **Left out of this repository on purpose:** the large model files, our own voice recordings, enrolled voiceprints and the local audit log.

## 🚧 Limitations and roadmap

- [x] Real-time fusion of AI voice, voiceprint and scam language
- [x] Two-party call demo with automatic payment blocking
- [x] Detector repaired for consumer audio processing (Step 8c), live in the server
- [x] Attack test with a clone of the *enrolled* person's own voice through the call (open-source clone, Step 9)
- [ ] The same attack with a *premium* clone of the enrolled person
- [ ] Calibrate the call channel (`call` profile) with real callers
- [ ] Train with *real* device processing: Windows still causes one false alarm with enhancements on
- [ ] Detector on the GPU, and a real phone as the caller (HTTPS)

## 📚 Learn more

| | |
|---|---|
| 📖 [explain.md](explain.md) | Every step in plain words, with a glossary |
| 🔧 [voiceshield/README.md](voiceshield/README.md) | Install, run, screens, demo script, WebSocket protocol |
| 🔬 [Research report and results log](docs/SIH26104_Research_and_Build_Plan.md) | Every experiment with its numbers (Steps 2–8d) |
| 📄 [Research paper](docs/paper/voiceshield_paper.tex) | LaTeX source, compile on Overleaf |

## 📜 Licences and credits

- **Detector:** NII AntiDeepfake, CC BY-NC-SA 4.0.
- **Indian-language data:** IndicSynth (CC BY-NC 4.0) and Kathbath (CC BY 4.0).
- **Demo clips:** from `garystafford/deepfake-audio-detection` (CC BY 4.0).
- **Other tools:** Silero VAD (MIT), WeSpeaker, Whisper.

This project is for **non-commercial research and education** only. Built for the **Smart India Hackathon 2026** (problem statement SIH26104, AICTE Cyber Security Cell).
