# SIH26104: Research Report and Build Plan
**AI-Powered Real-Time Detection and Prevention of Voice Cloning Impersonation Attacks**
Compiled 9 Nov 2026 from research papers, benchmarks and other SIH26104 team repos.

---

## 0. TL;DR

1. **Stop training from scratch.** A CNN on spectrograms trained on ASVspoof usually scores **33 to 37% EER** on real-world fakes, close to guessing. Your failure was expected, not a bug in your code.
2. **Start from a detector that has already seen a lot of fake speech.** The best free option is NII's **AntiDeepfake** model (`nii-yamagishilab/wav2vec-large-anti-deepfake`). It was post-trained on 56k hours of real speech and 18k hours of fake speech in 100+ languages. Its model card reports **1.91% EER on In-the-Wild with no extra training**. The second choice is **XLS-R + SLS**, the best open model on the Speech DF Arena leaderboard.
3. **In-domain data matters more than the model, and you don't need to record anything.** Take real Indian speech from public corpora (Kathbath, IndicVoices, Svarah). Use ready-made fakes of the same speakers (IndicSynth), and clone more of those same speakers yourself with IndicF5 / XTTS-v2 / kNN-VC. Push both classes through the same simulated phone channel. Fine-tune lightly (LR around 1e-6). Most "wav2vec still gives wrong answers" problems come from data shortcuts, a wrong label index or a wrong threshold. See section 3.
4. **Deepfake detection alone does not solve this problem statement.** Build four layers: a synthetic-voice detector, speaker verification against an enrolled voiceprint, prosody forensics for explanations, and a scam-language check on the transcript. They feed a risk engine that outputs PASS / VERIFY / HOLD and triggers prevention actions.
5. **Stock Android cannot read cellular call audio.** Demo with a WebRTC call that you control. That is what the reliable prototypes do.

---

## 1. What the problem statement asks for

Source: AICTE Cyber Security Cell, theme Blockchain & Cybersecurity, category Software.

| PS requirement | What we build |
|---|---|
| Deep-learning acoustic and spectral analysis for synthesis artifacts | AntiDeepfake / XLS-R-SLS detector on sliding windows |
| Prosody and behavioral modeling (rhythm, pitch, pauses) | librosa/Parselmouth features: jitter, shimmer, HNR, pause stats. Used for explanations, low weight in the score |
| Cross-session checks against historical genuine samples | Speaker verification (ECAPA-TDNN) against the enrolled voiceprint of the claimed caller |
| Real-time risk scoring engine, configurable thresholds | Smoothed fused score, with thresholds set per scenario (e.g. high-value transfer) |
| Contextual enrichment (call origin, known contacts, transaction) | Rule features: unknown number, amount, time of day. Plus the scam-language score from the transcript |
| Alerts (UI, SMS/email, in-app), pre-transaction prompts | Dashboard alert, webhook, SMS/email (mock or Twilio). Prompts for call-back, MFA or supervisor escalation |
| Privacy: minimal retention, edge inference, feature-only logs | No raw audio stored, only scores and features in logs. ONNX/int8 export for on-device inference |
| REST/gRPC APIs and SDKs | FastAPI REST + WebSocket stream (gRPC optional) |
| Multilingual, Indian accents | Multilingual SSL backbone + Indian-language training data (IndicSynth, Kathbath, IndicF5 clones) |

---

## 2. What the research says

### 2.1 Spectrogram CNNs don't generalize
- **Müller et al., "Does Audio Deepfake Detection Generalize?" (Interspeech 2022).** Models trained on ASVspoof jumped from low EER to roughly 34 to 91% EER on the In-the-Wild dataset (real celebrity and politician deepfakes). The spectrogram models LCNN and MesoNet reached about 37% EER on In-the-Wild.
- **Speech DF Arena (2025)** is a 14-dataset leaderboard with 17 open-source systems. Raw-waveform and small models (RawNet2, RawGAT-ST, AASIST) averaged **33 to 36% EER**. SSL-based mid-size models were clearly best.

| Open-source model (Speech DF Arena) | Avg EER | Pooled EER |
|---|---|---|
| **XLSR + SLS** | **13.84%** | **15.68%** |
| TCM | 15.77% | 16.35% |
| Nes2NetX | 16.11% | 17.04% |
| Wav2Vec2-AASIST (Tak 2022) | 18.02% | 19.47% |
| XLSR-Mamba | 14.21% | 20.12% |
| Whisper-MesoNet | 28.69% | 23.76% |
| AASIST | 34.49% | 33.16% |
| RawNet2 | 35.75% | 35.66% |

### 2.2 Data scale beats architecture
- **AntiDeepfake (Ge, Wang, Liu, Yamagishi, ASRU 2025, arXiv 2506.21090)** post-trains wav2vec2/HuBERT/MMS on 56k hours real and 18k hours fake. Zero-shot results from the `wav2vec-large-anti-deepfake` model card:

| Test set | AUC | EER | Threshold at EER |
|---|---|---|---|
| In-the-Wild | 0.997 | **1.91%** | 0.33 |
| FakeOrReal | 1.000 | 0.67% | 0.36 |
| DeepVoice | 0.991 | 4.44% | 0.997 |
| ADD2023 | 0.950 | 13.25% | 0.85 |
| Deepfake-Eval-2024 | 0.741 | 33.36% | 0.9995 |

  After fine-tuning on the Deepfake-Eval-2024 training split, it reached EER 10.52% (30 s input).
  - **Important:** the best threshold ranges from 0.33 to 0.9995 depending on the dataset. A fixed 0.5 argmax gives wrong answers. **Calibrate the threshold on your own dev set.**
  - License: CC BY-NC-SA 4.0 (non-commercial). This is fine for a capstone or hackathon. Say so in the report.

### 2.3 Real-world fakes are still hard, so fine-tune on your target domain
- **Deepfake-Eval-2024 (arXiv 2503.02857)** collected real deepfakes from social media in 52 languages. Open-source audio detectors lost about 48% AUC compared with academic benchmarks. Fine-tuning on in-domain data closed most of that gap: a fine-tuned audio model reached 86% accuracy.
- **"Audio Deepfake Detection in the Age of Advanced TTS" (arXiv 2601.20510)** tested new TTS systems. XLS-R-SLS scored 7% EER on Dia2 but 27% on MeloTTS. Whisper-MesoNet showed the reverse pattern. **No single detector wins on every generator**, so ensembles and multi-view systems help.
- **RADAR Challenge 2026** used codecs, noise, music and reverb across many languages. The Tak 2022 W2V2-AASIST baseline fell to **42.6% EER**. Strong development scores did not carry over to unseen transformations.
- Speech DF Arena found that adding **reverberation raises EER by about 50% (relative)** for every model.

### 2.4 Shortcut learning: the most common hidden bug
- **"Speech is Silver, Silence is Golden" (Müller et al., 2021).** On ASVspoof 2019, the length of leading silence alone predicts the label. A model that only sees silence duration gets 3.6% EER. Trimming silence pushed RawNet2's DF EER from 6.28% to 20.79%. In other words, models learn the silence, not the voice.
- The same thing happens with **sample rate, codec, bitrate, loudness, language and recording device**. If all your real clips come from one source and all your fake clips from another, the model learns the source.

### 2.5 Telephony and real-time
- **RTCFake (ACL Findings 2026, arXiv 2604.23742)** is about 600 hours of real and fake speech sent through real RTC platforms, which apply noise suppression and codecs. The authors note that offline-trained detectors break down after RTC transmission. Dataset: `JunXueTech/RTCFake` on Hugging Face.
- Commercial practice: Pindrop scores **2-second segments** to stay under a second of latency. Deployments at the edge or on the agent device take about 50 to 200 ms per decision.
- SATYAVAANI (another SIH26104 team) gets 18 ms per 3-second window on a laptop GPU with wav2vec2-base, and about 270 ms on CPU.

### 2.6 Speaker verification alone is fooled by clones
- **SASV Challenge 2022.** ECAPA-TDNN reaches 1.86% EER at telling speakers apart, but **30.75% EER** against spoofed speech. Countermeasures such as AASIST are the opposite: good at spotting spoofs, useless for identity (about 50% SV-EER). **Fusing both** brought the combined SASV-EER to about 1.7%.
- SATYAVAANI measured that **43% of XTTS clones fooled their speaker check** on phone-line audio. So "voice matches" must **never lower** the risk score.

### 2.7 Prevention (proactive)
- **AudioSeal (Meta, arXiv 2401.17264)** is a localized watermark. It proves that audio is AI-generated or that it is *your* audio, but it cannot stop cloning. A notch filter or speech enhancement can strip it.
- **AntiFake / CloneShield** add adversarial noise to published voice samples so cloning tools produce poor clones. Purifiers such as De-AntiFake can neutralize them. These make good optional demo features but cannot be the core of the system.
- Practical prevention in the problem statement is about **workflow**: hold the transaction, ask for an out-of-band call-back, require MFA or supervisor approval.

### 2.8 Context: scam-language detection
- Real-time LLM warnings during phone scams were found useful by users ("It Warned Me Just at the Right Moment", arXiv 2502.03964).
- Indian data: `DatasetNewUser/Indian_Cyber_Scam_PhoneCall_Hinglish_Dataset` on Hugging Face has about 10k labelled English/Hindi/Hinglish scam and non-scam records (Apache-2.0).

---

## 3. Why your two attempts probably failed

| Symptom | Likely cause | Fix |
|---|---|---|
| CNN does well on the test split but fails on new audio | Spectrogram CNNs don't generalize (about 37% EER in the wild). Possibly also the silence or source shortcut | Drop it. Use an SSL model |
| wav2vec gives "wrong" results | **Backbone LR too high** (1e-4 destroys pretrained features; use about 1e-6 for the backbone, about 1e-4 for the head) | Discriminative learning rates, few epochs, early stopping on an **out-of-domain** dev set |
| | **Frozen backbone with only the last layer pooled.** Artifacts sit in the middle layers | Fine-tune the whole model, or use layer-weighted pooling (the SLS idea) |
| | **Wrong preprocessing**: not 16 kHz, stereo, int16 not scaled, no layer norm | 16 kHz mono float32 in [-1, 1], normalized as the model card says |
| | **Wrong label index**: AntiDeepfake outputs `[fake, real]`, other HF models use other orders | Check `id2label` and verify on 5 known clips before anything else |
| | **Threshold at 0.5** | Pick the threshold at EER or at a target false-alarm rate on your own dev set |
| Everything gets flagged fake (or real) on mic recordings | Domain mismatch: trained on clean TTS files, tested on mic/phone audio, or on a **replay** through speakers | Train with phone-quality real speech (IndicVoices 8 kHz conversational, RTCFake) plus codec augmentation on both classes |
| High validation accuracy, poor demo | **Leakage**: same speakers or sentences in train and test, or real and fake from different sources | Split by speaker and by generator. Make real and fake from the same speakers and sentences. Trim silence for both classes |
| Unstable on short clips | Clips under 2 s | Score at least 3 to 4 s windows. Mark short clips "uncertain" |

> Send us what you used before (dataset, model ID, how you split train/test, how you tested on new audio). The exact cause is almost certainly in this table.

---

## 4. Recommended system architecture

```
┌──────────── Client (React) ────────────┐
│ WebRTC call between 2 browsers, or mic │──16 kHz PCM, 0.5 s frames──┐
│ Live risk gauge · alerts · transcript  │                            │ WebSocket
└────────────────────────────────────────┘                            ▼
                                       ┌─────────── FastAPI backend ───────────┐
                                       │ Ring buffer → Silero VAD (skip silence)│
                                       │ 4 s window, 1 s hop                    │
                                       │  ├─ A. Synthetic detector (AntiDeepfake│
                                       │  │      wav2vec-large, fine-tuned)     │→ p_fake
                                       │  ├─ B. Speaker verification (ECAPA)    │→ p_mismatch
                                       │  ├─ C. Prosody/DSP forensics (explain) │→ z-scores
                                       │  └─ D. Streaming ASR (faster-whisper)  │
                                       │         → scam-language classifier     │→ p_scam
                                       │ Risk engine: EMA smoothing + fusion    │
                                       │   + context (unknown no., amount)      │
                                       │   → PASS / VERIFY / HOLD               │
                                       │ Action engine: webhook, SMS/email,     │
                                       │   MFA/call-back prompt, audit log      │
                                       │   (features only, no raw audio)        │
                                       └────────────────────────────────────────┘
```

### Risk engine (starting design; tune on your dev set)
- Per window: `s_df = calibrated p_fake`, `s_spk = 1 - sigmoid(a*cos + b)` if a voiceprint is enrolled, `s_scam` from the transcript classifier.
- Smoothing: `R_t = 0.6*R_{t-1} + 0.4*max(w1*s_df, w2*s_spk) + w3*s_scam + context_bonus`. Hold `R` during silence.
- Rules learned from other teams:
  - A matching voice **never lowers** risk.
  - One odd window should not turn the call red; require k of the last n windows.
  - Thresholds depend on scenario. Example: VERIFY at 45 and HOLD at 75 normally, lower during "fund transfer" context.
- Calibration: Platt or isotonic regression on your dev set, so 0.7 really means about 70%.

### Prevention actions (what "prevention" means in the demo)
- **VERIFY**: on-screen prompt "Call back on the registered number", OTP or MFA push, or a liveness question (unexpected personal question; ask them to repeat a random phrase *immediately*. Real-time voice conversion can pass this, so it is only one signal).
- **HOLD**: block the pending transaction via webhook, notify a supervisor (email/SMS), log an incident.
- Optional extra: AudioSeal-watermark an organization's official outbound voice so customers can verify genuine calls.

---

## 5. Data plan (no recording needed)

**Rule: every source of real audio has a matching source of fake audio from the same speakers, channel and format.** Otherwise the model learns "which dataset is this" instead of "is this cloned".

### 5.1 Real speech (Indian languages and accents)
| Dataset | What | Size | Access / license |
|---|---|---|---|
| **Kathbath** (`ai4bharat/Kathbath`) | Read speech, 12 Indian languages | 1,684 h, 1,218 speakers. Hindi 150 h (121 speakers), Telugu 155 h (104 speakers) | Hugging Face, gated (accept terms). CC-BY-4.0 |
| **IndicVoices** (`ai4bharat/IndicVoices`) | Read, extempore and **conversational** speech, 22 languages. The conversational part is **8 kHz** (phone-like) | 23.7k h, 51k speakers | Hugging Face, CC-BY-4.0. Use `streaming=True` so you don't download everything |
| **Svarah** (`ai4bharat/Svarah`) | **Indian-accented English**, read and conversational | 9.6 h, 117 speakers | Hugging Face (needs HF token). CC-BY-4.0 |
| Mozilla Common Voice / Google FLEURS | Crowd-sourced Hindi, Telugu, Tamil and more (noisy, many mics) | varies | CC-0 / CC-BY |

### 5.2 Ready-made fakes (with matching real data)
| Dataset | Real side | Fake side | Notes |
|---|---|---|---|
| **IndicSynth** (`vdivyasharma/IndicSynth`) | IndicSUPERB = **Kathbath** speakers | XTTS-v2, VITS and FreeVC24 clones. Hindi 206k clips, Telugu 170k, Tamil 282k | 845 GB in total, so load one language (`name="Hindi"`) and take a sample. CC BY-NC 4.0 |
| ASVspoof 2019 LA | bona fide | 2019-era TTS/VC | Standard baseline. Trim silence (see 2.4) |
| **ASVspoof 5** (Zenodo, DOI 10.5281/zenodo.14498691) | MLS English, crowd-sourced | 2024-era attacks, codec conditions | Newer than 2019. Free |
| **MLAAD + M-AILABS** | M-AILABS (8 languages) | MLAAD: 200+ TTS models, 54 languages | MLAAD v8+ is CC BY-NC |
| **SpoofCeleb** | VoxCeleb1 in-the-wild speech (1,251 speakers) | 23 TTS/VC systems | Real-world noise and channel |
| **RTCFake** (`JunXueTech/RTCFake`) | LibriHeavy speech sent through real RTC platforms | TTS/VC sent through the same platforms | Closest to a real call. About 600 h, so use a subset |
| Fake-or-Real, `for-rerec` | Real speech | Commercial TTS (Polly, Azure, Google and others), **played through a speaker and re-recorded** | Kaggle mirror. GPLv3 |

### 5.3 Make your own clones without recording yourselves
Use clips from Kathbath, Svarah or IndicVoices as the **voice reference**, then generate fakes of the **same speakers** with:
- **IndicF5** (`ai4bharat/IndicF5`, 11 Indian languages, MIT)
- **XTTS-v2** (English + Hindi)
- **F5-TTS** (English)
- **kNN-VC** / OpenVoice (voice conversion: turns a real utterance into the target speaker's voice)

About 2k to 5k clips per generator is enough. Keep one generator out of training entirely (e.g. kNN-VC) so you can report an "unseen generator" EER. This replaces the "record your team" step.

### 5.4 Test-only sets (never train on these)
- **In-the-Wild** (`mueller91/In-The-Wild`): real celebrity deepfakes, English.
- **garystafford/deepfake-audio-detection**: 933 real + 933 fake clips from ElevenLabs, Polly, Hume, Speechify and Kokoro (modern commercial TTS). CC-BY-4.0.
- Your held-out generator and held-out speakers.
- IndicVoices conversational (8 kHz) real speech, plus your clones passed through the same phone channel.
- **IndieFake**: Indian-English speakers with ElevenLabs/Play.ht/Polly fakes, 27 h. The paper says it will be public, but there was no download link yet. Email the IIT Ropar authors if you want it.

### 5.5 Suggested starter mix (fits Kaggle)
| Source | Real clips | Fake clips |
|---|---|---|
| ASVspoof 2019 LA train | about 2.5k | about 5k (subsample) |
| Kathbath (Hindi + Telugu) / IndicSynth (Hindi + Telugu) | about 8k | about 8k |
| Your clones (IndicF5 + XTTS-v2) of Kathbath/Svarah speakers | (the reference speakers' real clips are already counted above) | about 5k |
| MLAAD / M-AILABS subset | about 5k | about 5k |
| RTCFake subset | about 3k | about 3k |
| IndicVoices conversational 8 kHz / your clones through the phone channel | about 3k | about 3k |

That is roughly 20k+ real and 30k fake. Use weighted cross-entropy for the imbalance.

**Augmentation (on the fly, applied to both classes):** RawBoost; codecs (MP3 32 to 128 kbps, Opus 8 to 24 kbps, AMR-NB, G.711 µ-law); 8 kHz band-limit then resample to 16 kHz; noise (MUSAN) at 5 to 20 dB SNR; reverb (RIR).

**Sample-rate trap:** IndicVoices conversational audio is 8 kHz. If only your *real* clips are narrowband, the model learns "narrowband = real". Band-limit a random 30 to 50% of **both** classes to 8 kHz. Do the same with codecs.

**Preprocessing:** 16 kHz mono; trim leading and trailing silence (`librosa.effects.trim`, fixed top_db) on **both** classes; loudness-normalize; random 4 s crops for training.

**Splits:** by speaker *and* by generator. Report EER on:
1. in-domain
2. unseen speaker
3. unseen generator
4. phone channel (IndicVoices 8 kHz real + phone-simulated clones; RTCFake)
5. In-the-Wild

That table makes a strong capstone report. Most other SIH teams report a single accuracy number or none.

---

## 6. Model plan

| Stage | What | Why |
|---|---|---|
| **M0: baseline (day 1-2)** | Run `wav2vec-large-anti-deepfake` (and optionally XLS-R-SLS) zero-shot on your test set | Tells you how good "free" already is. Gives the first table in the report |
| **M1: fine-tune** | Fine-tune AntiDeepfake on the data mix in section 5. Backbone LR 1e-6, head 1e-4, AdamW, weighted cross-entropy, RawBoost + codec augmentation, 4 s crops, fp16, early stop on **unseen-generator** dev EER | Adapts to Indian languages, your channel and modern cloners |
| **M2: ensemble (optional)** | Average calibrated scores of AntiDeepfake-FT + XLS-R-SLS (or + Whisper-MesoNet) | Different detectors fail on different generators |
| **M3: deploy** | Export to ONNX, int8 quantization, measure latency on CPU and GPU | Supports the PS's "edge inference" point |
| Speaker verification | `speechbrain/spkrec-ecapa-voxceleb`, cosine score, threshold re-tuned on Indian-accent phone audio | Covers "cross-session checks". Off the shelf, no training |
| ASR | `faster-whisper` (small or medium, multilingual) on VAD segments | Transcript for scam-language detection and the UI |
| Scam classifier | Fine-tune `xlm-roberta-base` / MuRIL on the Hinglish scam dataset, plus keyword rules (OTP, urgent, police, account blocked, transfer) | Context layer. Works for Hindi, Hinglish and English |

**Compute:** Kaggle (free 2×T4, about 30 h/week) or Colab. `fairseq` (needed by AntiDeepfake and SLS) installs badly on Windows. Train on Kaggle/Colab or WSL2 Ubuntu, and serve the backend from Docker or WSL.

---

## 7. Step-by-step build plan (about 6 weeks; compress if needed)

| Week | Deliverable |
|---|---|
| **1** | Set up the repo and Kaggle notebooks. Build test set v0 from public data: an In-the-Wild sample, the garystafford set, Svarah + Kathbath real clips, and clones of those speakers from 2 generators. Run M0 baselines; produce EER table v0 |
| **2** | Data pipeline: download and format ASVspoof19, IndicSynth (hi/te/ta) + Kathbath, MLAAD subset, RTCFake subset. Augmentation module. Leakage checks (speaker overlap, silence trim) |
| **3** | Fine-tune M1 on Kaggle. Evaluate on the 5 test conditions. Calibrate the threshold |
| **4** | Backend: FastAPI WebSocket streaming, VAD, sliding window, detector + ECAPA + faster-whisper, risk engine, SQLite audit log (features only) |
| **5** | Frontend: WebRTC two-party call page, live risk gauge, transcript with highlighted scam phrases, explanation panel (prosody charts), enrollment page, admin page for thresholds and workflows. Webhook/SMS/email mock |
| **6** | ONNX export and latency benchmark. Red-team demo (live clone via XTTS/kNN-VC played into the call). Write the report: results table, ablations, limitations, privacy design |

**Demo script:**
1. A genuine caller gets PASS.
2. A cloned "bank manager" voice asks to transfer money: risk rises, then HOLD, the transaction is blocked, and an alert is sent.
3. A real human impersonator triggers a speaker mismatch, then VERIFY with a call-back prompt.

---

## 8. Other SIH26104 teams (what to beat)

| Repo | Approach | Note |
|---|---|---|
| asmit-1101/satyavaani | frozen wav2vec2-base + layer-weighted head + speaker head → PASS/VERIFY/HOLD | Best documented. 4.6% EER in-domain but **about 40% EER on an unseen generator** |
| Tejahudson/voice-cloning-detector | `garystafford/wav2vec2-deepfake-voice-detector` + librosa forensics | Not evaluated on an independent dataset. Upload-based, not a live call |
| RYashraj/voxguard | wav2vec2/AASIST + prosody, chunked | |
| kalyanvijay05/SIH26104-voice-security | detection + speaker verification + explainable risk | |
| kanakpadotra2101-sketch/SENTINEX-SIH26104 | MFCC → CNN on ASVspoof19 | Frontend only |

**How to stand out:** a stronger pretrained detector, honest cross-generator, Indian-language and phone-channel numbers, a real WebRTC live-call demo, and working prevention workflows.

---

## Results log

### Step 2: zero-shot baseline (9 Nov 2026)
Setup: Kaggle T4, 4.04 s window (64,600 samples), trimmed silence. 300 real + 300 fake per set, except Hindi: 50 real (10 speakers) + 300 fake.

| Model | Test set | EER % | AUC | Threshold at EER | Accuracy at 0.5 % |
|---|---|---|---|---|---|
| A: AntiDeepfake W2V-Large | Hindi (Kathbath vs IndicSynth) | 0.0 | 1.000 | 0.895 | 100.0 |
| A: AntiDeepfake W2V-Large | Commercial TTS (garystafford) | 10.0 | 0.963 | 0.001 | 78.2 |
| A: AntiDeepfake W2V-Large | In-the-Wild | 4.0 | 0.997 | 0.012 | 95.8 |
| B: garystafford wav2vec2 | Hindi (Kathbath vs IndicSynth) | 32.0 | 0.737 | 0.012 | 51.4 |
| B: garystafford wav2vec2 | Commercial TTS (*its own training data*) | 1.3 | 0.999 | 0.653 | 98.8 |
| B: garystafford wav2vec2 | In-the-Wild | 37.7 | 0.686 | 0.003 | 68.5 |

Findings:
- **Our pipeline is correct.** Model A gets 4.0% on In-the-Wild, close to the published 1.9 to 2.9%. The gap comes from the 4 s crop and the small sample.
- **Model B fails outside its own training data.** It scores 32 to 38% EER on Hindi and In-the-Wild. Its 1.3% on commercial TTS is invalid, because it was trained on that set.
- **Model A's weak spot is modern commercial TTS** (10% EER). It gives many ElevenLabs-style fakes a fake-probability below 0.5.
- **The best threshold swings from 0.001 to 0.895 across domains.** A fixed 0.5 cut-off is wrong, so calibrate on our own domain.
- **The Hindi 0% is not trustworthy yet.** It has only 10 real speakers, and real and fake come from different sources, so the model may be telling the recording setups apart. Step 3 builds a same-speaker, same-channel test.

### Step 3: same-speaker clones, clean vs phone (9 Nov 2026)
Data: 20 Kathbath speakers each for Hindi and Telugu (gender-balanced). Real: 25 clips per speaker. Fake: 25 kNN-VC conversions per speaker, made from about 40 of that speaker's own reference clips. In total 500 real + 500 fake per language. Split by speaker: 14 train / 3 dev / 3 test. Phone = 300 to 3400 Hz band, 8 kHz, G.711 µ-law. Model A ONNX window is 64,000 samples (4.0 s).

| Model | Test set | Channel | EER % | AUC | Threshold at EER | Accuracy at 0.5 % |
|---|---|---|---|---|---|---|
| A (zero-shot) | Hindi same-speaker kNN-VC | clean | 0.1 | 1.000 | 0.877 | 99.8 |
| A (zero-shot) | Hindi same-speaker kNN-VC | phone | 1.5 | 0.997 | 0.081 | 97.7 |
| A (zero-shot) | Telugu same-speaker kNN-VC | clean | 0.0 | 1.000 | 0.342 | 99.9 |
| A (zero-shot) | Telugu same-speaker kNN-VC | phone | 5.2 | 0.991 | 0.048 | 90.8 |

Findings:
- Model A already catches same-speaker voice conversion in Hindi and Telugu. The Step 2 result was not just a dataset shortcut.
- **The phone channel is the main weakness**: Telugu EER goes from 0.0% to 5.2%. The best threshold collapses from 0.34–0.88 (clean) to 0.05–0.08 (phone), so under phone audio the model calls many fakes "real".
- Listening check: the kNN-VC fakes sound **cleaner** than the real Kathbath clips, which have some background noise. If we fine-tuned on this as-is, the model could learn "clean = fake". So both classes must get the same noise and channel augmentation, and some real clips should be denoised.

### Step 4: fine-tuning vs zero-shot (9 Nov 2026)
**Setup:**
- **Detector:** NII AntiDeepfake rebuilt in PyTorch. It matches the ONNX model (max logit difference 0.0001); 315 M parameters.
- **Training set:** 3,760 clips (1,884 real / 1,876 fake):
  - Step 3 training speakers: 700 / 700
  - extra Kathbath real: 600
  - IndicSynth fakes: 600
  - commercial TTS (Polly, Speechify, Luvvoice) plus real recordings: 584 / 576
- **Augmentation:** the same random effects on both classes (noise at SNR 5 to 30 dB, denoise, phone line, low-pass).
- **Optimizer:** AdamW, backbone LR 5e-6, head 5e-5, batch 16, CNN front-end frozen.
- **Model selection:** mean EER on 4 dev sets: held-out speakers clean, held-out speakers phone, unseen Hume voices, and an In-the-Wild dev sample. The zero-shot model counts as step 0, and training stops early after 4 evaluations with no improvement.

**Dev curve (mean EER):**

| Step | 0 (zero-shot) | 100 | 200 | 300 | 400 |
|---|---|---|---|---|---|
| Mean EER | **1.50%** | 4.60% | 3.52% | 3.89% | 2.65% |

Training stopped early at step 400, and the **zero-shot model was kept**. During training, In-the-Wild dev EER was 7.0 to 8.5% (zero-shot: at most 6%), and unseen Hume went from 1.7% to 0.0%.

**Test results:** none of these sets were trained on. "Calibrated" uses the threshold p_fake = 0.0168, taken from the pooled dev EER.

| Test set | EER % | AUC | Accuracy at calibrated threshold % | Accuracy at 0.5 % |
|---|---|---|---|---|
| Hindi/Telugu test speakers, clean | 0.0 | 1.000 | 99.7 | 99.7 |
| Hindi/Telugu test speakers, phone | 2.0 | 0.995 | 96.3 | 97.0 |
| Commercial, unseen ElevenLabs + Kokoro | 8.2 | 0.956 | 74.5 | 62.9 |
| Commercial, unseen ElevenLabs + Kokoro, phone | 15.0 | 0.919 | 76.8 | 60.5 |
| In-the-Wild (test sample) | 4.7 | 0.992 | 94.8 | 93.2 |

Latency is 29 ms per 4 s window on a T4 (PyTorch fp16).

Findings:
- **Fine-tuning on about 3.8k clips did not beat the zero-shot model on unseen data.** It improved the unseen commercial voice (Hume) but got worse on In-the-Wild. That is typical overfitting and forgetting when a small dataset meets a model pre-trained on 74k hours. Selecting the model on unseen dev sets caught this and kept the original. Our first two attempts had no such safeguard.
- **Decision:** the live system uses the zero-shot AntiDeepfake model, from the existing ONNX file.
- **Remaining weakness: modern commercial TTS, especially over a phone line** (8.2% EER clean, 15.0% phone). The gap between EER and accuracy at the calibrated threshold (8.2% EER but only 74.5% accuracy) shows that one threshold does not fit every source.

  We handle this at system level:
  - average scores over several windows per call;
  - speaker verification against an enrolled voice;
  - scam-language context;
  - a VERIFY band in the risk engine instead of a hard real/fake cut-off.
- **Optional, only if time allows:** a gentler fine-tune. Train only the top layers at LR 1e-6, and add more commercial TTS (for example MLAAD).

### Step 5: live detection server (9 Nov 2026)
Code is in [voiceshield/](../voiceshield/README.md).

**Pipeline:**
1. FastAPI WebSocket receives 16-bit PCM in ~100 ms chunks. Other sample rates go through a streaming resampler with an elliptic anti-alias filter.
2. Silero VAD v5 checks each 32 ms frame (300 ms hangover), and only speech frames continue.
3. A rolling 4 s speech window is analysed after every 1 s of new speech, once 2 s of speech are in.
4. The AntiDeepfake ONNX detector gives a log-odds score (llr) on CPU.
5. The risk engine turns it into a decision:
   - window score = sigmoid(llr − line cut-off);
   - running risk = EMA (α 0.4) of window scores;
   - **VERIFY** at running risk ≥ 45 or 2 suspicious windows;
   - **HOLD** at running risk ≥ 75 **and** ≥ 3 of the last 5 windows suspicious;
   - the call verdict never de-escalates.
6. Actions: a prevention message to the client, an optional webhook, and a SQLite audit log that stores scores only, never audio.

**Line profiles (calibration):** the detector's raw score depends on the channel. Clean real speech sits around llr −10, while the In-the-Wild EER cut-off is about −4.4. So each line gets a profile:
- `default`: −4.07, from the Step 4 pooled dev sets.
- `studio`: −5.63, the window-level EER on the dev split of the commercial-TTS dataset (116 real clips from 4 speakers, 116 Hume fakes; dev window EER 2.1%).
- `mic`: recorded in the console from 25 s of the user's voice. The rule is the genuine 95th percentile + 1.5.

**Speed:** 330 ms per 4 s window on the laptop CPU (i5-12500H, ONNX Runtime). A real-time 1 s hop needs no GPU and no quantization, so the planned "speed-up" step is not needed.

**Demo calls:** these are the Step 4 test split, never used for calibration. Profile `studio`.

| Call | Verdict | First alert | HOLD at | Max risk |
|---|---|---|---|---|
| Genuine caller 1 (yt_0009, 34 s) | PASS | none | none | 10 |
| Genuine caller 2 (yt_0010, 35 s) | PASS | none | none | 5 |
| ElevenLabs clone (16 s) | VERIFY | 3.2 s | none | 71 |
| Kokoro TTS clone (18 s) | HOLD | 14.1 s | 16.5 s | 82 |

With the `default` profile, the genuine calls still PASS and both clones reach VERIFY (at 4.6 s and 15.5 s).

**Tests:** 19 automated tests pass. They cover the resampler, VAD windowing, chunk-size invariance, risk escalation, calibration rules, the WebSocket flow (genuine, cloned, 44.1 kHz input, calibration, bad start) and the real-model demo calls. The browser console streamed the ElevenLabs demo end to end with the same result (VERIFY at 3.2 s).

Findings:
- **No false alarms on genuine callers, and every clone was caught.** The detector alone gives premium clones VERIFY (call-back/OTP), not always HOLD. The Kokoro clone looked genuine for its first 13 s.
- **A per-line cut-off is essential.** The same model needs about −5.6 on a clean line and about −4 on mixed or phone audio. The live microphone must be calibrated on the demo laptop before presenting.
- **Next (Step 6):** add speaker verification against an enrolled voice, and scam-language detection on the transcript. These are the signals that should push premium clones from VERIFY to HOLD.

### Step 5b: live microphone check (10 Nov 2026)

| Input | Genuine voice, median llr (range) | Result with `default` profile |
|---|---|---|
| Browser "system default" mic (4 sessions, device unknown; laptop had a Bluetooth headset and Voicemeeter virtual inputs active) | **+2.2** (−2.8 to +5.2) | false **HOLD** every time |
| Laptop *Microphone Array (Intel Smart Sound)*, chosen explicitly, all browser processing off | **−7.1** (−9.0 to −2.4) | VERIFY (two short stretches reached −2.4 to −4.6) |

The recording was clean: no clipping, sound up to 7.4 kHz, speech about 32 dB above the noise floor. The suspicious stretches did not differ in level or spectrum, so this is ordinary variation for this mic.

Scoring the saved recording offline gave exactly the live numbers, so the browser (Brave) does not alter the audio.

**Fix:** the `mic` profile is calibrated on this laptop mic, giving a cut-off of **−1.39** (genuine 95th percentile −2.89 + 1.5).

Findings:
- **The input device matters more than anything else.** A real voice through the wrong input (most likely the Bluetooth hands-free headset or a virtual mixer) looks *more synthetic than most clones*. The console now shows which mic is used, plus a level meter, and can save a recording for `tools/diagnose_audio.py`. Demo rule: use the laptop or a wired mic, never a Bluetooth headset mic.
- **The trade-off:** a cut-off high enough to keep this mic's genuine speech at PASS (−1.39) would miss the premium demo clones at their studio-line scores (ElevenLabs −2 to −7.7, Kokoro −0.4 to −10).
  - The detector alone cannot separate *premium* clones from genuine laptop-mic speech.
  - Open-source clones (kNN-VC, XTTS/VITS/FreeVC in IndicSynth) scored far higher in Steps 2–3 (≈0% EER) and stay well above this cut-off.
  - This is why the final decision must fuse several signals (Step 6) and why the attack test (Step 9) must replay clones through the real call channel.

### Step 6: voiceprint + transcript + scam language, fused decision (10 Nov 2026)

**B. Voiceprint:** WeSpeaker ResNet34-LM (VoxCeleb, 26 MB ONNX).
- Kaldi-style 80-band filterbank computed in numpy, so PyTorch is not needed.
- 104 ms per 4 s window on CPU.
- Check on 7 real recordings: the recordings turned out to be two-person interviews, and the similarity matrices showed two clear blocks per recording. Within one person the cosine similarity was 0.6–0.87; between different people it was 0.0–0.25.
- Cut-off 0.45. Mismatch probability = sigmoid((0.45 − cos)/0.08) on the call's running voiceprint.

**C. Speech-to-text:** faster-whisper on utterances cut by the VAD (pauses, maximum 8 s).

| Setup | Time for 8 s of speech |
|---|---|
| Whisper small, RTX 3050 GPU (fp16, via pip `nvidia-cublas-cu12` + `nvidia-cudnn-cu12`) | 0.2–0.3 s |
| Whisper small, CPU int8, 8 threads | 3.3 s |
| Whisper base, CPU int8, 8 threads (the CPU fallback) | 1.0 s |

Whisper's known phantom phrases on silence are filtered out, and the model loads offline once downloaded.

**Scam language:** explainable rules, 9 tactics:
- asks for OTP / PIN;
- remote-access app;
- send money;
- police / arrest threat;
- account / KYC block;
- family emergency;
- lottery;
- urgency / secrecy;
- claims to be a bank.

The rules cover English, romanised Hindi/Telugu, Devanagari and Telugu script. Tactics combine by noisy-OR: "transfer money" alone = 40, "bank + account blocked + share OTP" = 84.

**Fusion:**
- **HOLD** if the synthetic voice is strong, **or** if scam language ≥ 50 **and** the voice was suspicious (detector or voiceprint mismatch) at any point in the call.
- **VERIFY** for any single suspicious signal.
- Scam words from a genuine voice give VERIFY, never HOLD.

**Results** (full pipeline, offline, same code as live; `studio` profile for the demo files, `mic` for the laptop recording):

| Call | Verdict | Synthetic voice V | Scam language C | Note |
|---|---|---|---|---|
| Genuine caller 1 | PASS | 2.6 | 0 | |
| Genuine caller 2 | PASS | 0.7 | 40 | the word "drugs" in a life story; below 50 |
| Laptop mic, team member's voice | PASS | 6.4 | 0 | |
| ElevenLabs clone (no scam script) | VERIFY | 40.6 | 0 | |
| Kokoro clone (no scam script) | HOLD | 81.5 | 0 | |
| **Scam call, Windows TTS reading a bank-fraud script** | **HOLD at 4.4 s** | 100 | 84 | tactics: claims SBI, account blocked in 30 min, share OTP, don't tell anyone |

**Live browser test:** the scam call played through the console reached HOLD at 4.4 s. The transcript appeared utterance by utterance, with highlighted phrases and tactic labels, and Whisper ran on the GPU.

**Tests:** 38 automated tests pass:
- 32 fast stub tests: fusion rules, multilingual scam tactics, voiceprint store and mismatch, utterance cutting, enroll / claimed identity / transcript WebSocket flows;
- 6 real-model tests: voiceprints separate two real speakers, and the scam call reaches HOLD with "asks for OTP".

**Privacy:** the audit log keeps scores, decisions and *names* of scam tactics only. Transcripts, audio and voiceprints are never logged; voiceprints live only in `data/voiceprints.json` and can be deleted from the console.

### Step 6b: Windows audio enhancements cause false alarms (10 Nov 2026)
On 10 Nov afternoon the same laptop mic and the same person gave false HOLDs, even though the voiceprint check correctly matched the person (similarity 0.81). The cause was **Windows audio enhancements** on the Intel Smart Sound mic array: about 180 DSP settings for noise suppression, beam-forming and auto-gain, applied before the browser sees the audio. The browser's own processing was already off.

| Recording (same person, same mic) | Loudness (rms) | SNR | Genuine voice, median llr | Result (`mic` cut-off) |
|---|---|---|---|---|
| Enhancements ON, 9 Nov night | −30 dBFS | 32 dB | −7.1 | VERIFY |
| Enhancements ON, 10 Nov afternoon | −35 dBFS | 27 dB | **−2.3** | false **HOLD** |
| **Enhancements OFF**, 10 Nov afternoon | −53 dBFS | 19 dB | **−10.0** | **PASS** |

- **The processing, not the noise, made a real voice look synthetic.** The raw signal was quieter and noisier, yet it looked completely genuine (−10, the same as clean studio speech), even after the phone-line simulation (−9.7).
- **The enhancements also make scores unstable from session to session** (−7 one night, −2 the next afternoon), so no fixed cut-off can work with them on.

**Fix:**
- Enhancements turned off.
- `mic` profile recalibrated on raw audio: cut-off **−7.11** (genuine 95th percentile −8.61 + 1.5).
- Voiceprint re-enrolled from raw audio with `tools/enroll_file.py`.

**Effect** (demo files are on a studio line, so these clone results are indicative only):

| Call | Old `mic` cut-off −1.39 | New `mic` cut-off −7.11 |
|---|---|---|
| ElevenLabs clone | missed | **HOLD at 4.6 s** |
| Kokoro clone | missed | **HOLD at 15.5 s** |
| Genuine callers 1–2 | PASS | **PASS** (max risk 30, 16) |
| User's raw voice | not recorded | **PASS** (max risk 38) |

**Demo rules:**
- enhancements OFF;
- no Bluetooth mic;
- calibrate on the demo laptop.

Margin note: the calibration used one 20 s recording, and the genuine maximum (−7.6) sits only 0.5 below the cut-off. More genuine recordings should be added before the final demo.

### Step 6c: live tests: self, ChatGPT voice, two fixes (10 Nov 2026)

| Live call (laptop mic, enhancements OFF, caller claims to be the enrolled user) | Median llr | Voiceprint similarity | Result |
|---|---|---|---|
| Enrolled user speaking | −9.25 | **0.82** | PASS (one brief VERIFY from the first window, fixed below) |
| **ChatGPT voice** (premium TTS) played from a speaker into the mic | **−7.0** | **0.0–0.1** | **HOLD at 6.6 s**: "voice does not match the claimed person (100/100)" + synthetic voice 82/100 |

The line profile had silently reset to `default` (−4.07) after a page reload. With the correct `mic` cut-off, 12 of the ChatGPT call's 38 windows are flagged, and replaying its scores gives **HOLD by the 3rd window from the voice detector alone**.

**Two fixes from these tests:**
1. **The running synthetic-voice risk now starts at 0 (genuine) instead of at the first window's score.** One odd first window (−4.2 for the genuine user) had caused a VERIFY, which contradicts the "one odd window cannot swing the call" rule. Clones still escalate within 3 windows (unit test from the real ChatGPT call: PASS → VERIFY → HOLD).
2. **The console remembers the selected line profile** across reloads.

**`mic` profile recalibrated on 2 genuine sessions (40 windows):** cut-off **−6.01**, from genuine 95th percentile −7.51 + 1.5.

| Call | Result, `mic` −6.01 |
|---|---|
| User, sessions 1 and 2 | PASS (max risk 7, 35) |
| Genuine callers 1–2 | PASS (max 14, 6) |
| ChatGPT voice through a speaker (replayed scores) | **HOLD at window 3** |
| Robot-voice scam call | **HOLD at 4.4 s** |
| Kokoro clone | **HOLD at 16.5 s** |
| ElevenLabs clone | VERIFY at 3.2 s (fusion makes it HOLD when a scam script is used) |

Findings:
- **A premium TTS replayed acoustically** (speaker → room → mic) is the hardest case for the detector. Its median llr (−7.0) is only about 2.5 above genuine speech on the same mic (−9.3 to −10), so voice-only detection is possible but needs a mic-specific cut-off.
- **When the cloned voice is not the claimed person, the voiceprint check is decisive** (similarity 0.0–0.1 against 0.82).
- **The hardest real attack** remains a clone *of the enrolled person* (the voiceprint matches) played acoustically. This is the Step 9 attack test.
- **Choosing the cut-off is a trade-off.** −7.11 (one session) caught the ElevenLabs demo as HOLD but would have raised false alarms on the user's second session. −6.01 (two sessions) keeps genuine calls at PASS and leaves premium clones without scam words at VERIFY.

### Step 6d: controlled live test: same script, human vs ChatGPT voice (10 Nov 2026)
Setup:
- The same text was read by the user and by ChatGPT's voice mode, played from a speaker into the laptop mic.
- Laptop mic, enhancements OFF, `mic` profile (−6.01), the updated risk engine.
- No claimed identity, so only the synthetic-voice detector and the scam rules were active.

| Speaker | Median llr | Windows above cut-off | Scam language | Result |
|---|---|---|---|---|
| **ChatGPT voice** (premium TTS, acoustic replay) | **−4.56** | **31 / 45** | 20 ("claims to be a bank") | VERIFY at 4.3 s, **HOLD at 5.3 s** |
| **Human (user)** | **−10.26** | **0 / 19** | 20 ("claims to be a bank") | **PASS** (max risk 24) |

**Conclusion:** with identical words, the decision came from the voice alone.
- The detector separated a premium AI voice played through a speaker from the real speaker by **5.7 log-odds** (median −4.6 vs −10.3), and reached HOLD in 5.3 s.
- The identical scam content gave both the same small score, well below the alert level.
- This is the strongest live evidence so far. The remaining hard case is a clone *of the enrolled person* (Step 9).

### Step 7: two-party call demo with prevention (10 Nov 2026)

**Built:**

| Screen | Purpose |
|---|---|
| `/caller` | the genuine caller or the attacker. Raw mic, or **injects a recording into the call** (Web Audio → WebRTC track), and can send a payment request |
| `/customer` | answers the call, plays the caller, and streams the **remote** audio to VoiceShield (`/ws/stream`). Shows the live verdict, the three signals, the timeline and the transcript, plus a mock bank app |
| `/security` | live incidents: verdict, reasons, scam tactics, prevention actions, KPIs |

**Call path:**
- WebRTC peer connection; the server relays only signalling (offer / answer / ICE / app messages) through `/ws/signal/<room>`.
- No STUN/TURN is needed on one laptop or one Wi-Fi.
- Opus forced to mono 128 kbit/s with no DTX, so the codec does not add "synthetic" artefacts.

**Prevention flow (bank app):**
- **PASS:** payment allowed.
- **VERIFY:** payment requires an out-of-band check: an OTP sent to the registered phone (with a "never share on a call" warning), or hang up and call back on the official number.
- **HOLD:** a pending or new payment is **blocked automatically** and the security team is alerted.
- Every step is recorded against the call (`TRANSFER_REQUESTED/APPROVED/BLOCKED/DECLINED`, `OTP_SENT/VERIFIED/FAILED`, `CALLBACK_REQUESTED`) and appears on `/security`.

**End-to-end test** (two browser tabs; the robot-voice scam recording injected by the caller; a ₹49,999 transfer request):

| Event | Call time |
|---|---|
| Customer answers | 0 s |
| Injected voice starts speaking | 14.9 s |
| VERIFY | 18.1 s |
| **HOLD** | **19.2 s (≈4 s of speech)** |
| Transfer request (sent during the call) | **blocked automatically**, "security team alerted (incident #19)" |

The transcript received through the call was accurate, with all four scam tactics labelled. `/security` showed the incident with its reasons, the tactics and `TRANSFER_BLOCKED`.

**Tests:** 37 fast tests (+6 real-model tests). New tests cover the pages, the signalling relay (join, peer-joined, offer/answer/app relay, peer-left, junk dropped, bad role refused) and the prevention-event API (valid / invalid kind / unknown call → incidents).

**UI redesign (10 Nov 2026):**
- One design system (`static/ui.css` + `static/shell.js`) for all four screens: shared top bar with navigation, live model status (detector device, Whisper device, number of voiceprints), and a dark/light theme switch.
- Verdict panel with a risk gauge, three signal meters, an alert banner, a chat-style transcript, a phone-style call card, a banking app in a phone frame (with the OTP as an SMS notification), and a security dashboard with KPI tiles and risk bars.
- Risk chart:
  - colours validated for colour-blind safety in both themes (blue risk line + magenta window dots; status amber/red for the thresholds, with text labels);
  - hover tooltip shows time, risk, window score, llr and decision.
- No external fonts or libraries, so it works offline at the venue. Checked in the browser in both themes, at desktop and narrower widths (no horizontal scroll); 37 fast tests pass.

Open items:
- Detector latency rose from 0.33 s to 1–8 s per window when the laptop was busy with other apps (IDE, language servers). Close heavy apps for the demo, or move the detector to the RTX 3050 (ONNX Runtime CUDA).
- Genuine-caller and impersonator scenarios through the call need a live microphone; the browser pane used for testing blocks mics. The WebRTC channel (Opus) should get its own `call` line profile, which is calibrated from the customer page.
- Phone-as-caller over Wi-Fi needs HTTPS for the microphone.

Limitations, carried to Step 9:
- The key fusion case, a *premium* clone + scam script ending in HOLD, is covered by unit tests but not yet by real premium-clone audio.
- A good clone of an enrolled person can pass the voiceprint check; the voiceprint targets human impersonators.
- The scam rules are keyword based (for example "pin" can fire on unrelated speech), and Whisper small is weaker on Telugu than on English and Hindi.

### Step 8: consumer audio processing vs the detector; DSP-aware LoRA, attempt 1 (17 Nov 2026)

**Research question:** does everyday audio processing (OS noise suppression, Bluetooth and call codecs, auto-gain) break an SSL deepfake detector? Can parameter-efficient fine-tuning fix it?

Notebook: `notebooks/step8_dsp_lora.ipynb`; builder and local test in `notebooks/build/`.

**Effects implemented:**
- noise suppression (non-stationary spectral gating);
- spectral subtraction (over-subtraction, gain floor, temporal smoothing);
- auto-gain control (fast attack, slow release, soft limiter);
- G.711 phone line;
- Bluetooth HFP CVSD (8 kHz, delta modulation at 64 kbit/s);
- Bluetooth mSBC-like (16 kHz, 7 kHz band, 8-bit companding);
- Opus at 8–24 kbit/s (ffmpeg).

In the test sets, each effect is applied after 15 dB background noise.

**Measurement: the original detector on unseen test sets.** "False alarms" and "missed fakes" are measured at the dev-calibrated threshold.

| Test audio | EER % | AUC | False alarms on real % | Missed fakes % |
|---|---|---|---|---|
| Test speakers (Hindi/Telugu kNN-VC), clean | 0.0 | 1.000 | 0.7 | 0.0 |
| … phone line (G.711) | 2.0 | 0.995 | 4.7 | 2.7 |
| … noise suppression | 5.0 | 0.988 | 4.7 | 6.0 |
| … spectral subtraction | 1.3 | 0.992 | 1.3 | 1.3 |
| … auto-gain + noise suppression | 2.0 | 0.998 | 3.3 | 0.7 |
| … Bluetooth CVSD | **10.0** | 0.953 | 1.3 | **26.7** |
| … Bluetooth mSBC-like | 2.0 | 0.996 | 2.7 | 2.7 |
| … Opus | 4.7 | 0.989 | **32.7** | 0.0 |
| ElevenLabs + Kokoro, clean | 8.2 | 0.956 | 0.4 | 51.0 |
| … noise suppression | **34.0** | 0.699 | 0.9 | **72.6** |
| … Opus | **45.1** | 0.589 | 32.6 | 58.1 |
| In-the-Wild, clean | 4.7 | 0.992 | 4.7 | 5.7 |
| … noise suppression | 7.5 | 0.974 | 5.0 | 12.3 |

Findings:
- **Everyday processing "launders" premium clones.** Noise suppression raised the EER on ElevenLabs/Kokoro from 8.2% to 34.0%, and Opus raised it to 45.1%, close to chance. A scammer gets this for free by calling over a VoIP app.
- **Opus shifts scores for real speech too:** 32.7% of real voices are flagged at the standard cut-off, even though the ranking is still good (EER 4.7%). Per-line calibration (Step 5) fixes this kind of shift; it cannot fix the information loss seen with Bluetooth CVSD or Opus on premium clones.
- **Bluetooth CVSD hurts even on Indian-language clones:** EER 0% → 10%, with 26.7% of fakes missed.

**Attempt 1: LoRA + symmetric DSP augmentation.**
- Setup:
  - LoRA r = 8, α = 16 on q/k/v/out and both feed-forward layers of all 24 blocks: **3.54 M trainable parameters of 319 M (1.11%)**, backbone frozen;
  - 4,360 training clips, augmentation identical for real and fake;
  - learning rate 1e-4 (LoRA) / 1e-5 (head);
  - the original model is step 0.
- Dev history (mean EER of 8 dev sets):

| Step | Mean | Clean | Phone | Hume | In-the-Wild | Noise suppr. | AGC + NS | BT CVSD | Opus |
|---|---|---|---|---|---|---|---|---|---|
| 0 (original) | **4.17** | 0.0 | 2.0 | 0.0 | **4.0** | 4.7 | 2.0 | 14.0 | 6.7 |
| 100 | 7.49 | 1.3 | 11.0 | 1.7 | 5.5 | 9.3 | 5.0 | 20.0 | 6.0 |
| 200 | 6.36 | 1.3 | 7.0 | 2.2 | 6.8 | 8.0 | 5.3 | 14.0 | 6.3 |
| 300 | 5.36 | 0.3 | 4.7 | 1.7 | 7.5 | 7.0 | 5.3 | 12.3 | **4.0** |
| 400 | 5.28 | 1.0 | 5.3 | 1.7 | **9.5** | 5.3 | 4.0 | **10.7** | 4.7 |

- Early stopping kept the original model, so the test results are unchanged.
- **Diagnosis:** the adapters did learn the processed conditions (Bluetooth CVSD dev 14.0 → 10.7, Opus 6.7 → 4.7) but **forgot real-world fakes** (In-the-Wild dev 4.0 → 9.5). Label-only training on 4.4k clips pulls the model away from what it learned on 74k hours.

**Next: attempt 2 (`step8b_consistency_lora.ipynb`), teacher-anchored consistency.**
- The frozen original model (LoRA off, original head) scores the *clean* crop. The LoRA model scores a *processed* copy of the same crop.
- Loss = cross-entropy(label) + λ·τ²·KL(teacher_clean ‖ student_processed), with λ = 1 and τ = 2. The KL term is used only when the teacher is right on the clean crop.
- Learning rate 5e-5; patience 6.
- Aim: keep the original's behaviour (In-the-Wild, clean) while learning to give the clean-audio answer after processing.

### Step 8b: teacher-anchored consistency LoRA, attempt 2 (18 Nov 2026)

**Setup:** the same data, development and test sets, LoRA configuration and model selection as attempt 1.
- **Loss:** cross-entropy(label) + λ·τ²·g·KL(teacher_clean ‖ student_processed), with λ = 1 and τ = 2. The gate g = 1 only when the teacher is right on the clean crop.
- **Teacher:** the frozen original model (adapters off, frozen copy of the original head).
- **Data:** 85% of clips are processed copies of the same crop.
- **Training:** LR 5e-5 (LoRA) / 1e-5 (head), 100 warm-up steps, patience 6.

**Training log:**
- CE fell from 1.61 to 0.20–0.30.
- KD fell from 3.72 to 0.48–0.84.
- The teacher was right on 92–96% of clean crops, so the anchor was active.
- The run stopped early at step 600 (16 min).

**Dev history (mean EER of 8 dev sets):**

| Step | Mean | Clean | Phone | Hume | In-the-Wild | Noise suppr. | AGC + NS | BT CVSD | Opus |
|---|---|---|---|---|---|---|---|---|---|
| 0 (original) | **4.17** | 0.0 | 2.0 | 0.0 | **4.0** | 4.67 | 2.0 | 14.0 | 6.67 |
| 100 | 6.67 | 1.0 | 6.33 | 0.43 | 4.25 | 11.0 | 4.33 | 20.0 | 6.0 |
| 200 | 7.42 | 1.0 | 9.67 | 2.16 | 6.5 | 10.67 | 5.33 | 18.33 | 5.67 |
| 300 | 6.17 | 0.33 | 8.0 | 1.72 | 8.0 | 8.0 | 5.0 | 14.0 | **4.33** |
| 400 | 5.79 | 0.33 | 5.0 | 1.72 | 8.25 | 6.67 | 5.33 | 13.0 | 6.0 |
| 500 | 6.43 | 0.33 | 6.0 | 2.59 | **9.5** | 8.0 | 5.33 | 13.67 | 6.0 |
| 600 | 6.07 | 1.0 | 7.0 | 1.29 | 9.25 | 6.67 | 4.33 | **11.33** | 7.67 |

**Result:** best = step 0 (the original), so the test results are identical to Step 8 and nothing was exported. The history is saved in `recordings/step8b_history.csv`.

Findings:
- **The anchor did not stop the forgetting.** Attempt 2 behaved almost exactly like attempt 1:
  - CVSD 14.0 → 11.3, Opus 6.7 → 4.3;
  - but In-the-Wild 4.0 → 9.5;
  - best mean 5.79% (attempt 1: 5.28%).
- **Diagnosis:**
  1. The consistency term only constrains the model on the *training* distribution (Indian-language speech, commercial TTS). Nothing holds its behaviour on In-the-Wild-like audio, which drifted the same way as before.
  2. The label loss is computed at p_fake = 0.5, while the model's calibrated threshold is 0.0185. The high starting CE (1.6) suggests that much of the training effort goes into moving the decision boundary towards the training domain.
- **Conclusion after 3 fine-tuning runs (Step 4 full, Step 8 LoRA, Step 8b LoRA + consistency):** with ≈4k clips against 74k hours of post-training, the detector does not get better. Robustness to consumer processing is handled at system level: a controlled input chain, per-line calibration and several signals.
- **Possible attempt 3 (only if time allows):**
  - compute the CE and the gate at the calibrated threshold instead of 0.5;
  - distil the original model on a large, diverse pool of *unlabelled* speech from other sources (Learning-without-Forgetting style), so behaviour outside the training domain is held in place.

### Step 8c: frozen detector, layer analysis, layer back-end and automatic line profile (20 Nov 2026)

**Idea:** never change the detector's weights, so it cannot forget.
- Run the frozen detector once per clip and keep the time-mean of all 25 hidden layers:
  - layer 0 is the input to the first transformer block;
  - layer 24 is the final output, which the original head reads.
- Train only small models on these cached features.

Notebook: `notebooks/step8c_frozen_backend.ipynb`; builder and CPU test in `notebooks/build/`.

**Setup:**
- **Training views:** the Step 8 training clips (4,360), each in 3 views:
  - 1 clean view;
  - 2 processed views: background noise at 5–30 dB with p = 0.5, then one of 8 processing types with p = 0.85 (phone, NS, SS, AGC, AGC + NS, CVSD, mSBC, Opus).
  - In total 13,080 views.
- **Inputs:** each layer standardised with the training mean and standard deviation.
- **Development and test sets:** the same 8 development sets and the same test sets as Step 8, with 5 extra test sets.
  - Extra held-out sets (spectral subtraction, auto-gain, mSBC) are used only to calibrate thresholds.
- **Three parts:**
  1. **Layer analysis:** one linear probe per layer (AdamW, lr 1e-2, 300 full-batch steps).
  2. **Back-end:** a softmax weight per layer, then dropout 0.2 and a linear layer.
     - 1,050 parameters.
     - AdamW lr 1e-3, batch 256, 30 epochs; the epoch is chosen on the dev mean EER.
     - **Fusion:** (1 − w)·z(original llr) + w·z(back-end).
     - w is picked from 0, 0.05, …, 1 on the dev mean EER; w = 0 is exactly the original model.
  3. **Automatic line profile:** the same structure with 9 outputs predicts the processing type (9,250 parameters, 20 epochs, chosen on held-out accuracy).
     - Each processing type has its own EER threshold, calibrated on held-out speakers.
     - Clean is calibrated on held-out clean + Hume + In-the-Wild dev.
     - The predicted type picks the threshold.

**Run:** Kaggle T4. Results are saved in `recordings/step8c_results_from_log.csv` and `recordings/step8c_layers_from_log.csv`, copied from the notebook output.

**1. Layer analysis:** linear-probe EER % per layer (selected rows; test mean = 13 test sets).

| Layer | Clean | In-the-Wild dev | NS | BT CVSD | Opus | Dev mean | Test mean |
|---|---|---|---|---|---|---|---|
| 0 (input to block 1) | 4.67 | 39.75 | 14.00 | 25.33 | 15.33 | 16.07 | 16.12 |
| 4 | 0.00 | 10.50 | 2.33 | 5.33 | 3.67 | 3.21 | 5.13 |
| 9 | 0.00 | 3.50 | 0.67 | 3.33 | 2.67 | **1.66** | 4.47 |
| 11 | 0.00 | 3.00 | 0.33 | 4.67 | 4.33 | 1.88 | **4.18** |
| 16 | 0.00 | 4.00 | 1.33 | 5.33 | 4.00 | 2.13 | 4.34 |
| 20 | 0.00 | 4.00 | 0.67 | 4.67 | 6.00 | 2.50 | 5.63 |
| 23 | 0.00 | 3.25 | 1.67 | **11.33** | 4.67 | 3.70 | 7.87 |
| 24 (last, read by the head) | 0.00 | 2.00 | 0.67 | 5.67 | 6.67 | 2.62 | 6.55 |
| Original head (layer 24) | 0.00 | 4.00 | 4.67 | 14.00 | 6.67 | 4.17 | 9.73 |

- **Middle blocks (5–11) keep the most evidence under processing:** dev mean 1.7–2.0%, test mean 4.2–4.5%.
- The blocks just before the last (21–23) lose it under Bluetooth CVSD (9–11%).
- Even the **last layer** gives 6.55% test mean with a probe trained on processed views, against 9.73% for the original head.
  - So much of the damage is in the **readout (head)**, not in the representation.
- The back-end's learned layer weights agree:
  - highest at blocks 9–11 (0.053–0.054; uniform would be 0.040);
  - lowest at blocks 21–23 and layer 0 (0.025–0.029);
  - blocks 7–19 hold 63% of the weight.

**2. Back-end and fusion.**
- **Dev:** best epoch 14 (dev mean 1.51% vs 4.17%).
- **Fusion:** dev mean EER vs w, 4.17 (w = 0) → 1.98 (0.4) → 1.43 (0.8) → 1.51 (1.0); **w = 0.85 chosen**, fused dev mean **1.39%**.
- **Unseen test sets, EER %:**

| Test set | Original | Back-end | Fused (w = 0.85) |
|---|---|---|---|
| Test speakers, clean | 0.00 | 0.00 | 0.00 |
| … phone | 2.00 | 0.00 | 0.00 |
| … noise suppression | 5.00 | 1.33 | 1.33 |
| … spectral subtraction | 1.33 | 0.33 | 0.33 |
| … auto-gain + NS | 2.00 | 1.00 | 1.33 |
| … Bluetooth CVSD | 10.00 | 3.33 | 4.00 |
| … Bluetooth mSBC | 2.00 | 0.00 | 0.00 |
| … Opus | 4.67 | 0.00 | 1.33 |
| ElevenLabs + Kokoro, clean | 8.23 | 3.38 | **2.95** |
| … noise suppression | 33.97 (AUC 0.699) | 11.19 (0.950) | **12.66 (0.946)** |
| … Opus | 45.15 (AUC 0.589) | 14.14 (0.919) | **16.67 (0.890)** |
| In-the-Wild (test sample) | 4.67 | 4.67 | 4.67 |
| In-the-Wild, noise suppression | 7.50 | 7.67 | 7.33 |
| **Mean of 13 sets** | **9.73** | **3.62** | **4.05** |
| Mean of 10 processed sets | 11.36 | 3.90 | 4.50 |

**3. Automatic line profile** (original model).
- **Thresholds (llr):**
  - single (pooled dev EER) −3.99;
  - per processing: clean −4.28, phone −3.26, NS −4.05, SS −4.76, AGC −5.80, AGC + NS −4.76, CVSD −6.95, mSBC −3.10, **Opus +0.20**.
  - Opus shifts the scores by about 4 log-odds.
- **Processing classifier:**
  - 86.6% mean accuracy on held-out speakers;
  - 80.7–98.0% on test speakers, 92.8–98.5% on processed premium clones;
  - only 59.5% on clean premium clones and 29.7% on In-the-Wild "clean", which it mostly calls Opus. In-the-Wild clips are internet audio that was already compressed, so this may be correct.
- **False alarms on real / missed fakes, %:**

| Set | Original, single | Original, automatic | Fused, single | Fused, automatic |
|---|---|---|---|---|
| Test speakers, Opus | **32.7** / 0.0 | 13.3 / 2.0 | 7.3 / 0.0 | **2.7** / 0.0 |
| Test speakers, BT CVSD | 1.3 / **26.0** | 8.7 / 12.0 | 3.3 / 4.7 | 4.0 / 5.3 |
| Test speakers, AGC + NS | 3.3 / 0.7 | 10.0 / 0.7 | 2.0 / 0.0 | 1.3 / 0.7 |
| ElevenLabs + Kokoro, Opus | 32.6 / 57.7 | 12.9 / 78.4 | 12.4 / 22.4 | 7.3 / 32.8 |
| In-the-Wild | 4.7 / 5.7 | 3.3 / 9.7 | 0.7 / 8.0 | 0.7 / 9.7 |
| **Mean of 10 processed sets** | 8.9 / 18.2 | 6.8 / 19.0 | 3.8 / 5.7 | 3.2 / 6.7 |

- With the true processing label ("oracle") the results are almost the same as automatic, so the classifier is not the limit.
- Half total error on the processed sets:
  - original 13.6% → 12.9%;
  - fused 4.7% → 5.0%.

**Findings:**
- **The first method that beats the original model, and it does not forget.**
  - Mean EER over 13 unseen test sets fell from 9.73% to 4.05% (fused; 3.62% back-end alone).
  - In-the-Wild stayed at 4.67%. Every fine-tuning run (Steps 4, 8, 8b) had lost here (dev 4.0 → 9.5).
  - Unseen premium clones improved even without processing (8.2 → 3.0%), so the gain is not only "learned our DSP".
- **Consumer processing mostly breaks the readout, not the representation.**
  - Under noise suppression and Opus, the original head's AUC on premium clones was 0.70 and 0.59, near chance. The frozen features read by the back-end still give 0.95 and 0.92.
  - The earlier reading "information is lost" (Step 8) was only partly right. Some loss remains: 11–17% EER.
- **The automatic line profile corrects shifts but does not reduce total error.**
  - It cut Opus false alarms on real voices from 32.7% to 13.3% (original) and from 7.3% to 2.7% (fused; 4 of 150, 95% CI 1.0–6.7%).
  - But misses went up elsewhere (ElevenLabs Opus 57.7 → 78.4%).
  - With the fused score a single threshold is already good. The profile is optional, useful where Opus false alarms matter.
- **Why this worked where LoRA did not:**
  - The backbone is frozen, so behaviour on unseen data cannot drift.
  - 1,050 trainable parameters instead of 3.54 M.
  - The back-end only re-weights evidence the detector already computes, mostly from the middle layers.

**Caveats:**
- The processed test sets use the same DSP simulation code as the training views (different speakers and random draws). Real noise suppression, real Bluetooth and real VoIP apps are not tested yet. **Next check:** the laptop recordings with Windows enhancements ON/OFF.
- The epoch and w were chosen on the dev sets; the dev numbers are therefore optimistic, and only the test numbers count.
- The test sets are small: 150 real / 150 fake for test speakers; about 233 / 241 for ElevenLabs + Kokoro.

**Status:** integrated into the live server on 20 Nov (Step 8d below).

### Step 8d: Step 8c in the live server, tested on real recordings (20 Nov 2026)

**Integration** (`voiceshield/server/models.py`, `calibration.py`, `app.py`):
- `config.json` `"detector_mode": "step8c"` loads `voiceshield_layers.onnx` and runs the back-end in NumPy (`LayerBackend`).
  - `"original"` switches back to the old detector.
  - If the files are missing, the server logs a warning and falls back to the original.
- The fused score is reported on the original log-odds scale (mo + so·f), so w = 0 gives back the original score exactly, and the calibration rule (95th percentile + 1.5, clipped to [−9.5, 0]) still makes sense.
- Each score type has its own line profiles: `data/profiles.json` (original) and `data/profiles_step8c.json` (Step 8c).
  - Step 8c `default` = pooled dev EER of the fused score, **−1.91**.
  - New `mic` = **−1.64**, recalibrated on the same two enhancements-OFF sessions as the original `mic` (40 windows, genuine median −5.21).
  - New `studio` = **0.00**, recalibrated the same way as the original (window EER on the dev split of the commercial set: 116 real clips, 116 Hume fakes).
    - Dev window EER is **0.37%** (original detector: 2.13%).
    - The raw EER point was above 0, so it was capped at 0 by the existing safety limit.
  - Built-in `auto` profile: per-processing cut-offs chosen per window.
- The console shows "Detector + Step 8c" and a new *Audio type* column. Score messages also carry `llr_original`, `processing` and `window_threshold`.
- **Tests:**
  - 41 fast tests pass (4 new: back-end formula against the notebook's, per-score profiles, per-window cut-off, WebSocket with the `auto` profile).
  - 9 real-model tests pass (3 new: the exported model reproduces the original detector; demo calls give PASS for genuine and VERIFY/HOLD for clones). The Step 8c tests load one 1.26 GB model at a time, because the laptop has little free memory.

**Checks on real audio** (`tools/compare_step8c.py`, window level; `recordings/step8c_live_compare.csv`):
- `voiceshield_layers.onnx` gives the same detector score as the original ONNX: largest difference **0.0003** over 190 windows.
- CPU time: 727 → 762 ms per window (+5%) while the laptop was busy; 336–345 ms in the live console.

| Recording | Original: median llr, windows over −4.07 | Step 8c: median, windows over −1.91 | Audio type seen |
|---|---|---|---|
| Genuine caller 1 / 2 (demo) | −9.75 / −10.52, 0/31, 0/30 | −9.14 / −8.17, 0/31, 0/30 | clean |
| ElevenLabs clone (demo) | −5.81, **4/13** | **+0.92, 13/13** | clean |
| Kokoro clone (demo) | −6.55, 4/15 | −0.99, 10/15 | clean |
| Scam call, Windows TTS | +10.70, 18/18 | +11.69, 18/18 | Opus / mSBC |
| Laptop mic, enhancements OFF (×2) | −9.99 / −9.25, 0/40 | −5.85 / −4.34, 1/40 | clean |
| Laptop mic, enhancements ON, session 1 | −7.12, 4/23 | −3.32, 4/23 | **noise suppression** (20/23) |
| Laptop mic, enhancements ON, session 2 | −2.27, 17/20 | −0.37, 18/20 | **noise suppression** (18/20) |

**Full risk engine verdicts, demo calls.** Each detector is shown with its own profile of the same kind (max risk in brackets):

| Demo call | Original `default` (−4.07) | Step 8c `default` (−1.91) | Original `studio` (−5.63) | Step 8c `studio` (0.00) | Step 8c `auto` |
|---|---|---|---|---|---|
| Genuine caller 1 | PASS (2.5) | PASS (0.7) | PASS (10.1) | PASS (0.1) | PASS (1.5) |
| Genuine caller 2 | PASS (1.0) | PASS (1.2) | PASS (4.5) | PASS (0.2) | PASS (2.7) |
| ElevenLabs clone | VERIFY (51.7) | **HOLD at 5.6 s** (94.9) | VERIFY (68.8) | **HOLD at 12.3 s** (77.5) | HOLD at 5.6 s |
| Kokoro clone | VERIFY (68.5) | **HOLD at 13.1 s** (92.8) | HOLD at 16.5 s (81.5) | VERIFY (71.5) | HOLD at 11.0 s |
| Scam call, Windows TTS | HOLD at 4.4 s | HOLD at 4.4 s | HOLD at 4.4 s | HOLD at 4.4 s | HOLD at 4.4 s |

**Laptop recordings, `mic` profiles** (original −6.01, Step 8c −1.64):

| Recording | Original | Step 8c | Step 8c `auto` |
|---|---|---|---|
| Enhancements OFF, sessions 1 / 2* | PASS (6.9 / 34.6) | PASS (6.1 / 24.9) | PASS (15.4 / **44.6**) |
| Enhancements ON, session 1 | **false HOLD** | VERIFY (45.2) | VERIFY (54.5) |
| Enhancements ON, session 2 | **false HOLD** at 6.2 s | false HOLD at 15.8 s | false HOLD at 7.2 s |

\* The enhancements-OFF sessions were used to calibrate both `mic` profiles, so they are not a fair test.

**Findings:**
- **Premium clones are now held on voice alone with the default profiles.**
  - With `default`, both clones went from VERIFY to HOLD (ElevenLabs at 5.6 s, Kokoro at 13.1 s).
  - With `studio`, ElevenLabs went from VERIFY to HOLD at 12.3 s, but Kokoro dropped from HOLD at 16.5 s to VERIFY.
  - The Step 8c `studio` cut-off (0.00, capped) is strict, because the Hume calibration fakes are very easy for the fused score (median +8.5). It is still the rule we use, and we do not tune on the demo files.
  - In every setup, every clone gets at least VERIFY (with a scam script, fusion makes it HOLD), and genuine callers have even lower risk than before.
- **The processing classifier recognises real Windows noise suppression** as "noise suppression" (38/43 windows) and genuine raw-mic audio as "clean". This is the first evidence that it works on real device processing, not only on our simulations.
- **The Windows-enhancement false alarm is reduced but not solved.** Session 1 dropped from a false HOLD to VERIFY, but session 2 is still held. The simulated noise suppression in training does not cover Windows' noise suppression + beam-forming + auto-gain. **Rule unchanged: Windows audio enhancements OFF.**
- **The `auto` profile is more sensitive on raw mic audio:** a genuine session reached 44.6, just under VERIFY. Use the calibrated `mic` profile for live demos, and `auto` only for lines that have not been calibrated.

### Step 9: attack with a clone of the enrolled person (24 Nov 2026)

**Question:** what happens when the caller is an AI clone of the very person they claim to be? The voiceprint is then expected to match, so the detector and the scam rules must carry the decision.

**The clone (made locally, with consent; never uploaded):**
- **Method:** kNN-VC (WavLM-Large + prematched HiFi-GAN, `bshall/knn-vc`, the method of Step 3), run on the laptop CPU.
  - Script: `notebooks/build`-style helper in the scratchpad.
  - A small stand-in for `torchaudio`, because torch 2.15 has no matching torchaudio build.
- **Reference:** 119 s of the enrolled team member's own laptop recordings (`recordings/laptop_mic_enhancements_*.wav`).
- **Content:** the robot-voice bank-fraud script (`samples/scam_call_tts.wav`).
- **Output:** `recordings/step9_clone_of_enrolled_user.wav` (23.1 s). It is git-ignored, like all personal recordings.

**Offline, full pipeline** (`tools/attack_test.py`; the caller claims to be the enrolled person; `default` profiles):

| Caller | Voiceprint similarity (cut-off 0.45) | Detector windows flagged | Scam tactics | Verdict (original / Step 8c) |
|---|---|---|---|---|
| Clone of the enrolled person + scam script | **0.41** (min 0.31) | **18/18** (median score 10.6 / 14.1) | 3 of 4 | **HOLD at 4.4 s** / **HOLD at 4.4 s** |
| The enrolled person's real voice | 0.95 (min 0.79) | 0/19 | none | PASS (max 3) / PASS (max 9) |
| Robot-voice scam script | −0.06 | 18/18 | 4 | HOLD at 4.4 s / HOLD at 4.4 s |

**Through the live call:**
- **Setup:** two automated runs of the two-window demo (headless Chrome, a real WebRTC call, the clone uploaded on the caller page and played into the call). The customer claims to be the enrolled person, `default` profile, Step 8c detector.

| Run | VERIFY | HOLD | Voiceprint similarity (mean, min) | ₹49,999 transfer |
|---|---|---|---|---|
| 1 | 4.5 s (voice ≠ claimed person, 97/100) | 7.8 s (synthetic voice 78 + mismatch 80) | 0.32, 0.18 | **blocked automatically** |
| 2 | 3.8 s | 10.0 s | 0.33, 0.05 | **blocked automatically** |

- **Screenshots:** `docs/assets/ui-customer-hold.png` and `ui-caller-attack.png`, also used as the paper's Figure 3a/b.

**Findings:**
- **The voiceprint nearly accepted the clone.** Offline it scored 0.41, only 0.04 below the cut-off. A better clone would pass it, as expected.
- **The AI-voice detector carried the decision.** Every window was flagged by both detectors.
- **Through the call, the Opus channel lowered the clone's similarity further** (0.32), so identity mismatch also fired. HOLD came a little later (7.8–10 s) than offline, because of real-time analysis of call audio.
- **Scam words in the clone were transcribed slightly worse.** "Fraud department" became "Frog Department", so the "claims to be a bank" tactic was missed. The other three tactics were found.
- **Limits:**
  - the clone is open-source and made from 2 min of speech, and kNN-VC clones were easy for this detector in Step 3;
  - the test was digital, not played through a loudspeaker;
  - **a premium clone (e.g. ElevenLabs) of the enrolled person is still to be tested.**

The two-window call test also confirms the redesigned customer and caller pages work end to end: call setup, injection, verdicts, the bank-app block, incident logging.

### Step 10: genuine and cloned calls through WebRTC, and the `call` line profile (25 Nov 2026)

**Goal:** check for false alarms on genuine callers when their voice reaches the detector through the real call channel (WebRTC, Opus mono 128 kbit/s), and calibrate the `call` line profile. Until now, genuine callers had only been tested offline or on the laptop microphone.

**Method:**
- The same automated two-tab call as in Step 9: headless Chrome, audio muted, the recording injected into the call on the caller page.
- No identity claimed, so only the detector (Step 8c, fused) and the scam rules decide.
- **Calibration of `call`:** 48 s of genuine speech from the *dev* speakers only (yt_0001, yt_0003, yt_0007, yt_0013; never the demo speakers yt_0009 / yt_0010), played through the call. Then the customer page's "Calibrate with the caller's voice" button, the same rule as `mic` (95th percentile + 1.5, kept between -9.5 and 0).
- **Calls:** the four demo calls through the call, once with `default` (-1.91) and once with the new `call` profile.
- Numbers: `recordings/step10_webrtc_calls.csv`.

**Calibration:**
- 21 genuine windows: median -5.77, 95th percentile -0.87, maximum -0.67. Clean genuine speech scores about -9.6.
- The rule gave +0.63, capped at **0.00**.
- At the `default` threshold (-1.91), 3 of the 21 genuine windows (14%) were above the cut-off.
- **So the Opus call channel moves genuine dev speech about 4 log-odds units toward "fake".** This is the Step 8 Opus effect, here on a real WebRTC call at a high bitrate.

**Demo calls through WebRTC:**

| Call | `default` (-1.91) | `call` (0.00) |
|---|---|---|
| Genuine caller 1 | PASS, 0 of 31 windows flagged, max risk 2 | PASS, 0 of 31, max risk 0 |
| Genuine caller 2 | PASS, 0 of 31, max risk 40 | PASS, 0 of 31, max risk 40 |
| ElevenLabs clone | **HOLD 8.3 s** (VERIFY 6.0 s), 12 of 13 flagged | HOLD 9.3 s (VERIFY 7.3 s), 9 of 13 |
| Kokoro clone | **HOLD 14.1 s** (VERIFY 10.7 s), 10 of 15 flagged | VERIFY 15.8 s only, 5 of 15 |

Times are seconds of call audio. Genuine caller 2's risk of 40 comes from the scam rules, not the voice (no window flagged).

**Findings:**
- **No false alarm on genuine callers through the real call channel**, with either profile. Isolated high windows of dev speakers (14% at `default`) do not escalate, because HOLD needs 3 of 5 suspicious windows and a high running risk.
- **Both premium clones are held through the call with `default`.** The calibrated `call` profile is too lenient, because its cut-off hit the cap of 0: Kokoro only reached VERIFY.
- **Decision:**
  - The `call` profile was **not kept**.
  - The customer page selects `call` automatically when it exists, so keeping it would have weakened the call demo.
  - The call demo uses `default`.
  - Re-calibrate `call` only with a real genuine caller on the demo network, and compare it with `default` before using it.
- **Limits:**
  - two genuine demo speakers and four dev speakers;
  - one browser, one network (localhost);
  - a real network adds packet loss and lower Opus bitrates, which the Step 8 results suggest will move genuine speech further.

---

### Step 11: context-aware scam rules (26 Nov 2026)

**Problem:** a genuine bank warning such as "we will never ask for your OTP" matched the keyword rule "asks for OTP" (weight 0.6). On its own that reaches the scam-alert level (0.5).

**Change** (server/scam.py; config `scam_context`, default on; false restores the original behaviour):
- A context layer drops credential and secrecy matches that occur only inside protective advice, clause by clause. "Never share your OTP", "OTP kisi ko mat batayein", "किसी के साथ साझा न करें" and "ఎవరికీ చెప్పకండి" no longer count. "We never ask for your PIN, but I need the OTP" still counts.
- Named fraud scripts, used as explanations only:
  - fake bank / KYC;
  - digital arrest;
  - family emergency;
  - prize / refund;
  - remote-access takeover.
- More Hinglish phrasings, with the same tactics and weights.
- The keyword engine, its weights and the alert level are unchanged.

**Evaluation:** `tools/eval_scam.py` on `tests/data/scam_eval.json`: 24 short call transcripts (12 scam, 12 benign; English, Hinglish, Hindi, Telugu). The set was written by the team, so it is a sanity check, not an independent benchmark.

| Engine | TP | FP | FN | TN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| Original keyword rules | 12 | 7 | 0 | 5 | 0.63 | 1.00 | 0.77 |
| + Hinglish phrasings | 12 | 7 | 0 | 5 | 0.63 | 1.00 | 0.77 |
| + context layer | 12 | **2** | 0 | **10** | **0.86** | 1.00 | **0.92** |

**Remaining false alarms:**
- a genuine card-bill reminder from a bank (bank + payment: 0.52);
- a person mentioning their own OTP (0.60).

Fixing these needs intent understanding beyond rules; that is future work.

**Live check:** the Windows-TTS scam call through the console still reached HOLD, with "Fraud script recognised: Fake bank / KYC script" in the event feed.

---

## 9. Sources

**Problem statement:**
- [SIH26104 summary](https://sih2026.vuce.in/ps/SIH26104)
- [Tracker page](https://zaidsayyed.in/tools/sih-problem-statements/sih26104)

**Benchmarks and models:**
- [Speech DF Arena (arXiv 2509.02859)](https://arxiv.org/html/2509.02859)
- [AntiDeepfake repo](https://github.com/nii-yamagishilab/AntiDeepfake)
- [wav2vec-large-anti-deepfake model card](https://huggingface.co/nii-yamagishilab/wav2vec-large-anti-deepfake)
- [Post-training paper (arXiv 2506.21090)](https://arxiv.org/html/2506.21090v1)
- [XLS-R + SLS code](https://github.com/QiShanZhang/SLSforASVspoof-2021-DF)
- [XLSR-SLS mirror](https://huggingface.co/SpeechAntiSpoofingBenchmarks/XLSR-SLS)
- [Tak et al. 2022, SSL + RawBoost (arXiv 2202.12233)](https://arxiv.org/pdf/2202.12233)
- [Whisper features (Kawa 2023)](https://www.isca-archive.org/interspeech_2023/kawa23b_interspeech.pdf)

**Generalization and shortcuts:**
- [Does Audio Deepfake Detection Generalize?](https://arxiv.org/pdf/2203.16263)
- [Speech is Silver, Silence is Golden](https://arxiv.org/html/2106.12914v4)
- [Deepfake-Eval-2024](https://arxiv.org/html/2503.02857v5)
- [Audio deepfake detection vs advanced TTS](https://arxiv.org/html/2601.20510v1)
- [RADAR Challenge 2026](https://arxiv.org/html/2605.09568v3)

**Datasets:**
- [In-the-Wild](https://huggingface.co/datasets/mueller91/In-The-Wild)
- [MLAAD](https://huggingface.co/datasets/mueller91/MLAAD)
- [IndicSynth](https://huggingface.co/datasets/vdivyasharma/IndicSynth)
- [RTCFake](https://arxiv.org/html/2604.23742v1)
- [Hinglish scam calls](https://huggingface.co/datasets/DatasetNewUser/Indian_Cyber_Scam_PhoneCall_Hinglish_Dataset)
- [IndicF5](https://huggingface.co/ai4bharat/IndicF5/blob/main/README.md)
- [Kathbath](https://huggingface.co/datasets/ai4bharat/Kathbath)
- [IndicSUPERB](https://github.com/AI4Bharat/IndicSUPERB)
- [IndicVoices](https://huggingface.co/datasets/ai4bharat/IndicVoices)
- [IndicVoices-R paper (8 kHz note)](https://arxiv.org/html/2409.05356v2)
- [Svarah](https://huggingface.co/datasets/ai4bharat/Svarah)
- [SpoofCeleb](https://arxiv.org/pdf/2409.17285)
- [ASVspoof 5](https://arxiv.org/html/2502.08857v2)
- [Fake-or-Real](https://bil.eecs.yorku.ca/datasets/)
- [garystafford deepfake set](https://huggingface.co/datasets/garystafford/deepfake-audio-detection)
- [IndieFake paper](https://arxiv.org/abs/2506.19014)
- [Indic-CodecFake](https://helixometry.github.io/IndicFake/)
- [Dataset list (media-sec-lab)](https://github.com/media-sec-lab/Audio-Deepfake-Detection)

**Speaker verification:**
- [SASV multi-level fusion (arXiv 2203.15377)](https://arxiv.org/pdf/2203.15377)
- [SASV baselines (arXiv 2204.09976)](https://arxiv.org/pdf/2204.09976)

**Prevention:**
- [AudioSeal](https://arxiv.org/abs/2401.17264)
- [CloneShield](https://arxiv.org/pdf/2505.19119)
- [LLM real-time scam warnings](https://arxiv.org/pdf/2502.03964)

**Other teams:**
- [satyavaani](https://github.com/asmit-1101/satyavaani)
- [voice-cloning-detector](https://github.com/Tejahudson/voice-cloning-detector)
- [voxguard](https://github.com/RYashraj/voxguard)
- [SIH26104-voice-security](https://github.com/kalyanvijay05/SIH26104-voice-security)
- [Indian-language XLS-R student project](https://github.com/thehimanshhu/Group-3-DS-and-AI-Lab-Project)
