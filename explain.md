# VoiceShield, explained in plain words

This file explains the whole project, from the problem to the final demo, for someone with no background in AI or programming. Every technical word is explained the first time it appears, and there is a glossary at the end.

---

## 1. The problem in one minute

Today a computer can copy anyone's voice from a few seconds of audio taken from a WhatsApp voice note, a YouTube video or a phone call. This is called **voice cloning**, and the copied voice is a **deepfake voice**.

Scammers use this in India every day:
- **Fake bank officer:** "Sir, this is SBI. Your account will be blocked in 30 minutes. Tell me the OTP we just sent."
- **"Digital arrest":** a fake police or CBI officer says you are under investigation and must pay to clear your name.
- **Fake relative:** your son's or daughter's voice says "I had an accident, send money now, don't tell anyone."

People trust a familiar voice, and a phone call gives them no time to think. That is why these scams work.

The **Smart India Hackathon 2026** problem **SIH26104** asks for a system that:
1. notices, *during the call*, that a voice may be cloned;
2. checks whether the caller really is who they claim to be;
3. **prevents** the damage, for example by stopping a bank transfer, instead of only giving a warning.

**VoiceShield** is our answer.

---

## 2. What VoiceShield does, in one picture

Think of VoiceShield as **three security guards listening to the same phone call**. Each guard checks something different. A manager combines their opinions every second.

```
Caller's voice ──► Guard A: "Does this voice sound machine-made?"      (AI-voice detector)
                ──► Guard B: "Is this really the person they claim?"  (voiceprint check)
                ──► Guard C: "Are they using scam words like OTP?"    (speech-to-text + scam rules)
                                      │
                               Manager (risk engine)
                                      │
              PASS  ─ call looks fine, payments allowed
              VERIFY ─ something is odd: confirm another way (OTP to your own phone, or call the bank back)
              HOLD  ─ likely a clone or a scam: payment blocked automatically, security team alerted
```

Why three guards and not one?
- A very good clone can fool the voice detector for a while. It still cannot hide scam words like "share your OTP".
- A real person can say scary things ("drugs", "police") in an innocent story. On its own that is not enough to block anything.
- A clone of a *different* person fails the voiceprint check straight away.

The guards cover for each other.

---

## 3. The key ideas, explained simply

### 3.1 The AI-voice detector (Guard A)
- **Model:** a computer program that learned a skill from examples. Ours is called **AntiDeepfake** and was built by a Japanese research institute (NII). It learned from about **74,000 hours** of real and fake speech in more than 100 languages.
- **Pre-trained model:** a model somebody else has already trained. We started from one, because training such a model from nothing needs far more data and computing power than a student team has. (Our own earlier attempts at that failed; see Step 0.)
- **Window:** the detector listens to the last **4 seconds** of speech and gives one answer. It does this again after **every 1 second** of new speech. One such 4-second piece is one "window".
- **Score:** a number for each window, from very negative (sounds real) to positive (sounds machine-made). Real voices usually score about −10.
- **Cut-off (threshold):** the line where we say "this window is suspicious". The right cut-off depends on the microphone and the audio channel, so we measure it for each setup. A **line profile** is the cut-off saved for one setup, for example `mic` for the laptop microphone or `studio` for clean recordings.
- **Calibration:** setting that cut-off from about 25 seconds of the real user's voice on the real microphone.

### 3.2 The voiceprint (Guard B)
- A **voiceprint** is a list of 256 numbers that describes *how a person sounds*, like a fingerprint for the voice. Recording one is called **enrolment**: 25 seconds of the person speaking.
- During a call, VoiceShield builds a voiceprint of the caller and compares it with the enrolled one. The result is a **similarity** score:
  - the same person scores about 0.6–0.9;
  - different people score about 0.0–0.25;
  - the cut-off is 0.45.
- Only these 256 numbers are stored, never the recording.

### 3.3 Scam language (Guard C)
- **Whisper** is a speech-to-text model. It turns the caller's words into text.
- Our rules look for **9 scam tactics** in English, Hindi and Telugu (written in English letters, Devanagari or Telugu script):
  - asking for an OTP, PIN or password;
  - asking to install a remote-access app such as AnyDesk;
  - asking to send money;
  - threatening police or arrest;
  - saying the account, SIM or KYC will be blocked;
  - claiming a family emergency;
  - promising a prize or lottery;
  - pressure to act fast or keep it secret;
  - claiming to be a bank or an official.
- The tactics add up. One innocent word stays low ("transfer" alone gives 40 out of 100). "Bank + account blocked + share OTP + don't tell anyone" gives 84.

### 3.4 The manager (risk engine)
- It keeps a running "AI-voice risk" that rises slowly, so one strange window cannot cause an alarm.
- **HOLD** happens when:
  - the voice risk is high (75 or more) and at least 3 of the last 5 windows look fake; or
  - scam words are heard **and** the voice or the identity was suspicious at some point.
- **VERIFY** happens when any single check is suspicious.
- A call can only move up (PASS → VERIFY → HOLD), never back down. A matching voice never *lowers* the risk.
- Every alert explains itself in plain words, for example "voice does not match the claimed person (100/100)".

### 3.5 How we measure "good"
- **False alarm:** a real person is flagged as fake.
- **Missed fake:** a cloned voice is called real.
- **EER (equal error rate):** move the cut-off until false alarms and missed fakes happen equally often. That shared percentage is the EER. **Lower is better.** 0% is perfect; 50% is coin-tossing.
- **Test set:** recordings the model never saw while it was being built or tuned, so it cannot have memorised them. Only results on test sets count.

---

## 4. The journey, step by step

### Step 0: why our first two attempts failed (before this project)
We had trained two models ourselves: one that looks at pictures of sound (spectrograms) and one built on a general speech model. Both gave wrong answers on new audio.

Research explained why. Models trained on one small dataset learn shortcuts, for example "clips with silence at the start are fake". They then fail on real-world audio, with error rates of 33–37%. Lesson: **start from a detector that has already seen a huge amount of fake speech.**

### Step 2: testing ready-made detectors
We tested two downloadable models:
- **AntiDeepfake:** 4.0% EER on "In-the-Wild" (real celebrity deepfakes from the internet). Good.
- **A popular model from the Hugging Face website:** 32–38% EER on anything outside its own training data. Useless for us.

Weak spot of AntiDeepfake: modern, expensive commercial voices such as ElevenLabs, with a 10% error rate.

### Step 3: a fair test in Indian languages
- **Why:** comparing real Hindi speakers with fake voices from a *different* source is unfair, because the model might just recognise the source.
- **What we did:** we made our own clones of 20 Hindi and 20 Telugu speakers *from their own voices* (a method called kNN-VC). Real and fake clips then share speaker, language and microphone.
- **Result:**
  - The detector separated them almost perfectly (0.0–0.1% EER).
  - Over a simulated phone line, errors rose to 1.5–5.2%. **The phone line was the first weakness we found.**

### Step 4: we tried to improve the detector, and it got worse
- **What we did:** we re-trained the detector on 3,760 of our own clips. Re-training a pre-trained model on new data is called **fine-tuning**.
- **What happened:** it got better on the voices it was trained on, but worse on real-world deepfakes.
  - This is called **forgetting**: a few thousand clips pull a model away from what it learned from 74,000 hours.
- **The safeguard that caught it:** we always compare against the original model on clips it was never trained on. So we kept the original.

### Step 5: the live system
- **What we built:** a server program that listens to audio as it arrives and decides every second.
- **Clever part 1:** a small voice-activity detector (Silero VAD) passes on only the parts with speech, so silence never confuses the detector.
- **Clever part 2:** the three guards run side by side, so a slow transcript never delays the voice decision.
- **Speed:** one check takes about **0.33 seconds** on the laptop's processor, fast enough for real time.

### Step 6: adding Guards B and C
- **What we added:** the voiceprint check and the scam-language rules, combined by the risk engine.
- **Result:** a scam call read by a robot voice reached **HOLD after 4.4 seconds**, with four tactics recognised.

### Steps 6b–6d: lessons from real microphones
- **First live tests:** the system flagged *our own real voice* as fake.
  - **Cause:** Windows "audio enhancements" (noise removal, auto-volume) change the voice before any app hears it. This made a real voice score −2 instead of −10.
  - **Fix:** turn the enhancements off and calibrate the microphone. Real voice: −10, no false alarms.
- **Controlled test:**
  - a team member and **ChatGPT's voice** read the *same* script into the laptop;
  - the human passed;
  - the AI voice reached **HOLD after 5.3 seconds**.

### Step 7: the call demo with prevention
Three screens show a realistic attack:
- **Caller screen:** the genuine caller, or the attacker, who can play a cloned recording *into the call*.
- **Customer phone screen:** the person being called, with a mock **bank app**.
- **Security screen:** the bank's security team.

The two browser windows are connected by a real internet call (WebRTC, the technology video-call apps use). In the test, the injected scam voice reached:
- **VERIFY after 3.2 seconds** of speech;
- **HOLD after about 4 seconds** of speech.

A transfer request of ₹49,999 was **blocked automatically**, and the incident appeared on the security screen.

### Step 8: everyday sound processing breaks detectors (our main research finding)
- **The idea:** phones, laptops, Bluetooth headsets and calling apps all change the sound before anyone hears it: noise removal, auto-volume, Bluetooth compression, and the "Opus" compression used by WhatsApp-style apps. We simulated seven of these and measured the detector.
- **Results:**
  - Noise removal raised the error on expensive clones from **8% to 34%**. Opus compression raised it to **45%**, almost coin-tossing. **A scammer gets this disguise for free, just by calling through an app.**
  - Opus also made **33% of real voices** look fake at the normal cut-off.
  - Bluetooth headsets hid even simple clones: errors went from 0% to 10%.

### Steps 8 and 8b: two more re-training attempts, both rejected
- **What we tried:** we re-trained only a tiny part of the model (1.1% of it, a method called **LoRA**). The second time, we added a "teacher" copy of the original model to stop it drifting away.
- **What happened:** both times the model learned the processed audio but *again* got worse on real-world deepfakes (errors 4.0% → 9.5%).
- **Lesson:** three re-training attempts showed that, with a few thousand clips, *changing the detector itself* does not work.

### Step 8c: the fix that worked, without changing the detector
- **The idea:** the detector works in 25 stages, like 25 floors of a building, and each floor passes its understanding up to the next. Its answer normally comes only from the top floor. We **left the detector untouched**, so it cannot forget anything, and looked inside.
- **Discovery:**
  - After noise removal, Bluetooth or Opus, **the middle floors still carry the clues**.
  - The top floor, the only one the detector normally uses for its answer, loses them.
  - So the evidence was not destroyed. The detector was just reading it from the wrong floor.
- **Fix 1: a small add-on model** (only 1,050 numbers, against 315 million in the detector). It reads all 25 floors, mostly the middle ones, and its answer is combined with the detector's own answer.
- **Fix 2: an "audio type" model** that recognises how the sound was processed (clean, phone line, noise removal, Bluetooth, Opus…) and can pick a suitable cut-off automatically.

| Results on test recordings never used for training | Before | After |
|---|---|---|
| Average error over 13 test sets | 9.7% | **4.1%** |
| Expensive clones, clean | 8.2% | **3.0%** |
| Expensive clones after noise removal | 34.0% | **12.7%** |
| Expensive clones after Opus (WhatsApp-style) | 45.1% | **16.7%** |
| Real-world internet deepfakes | 4.7% | 4.7% (nothing forgotten) |

What the audio-type model achieved:
- It cut false alarms on Opus audio from 33% to 13% for the original detector, and to 3% together with the add-on.
- Across all processed sets it mostly *traded* one kind of mistake for another. So it is an option, not the main fix.

### Step 8d: the fix inside the live system, tested on real recordings
The new detector is now switched on in the live server. One setting in `config.json` switches back to the old one. Checks:
- **Same detector:** the new model file gives exactly the same raw scores as the old one (difference 0.0003), and only about 5% more time per check (about 0.34 s).
- **Premium AI clones:** the ElevenLabs clone, which the old system only marked VERIFY, is now **held (HOLD) after 5.6 seconds**. The Kokoro clone is held after 13.1 seconds instead of only being marked VERIFY. Genuine callers still pass, with even lower risk.
- **Honest exception:** with the stricter `studio` setting, the Kokoro clone dropped from HOLD to VERIFY.
- **Still unsolved:** the laptop's Windows noise removal still caused one false alarm out of two sessions. **Rule for the demo: keep Windows audio enhancements OFF.**

### Step 9: the hardest attack, a clone of *you*
What if the scammer clones the voice of the very person they pretend to be? Then the voiceprint check (Guard B) may be fooled, because the voice really sounds like that person.

- **How we tested it:**
  - We made a clone of the enrolled team member's own voice on the laptop (kNN-VC), with his consent, from about 2 minutes of his own recordings, and made it read the bank-fraud script. Nothing was uploaded anywhere.
  - Then we attacked the system while claiming to be him: first offline, then by playing the clone *into a real call* between the Customer and Caller screens (an automated test, run twice).
- **What happened:**
  - **Guard B was almost fooled:** the clone's voiceprint similarity was 0.41, and the cut-off is 0.45. His real voice scores 0.95.
  - **Guard A caught it anyway:** the AI-voice detector flagged **every single window** (18 of 18) as machine-made.
  - **Offline:** HOLD after 4.4 seconds.
  - **Through the live call:** VERIFY after about 4 seconds, HOLD after 8–10 seconds, and the scammer's ₹49,999 payment request was **blocked automatically** both times.
- **The lesson:** this is exactly why VoiceShield uses several guards. When one is nearly fooled, another one catches the attack.
- **Still to test:** the same attack with an expensive commercial clone (for example ElevenLabs) of the enrolled person.

### The research paper
Everything above is written up as a two-column research paper (`docs/paper/voiceshield_paper.tex`). It includes the negative results, because reporting what did *not* work is part of honest science.

### The new interface (redesign)
The screens were redesigned as one professional, dark "security operations" application. Section 6 describes it.

---

## 5. One scam call, from start to finish

1. A scammer calls the customer and plays an AI-cloned voice into the call.
2. The customer's app sends the audio it *receives* to VoiceShield, in tiny pieces every 0.1 seconds. The audio is analysed in memory and **never saved**.
3. Silence is skipped. After 2 seconds of speech the first check runs, then one every second.
4. **Guard A** scores the last 4 seconds: "sounds machine-made". The AI-voice risk starts to climb.
5. **Guard C** turns the words into text: "this is your bank… your account will be blocked… share the OTP". It recognises three scam tactics.
6. **The manager** sees a suspicious voice *and* scam words. The verdict becomes **HOLD** within a few seconds of speech.
7. **Prevention:**
   - the bank app blocks the scammer's payment request on its own;
   - the security team's screen shows the incident with the reasons;
   - for a VERIFY, the app would instead require an OTP sent to the customer's *own* registered phone, or a call back to the bank's official number.
8. **Afterwards:** the audit log keeps only scores, decisions and the *names* of the scam tactics. It never keeps the audio, the transcript or the voiceprint.

---

## 6. What you see on the screens

Open `http://127.0.0.1:8000` after starting the server. The left sidebar has:

| Screen | What it is for |
|---|---|
| **Command center** | The main live screen. Start the microphone, a demo call or a file:<br>- a large **verdict** (PASS / VERIFY / HOLD, always with an icon and a word);<br>- the **AI-voice probability** of the latest window and how many of the last 5 windows looked fake;<br>- a **live waveform** of the actual sound being analysed, coloured cyan → amber → red as the decision changes;<br>- a **timeline of suspicious segments**, a risk chart, the three guards' meters, what the system did, an event list and the transcript. |
| **Voice analysis** | Load a recording and see its waveform and **spectrogram** (a picture of which pitches are loud when). Press play: every 4-second window is scored while you listen, the suspicious parts light up, and you get a final **Authentic / Suspicious / Spoofed** assessment. |
| **Incidents** | The bank security team's live list of protected calls, with the reasons and the actions taken. A new HOLD makes a notice pop up. |
| **Detection history** | Every past session, with search and filters. Click one to investigate it. |
| **Investigation** | One session in detail, with the evidence kept separate:<br>1. what the AI detector predicted;<br>2. supporting signals (voiceprint, scam tactics);<br>3. the security actions actually taken. |
| **Analytics** | Charts built from the real audit log (outcomes per day, risk levels, response speed, scam tactics heard, payments blocked), plus the offline accuracy test results, clearly labelled. |
| **Settings & enrolment** | Record voiceprints, calibrate the microphone, and see the rules the system uses. |
| **Customer phone / Caller** | The two ends of the demo call (Step 7). |

Design choices:
- Nothing on the screens is invented. Every number, chart and waveform comes from real audio or the real log.
- Danger is never shown by colour alone. There is always an icon and a word too.
- The whole interface works offline, so the demo cannot fail because of the internet.

---

## 7. Privacy

- **Never stored:** audio, transcripts and voiceprint *recordings*.
- **Stored:** scores, decisions, the names of scam tactics and prevention events (for example "payment blocked").
- **Where it runs:** everything runs on the laptop; nothing is sent to an outside company.
- **Deleting voiceprints:** they can be deleted from the Settings screen.
- **Licences:** the detector's licence allows only non-commercial use (research and education).

---

## 8. Honest limitations

1. A very good clone *of the enrolled person themselves* can pass the voiceprint check. That is exactly why the AI-voice detector exists.
2. Most noise-removal, Bluetooth and Opus results come from *simulations*. On a real laptop, Windows' own noise removal still causes some false alarms, so it must be switched off.
3. The scam rules look for words. They can fire on innocent sentences, which is why words alone never cause a HOLD.
4. The demo uses an internet call between two browsers, not a normal mobile network call. Ordinary Android apps are not allowed to read cellular call audio.
5. The test sets are small (hundreds of clips), so differences of one or two percentage points are not meaningful.
6. The laptop slows down when many heavy programs are open.

---

## 9. What is left to do

**Done:**
- the attack test with an open-source clone of the enrolled person (Step 9);
- the presentation slides;
- the demo video script (`docs/demo_video_script.md`);
- the paper's screenshots and GitHub link.

**Still open:**
1. **Premium attack test:** the same attack with a premium clone (for example ElevenLabs) of the enrolled person. This needs an account and the person's consent.
2. **Call calibration:** record real callers through the call app to set the `call` cut-off.
3. **Live microphone check** of the new detector with a real person speaking.
4. **Record the demo video** using the script.
5. **Paper details:** confirm the author list and three references.
6. **Optional:** run the detector on the graphics card, and use a real phone as the caller.

---

## 10. Glossary

| Word | Meaning |
|---|---|
| AI / model | A computer program that learned a skill from examples instead of being given rules |
| Deepfake voice / voice clone | A computer-generated imitation of a real person's voice |
| Pre-trained model | A model someone else already trained on a large amount of data |
| Fine-tuning / re-training | Training a pre-trained model a bit more on your own data |
| Forgetting | When re-training makes a model lose skills it had before |
| LoRA | A way to re-train only a tiny part of a big model |
| Window | The last 4 seconds of speech, checked once per second |
| Score (llr) | The detector's number for a window: negative = sounds real, positive = sounds fake |
| Cut-off / threshold | The score above which a window counts as suspicious |
| Line profile / calibration | The cut-off measured for one microphone or channel / the act of measuring it |
| Voiceprint / enrolment | 256 numbers describing how a person sounds / recording them |
| Similarity | How close two voiceprints are (1 = identical, 0 = unrelated) |
| Whisper / transcript | The speech-to-text model / the text it produces |
| Risk engine | The rules that combine the three checks into PASS / VERIFY / HOLD |
| PASS / VERIFY / HOLD | Fine / confirm another way / block payments and alert security |
| False alarm / missed fake | A real voice flagged as fake / a fake voice called real |
| EER | The error rate where false alarms equal missed fakes; lower is better |
| Test set | Recordings never used to build or tune the system, used only to measure it |
| Noise suppression | Software that removes background noise (in Windows, phones, apps) |
| Opus | The sound compression used by internet calling apps such as WhatsApp |
| Bluetooth CVSD / mSBC | The low-quality sound formats used by Bluetooth headsets during calls |
| WebRTC | The technology browsers use for live audio and video calls |
| Spectrogram | A picture of sound: time left to right, pitch bottom to top, loudness as brightness |
| Layers ("floors") | The 25 processing stages inside the detector |
| Back-end / add-on model | The small model we trained to read those layers (Step 8c) |
| ONNX | A file format that lets a trained model run fast without the training software |
| Kaggle | A free website with powerful graphics cards, where we trained and tested models |
| Audit log | The record of scores and decisions kept for the security team (no audio) |

---

## 11. Running it

On a Windows laptop, open a terminal in the `voiceshield` folder:

```bash
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python setup_models.py
.venv\Scripts\python run.py
```

Then open **http://127.0.0.1:8000** in Chrome or Edge.

The Step 8c files (`voiceshield_layers.onnx` and `step8c_backend.npz`) come from the Step 8c Kaggle notebook and go into `voiceshield/models/`. Without them the server automatically uses the original detector.

Before a live demo:
1. Turn **off** Windows audio enhancements.
2. Use the built-in or a wired microphone, not Bluetooth.
3. Calibrate the microphone once from **Settings & enrolment**.

---

## 12. Where everything is

| Folder or file | What is inside |
|---|---|
| `voiceshield/` | The working system: server (`server/`), screens (`static/`), demo recordings (`samples/`), tools, tests |
| `notebooks/` | The Kaggle notebooks used to test and train models (Steps 2–8c) |
| `docs/SIH26104_Research_and_Build_Plan.md` | The detailed research report and the full results log for every step |
| `docs/paper/voiceshield_paper.tex` | The research paper |
| `recordings/` | Result tables from the experiments (CSV files) |
| `explain.md` | This file |

Tests: 41 quick tests and 9 tests with the real models, all passing.
