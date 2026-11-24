# VoiceShield demo video: script (4–5 minutes)

A ready-to-record script for the SIH / capstone demo video. Record the screen at 1920×1080 with OBS or the Windows Game Bar (Win+Alt+R), and the narration with a wired microphone. Everything shown is the real system; nothing is staged.

## Before recording (5 minutes of setup)
1. **Windows audio enhancements OFF** (Settings → System → Sound → your mic → Audio enhancements: Off). No Bluetooth headset.
2. Start the server: double-click `Start VoiceShield.bat`, or run `voiceshield\.venv\Scripts\python voiceshield\run.py`.
3. Open three Chrome windows side by side:
   - `http://127.0.0.1:8000/#/live` (console);
   - `http://127.0.0.1:8000/customer`;
   - `http://127.0.0.1:8000/caller`.
4. In **Settings & enrolment**:
   - enrol the presenter's voice (25 s);
   - calibrate the microphone (25 s).
5. Customer window: set *Caller claims to be* = the presenter, then click **Go online**.
6. Close other heavy apps, so the detector stays at about 0.35 s per window.

---

## Scene 1 · The problem (0:00–0:30)
**Show:** the GitHub page banner, or a title slide.

> "Scammers can now clone anyone's voice from a few seconds of audio: a bank officer asking for your OTP, a police officer threatening digital arrest, or your own child asking for money. A phone call gives the victim no time to think. VoiceShield listens to the call while it happens and stops the money from leaving."

## Scene 2 · How it decides (0:30–1:00)
**Show:** the README diagram "How it works", or the Command center with the three signal meters.

> "Three independent checks run every second: an AI detector that recognises machine-made voices, a voiceprint check against the person the caller claims to be, and scam-language rules on a live transcript in English, Hindi and Telugu. A risk engine turns them into PASS, VERIFY or HOLD, and every alert explains itself."

## Scene 3 · A genuine call passes (1:00–1:40)
**Do:**
1. Caller window: **Call**, then **Answer** in the Customer window.
2. The presenter speaks normally for 10–15 s.
3. Caller window: send a ₹2,000 payment request; the customer pays.

> "This is a genuine call. The voice matches the enrolled voiceprint, the detector sees a real voice, and the verdict stays PASS. A normal payment goes through."

## Scene 4 · The attack: a cloned voice with a scam script (1:40–3:00)
**Do:**
1. Caller window: **Attacker toolkit** → choose the scam call (or your own clone recording) → **Play into the call**.
2. After it reaches HOLD: **Send payment request** of ₹49,999.
3. Point at the Customer window: the HOLD verdict, the highlighted scam words, and the bank app showing *Payment blocked*.

> "Now the attacker plays a cloned voice into the same call. Within about four seconds of speech the detector flags the voice, the transcript catches 'your account will be blocked' and 'share the OTP', and the call goes to HOLD. The scammer asks for ₹49,999, and the bank app blocks it automatically. The customer never had to decide under pressure."

## Scene 5 · What the security team sees (3:00–3:40)
**Do:**
1. Console: open **Incidents**.
2. Click the call: the **Investigation** view.
3. Show the three columns: model prediction, supporting signals, actions taken.

> "The bank's security team sees the incident immediately, with the reasons, the scam tactics and the blocked payment. The investigation view separates what the AI predicted from the evidence and from the actions taken. No audio or transcript is ever stored, only scores and decisions."

## Scene 6 · The research behind it (3:40–4:30)
**Do:**
1. Console: **Voice analysis** → pick *Cloned voice: ElevenLabs* → **Analyse & play** (let 5–6 s play).
2. Then show the README charts.

> "We found that everyday audio processing (noise suppression, Bluetooth and WhatsApp-style compression) can hide a premium clone: the error rose from 8 to 45 percent. Retraining the detector made it forget real-world fakes, three times. Our fix leaves the detector untouched and reads its middle layers instead. That halved the errors, and a premium ElevenLabs clone that used to get only VERIFY is now held."

## Scene 7 · Close (4:30–4:50)
**Show:** the Analytics view or the GitHub page.

> "VoiceShield: real-time detection and prevention of voice-cloning scams, built for SIH26104. The code, the experiments and the research paper are open on GitHub."

---

### Tips
- If a false alarm happens on the real voice: check that Windows enhancements are off, then recalibrate in Settings.
- Rehearse Scene 4 once. HOLD needs about 4 s of the attacker's *speech*, not 4 s of call time.
- Keep the mouse still while the verdict changes; let the screen do the talking.
