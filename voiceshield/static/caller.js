"use strict";
// Caller page: sends microphone or an injected recording to the customer over WebRTC.
const C = { sig: null, pc: null, ctx: null, out: null, micGain: null, clipGain: null, mic: null, micOn: true,
            micSrc: null, clipSrc: null, connected: false, clip: null, started: 0, tick: null, playT0: 0, playDur: 0 };
const I = (n, c = "") => VSUI.icon(n, c);

function setStatus(text, live) {
  $("status").textContent = text;
  $("statusDot").className = "speech-dot" + (live ? " on" : "");
}

async function init() {
  VSUI.mountTopbar("/caller");
  VSUI.hydrate();
  $("room").value = new URLSearchParams(location.search).get("room") || store.get("vs-room") || "demo";
  $("room").onchange = () => store.set("vs-room", $("room").value.trim());
  fillMicSelect($("micSelect"), "vs-caller-mic");
  let items = [];
  try { items = await api("/api/samples"); } catch (e) { /* none */ }
  $("clips").innerHTML = items.map((s, i) => {
    const bad = s.kind !== "genuine";
    return `<button class="item${i === 0 ? "" : ""}" data-url="/samples/${encodeURIComponent(s.file)}" data-label="${esc(s.label)}">` +
      `<span class="ic ${bad ? "crit" : "good"}">${I(bad ? "zap" : "user")}</span>` +
      `<span class="txt"><b>${esc(s.label)}</b><span>${esc(s.description || "")}</span></span>` +
      `<span class="tag ${bad ? "crit" : "good"}">${bad ? "AI voice" : "real voice"}</span></button>`;
  }).join("") || `<div class="empty">${I("music")}No demo recordings found in samples/</div>`;
  $("clips").querySelectorAll(".item").forEach((b) => { b.onclick = () => selectClip(b.dataset.url, b.dataset.label, b); });
  const first = $("clips").querySelector('.item .ic.crit');
  if (first) first.closest(".item").click();
  $("clipFile").onchange = () => {
    const f = $("clipFile").files[0];
    if (f) { selectClip(null, f.name, null, f); $("fileLbl").textContent = `Selected: ${f.name}`; }
  };
  $("callBtn").onclick = startCall;
  $("hangBtn").onclick = () => hangUp(true);
  $("micMode").onclick = () => useMic(!C.micOn);
  $("playBtn").onclick = playClip;
  $("stopPlayBtn").onclick = () => stopClip();
  $("askBtn").onclick = askTransfer;
  $("hearCustomer").onchange = () => { $("remoteAudio").muted = !$("hearCustomer").checked; };
}

function selectClip(url, label, el, file) {
  C.clip = { url, label, file };
  $("clips").querySelectorAll(".item").forEach((b) => b.classList.toggle("selected", b === el));
  if (!file) { $("clipFile").value = ""; $("fileLbl").textContent = "Or choose your own recording (a clone you made, wav / mp3 / flac)"; }
}

// Audio graph: mic -> micGain -> out ; clip -> clipGain -> out ; out.stream goes into the call.
async function buildAudio() {
  if (!C.ctx) {
    C.ctx = new AudioContext();                         // native rate (48 kHz), same as the mic and Opus
    C.out = C.ctx.createMediaStreamDestination();
    C.micGain = C.ctx.createGain();
    C.clipGain = C.ctx.createGain();
    C.micGain.connect(C.out);
    C.clipGain.connect(C.out);
  }
  await C.ctx.resume();
  if (!C.mic) {
    try {
      C.mic = await rawMic($("micSelect").value);
      C.micSrc = C.ctx.createMediaStreamSource(C.mic);
      C.micSrc.connect(C.micGain);
      fillMicSelect($("micSelect"), "vs-caller-mic");
      $("micState").textContent = `Microphone: ${C.mic.getAudioTracks()[0].label}. No browser processing.`;
    } catch (e) {
      $("micState").textContent = "No microphone available. You can still play recordings into the call.";
    }
  }
  return C.out.stream;
}

async function startCall() {
  const room = $("room").value.trim();
  if (!/^[A-Za-z0-9]{1,32}$/.test(room)) { alert("Room: letters and digits only"); return; }
  store.set("vs-room", room);
  const stream = await buildAudio();
  useMic(true);
  $("callBtn").disabled = true;
  setStatus("Connecting…");
  C.sig = new Signal(room, "caller", async (m) => {
    if (m.type === "joined") {
      if (m.peer) await makeOffer(stream); else setStatus("Waiting for the customer to go online…");
    } else if (m.type === "peer-joined") {
      await makeOffer(stream);
    } else if (m.type === "answer" && C.pc) {
      await C.pc.setRemoteDescription({ type: "answer", sdp: m.sdp });
    } else if (m.type === "ice" && C.pc) {
      try { await C.pc.addIceCandidate(m.candidate); } catch (e) { /* late candidate */ }
    } else if (m.type === "app" && m.action === "declined") {
      setStatus("The customer declined the call");
      hangUp(false);
    } else if (m.type === "bye" || m.type === "peer-left") {
      setStatus("The customer ended the call");
      hangUp(false);
    }
  });
  $("hangBtn").disabled = false;
}

async function makeOffer(stream) {
  if (C.pc) C.pc.close();
  setStatus("Ringing…");
  $("avatar").classList.add("ringing");
  C.pc = newPeer(C.sig, (remote) => { $("remoteAudio").srcObject = remote; }, (state) => {
    if (state === "connected") {
      C.connected = true; setStatus("In call with the customer", true); setTools(true);
      $("avatar").classList.remove("ringing");
      C.started = Date.now(); clearInterval(C.tick);
      C.tick = setInterval(() => {
        const s = Math.floor((Date.now() - C.started) / 1000);
        $("timer").textContent = `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
      }, 500);
    }
    if (state === "failed" || state === "disconnected") { setStatus("Call connection " + state); setTools(false); }
  });
  stream.getAudioTracks().forEach((t) => C.pc.addTrack(t, stream));
  const offer = await C.pc.createOffer();
  offer.sdp = hqOpus(offer.sdp);
  await C.pc.setLocalDescription(offer);
  C.sig.send({ type: "offer", sdp: offer.sdp });
}

function setTools(on) {
  $("playBtn").disabled = !on;
  $("askBtn").disabled = !on;
}

function useMic(on) {
  C.micOn = on;
  if (C.micGain) C.micGain.gain.value = on ? 1 : 0;
  $("micMode").className = "round" + (on ? " on" : "");
  $("micMode").innerHTML = I(on ? "mic" : "mic-off", "lg");
  $("micLbl").textContent = on ? "Mic live" : "Mic muted";
}

async function playClip() {
  stopClip();
  if (!C.clip) { alert("Choose a recording first"); return; }
  const bytes = C.clip.file ? await C.clip.file.arrayBuffer() : await (await fetch(C.clip.url)).arrayBuffer();
  const audio = await C.ctx.decodeAudioData(bytes);
  useMic(false);                                       // the attacker's own voice is not sent while the clip plays
  C.clipSrc = C.ctx.createBufferSource();
  C.clipSrc.buffer = audio;
  C.clipSrc.connect(C.clipGain);
  if ($("hearClip").checked) C.clipSrc.connect(C.ctx.destination);
  C.clipSrc.onended = () => stopClip(true);
  C.clipSrc.start();
  C.playT0 = C.ctx.currentTime; C.playDur = audio.duration;
  $("stopPlayBtn").disabled = false;
  $("playState").innerHTML = `${I("zap", "sm")} Playing <b>${esc(C.clip.label)}</b> into the call (${audio.duration.toFixed(0)} s). Your mic is muted meanwhile.`;
  const tickBar = () => {
    if (!C.clipSrc) return;
    $("playBar").style.width = `${Math.min(100, 100 * (C.ctx.currentTime - C.playT0) / C.playDur)}%`;
    requestAnimationFrame(tickBar);
  };
  tickBar();
}

function stopClip(ended) {
  if (C.clipSrc) {
    C.clipSrc.onended = null;
    try { C.clipSrc.stop(); } catch (e) { /* already stopped */ }
    C.clipSrc.disconnect();
    C.clipSrc = null;
    useMic(true);
    $("playState").textContent = ended ? "Recording finished. Your microphone is live again." : "";
  }
  $("playBar").style.width = "0";
  $("stopPlayBtn").disabled = true;
}

function askTransfer() {
  const amount = Math.max(1, Math.round(Number($("amount").value) || 0));
  C.sig.send({ type: "app", action: "transfer_request", amount, payee: $("payee").value.slice(0, 40) });
  $("askBtn").innerHTML = `${I("check")}Request for ₹${amount.toLocaleString("en-IN")} sent`;
  setTimeout(() => { $("askBtn").innerHTML = `${I("send")}Send payment request to the customer`; }, 3000);
}

function hangUp(tellPeer) {
  stopClip();
  if (tellPeer && C.sig) C.sig.send({ type: "bye" });
  if (C.pc) { C.pc.close(); C.pc = null; }
  if (C.sig) { C.sig.close(); C.sig = null; }
  C.connected = false;
  clearInterval(C.tick);
  $("avatar").classList.remove("ringing");
  setTools(false);
  $("callBtn").disabled = false;
  $("hangBtn").disabled = true;
  if (tellPeer) setStatus("You hung up");
}

init();
