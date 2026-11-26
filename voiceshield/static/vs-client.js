"use strict";
// Shared helpers for the call demo pages (customer / caller / security).

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (x, d = 2) => (x === null || x === undefined || Number.isNaN(Number(x)) ? "–" : Number(x).toFixed(d));
const wsUrl = (path) => (location.protocol === "https:" ? "wss://" : "ws://") + location.host + path;
const store = {                                         // per-browser conveniences only
  get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* private mode */ } },
};

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json();
}

function toInt16(f) {
  const o = new Int16Array(f.length);
  for (let i = 0; i < f.length; i++) {
    const s = Math.max(-1, Math.min(1, f[i]));
    o[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return o.buffer;
}

// Microphone without any browser processing: echo cancellation / noise suppression / auto gain change the
// voice and made genuine speech look synthetic in our tests (see docs, Step 6b).
function rawMic(deviceId) {
  return navigator.mediaDevices.getUserMedia({ audio: {
    channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false,
    ...(deviceId ? { deviceId: { exact: deviceId } } : {}) } });
}

async function fillMicSelect(sel, key) {
  if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
  const devs = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "audioinput");
  const want = sel.value || store.get(key) || "";
  sel.innerHTML = '<option value="">System default microphone</option>' + devs
    .filter((d) => d.deviceId && d.deviceId !== "default" && d.deviceId !== "communications")
    .map((d, i) => `<option value="${esc(d.deviceId)}">${esc(d.label || `Microphone ${i + 1}`)}</option>`).join("");
  if ([...sel.options].some((o) => o.value === want)) sel.value = want;
  sel.onchange = () => store.set(key, sel.value);
}

// High-quality Opus for the call: mono, 128 kbit/s, no DTX. Low-bit-rate speech codecs can also look "synthetic".
function hqOpus(sdp) {
  const m = sdp.match(/a=rtpmap:(\d+) opus\/48000/i);
  if (!m) return sdp;
  const pt = m[1];
  const params = "maxaveragebitrate=128000;stereo=0;sprop-stereo=0;useinbandfec=1;usedtx=0";
  const re = new RegExp(`a=fmtp:${pt} ([^\\r\\n]*)`);
  return re.test(sdp) ? sdp.replace(re, (_, p) => `a=fmtp:${pt} ${p.replace(/;?(maxaveragebitrate|stereo|sprop-stereo|usedtx)=\d+/g, "")};${params}`)
                      : sdp.replace(m[0], `${m[0]}\r\na=fmtp:${pt} ${params}`);
}

// Streams one MediaStream (e.g. the remote caller's audio) to /ws/stream and reports every server message.
class VSAnalyzer {
  constructor(onMessage, ctx = null) { this.onMessage = onMessage; this.ctx = ctx; this.running = false; }

  // Call from a click handler: browsers only let audio start after a user gesture.
  static async prepare(ctx) {
    if (!ctx) {
      try { ctx = new AudioContext({ sampleRate: 16000 }); } catch (e) { ctx = new AudioContext(); }
      try { await ctx.audioWorklet.addModule("/static/capture-worklet.js"); } catch (e) {
        ctx.close();                                     // not kept: the next attempt starts clean
        throw new Error("The audio capture module could not be loaded from the server. Is VoiceShield running?");
      }
    }
    await ctx.resume();
    return ctx;
  }

  async start(stream, startMsg) {
    this.ctx = await VSAnalyzer.prepare(this.ctx);
    this.src = this.ctx.createMediaStreamSource(stream);
    this.node = new AudioWorkletNode(this.ctx, "capture", { processorOptions: { chunk: Math.round(this.ctx.sampleRate / 10) } });
    this.mute = this.ctx.createGain();
    this.mute.gain.value = 0;
    this.node.connect(this.mute).connect(this.ctx.destination);
    const ws = new WebSocket(wsUrl("/ws/stream"));
    this.ws = ws;
    const ready = await new Promise((resolve, reject) => {
      ws.onopen = () => ws.send(JSON.stringify({ type: "start", sample_rate: this.ctx.sampleRate, ...startMsg }));
      ws.onmessage = (e) => { const m = JSON.parse(e.data); if (m.type === "ready") resolve(m); this.onMessage(m); };
      ws.onerror = () => reject(new Error("VoiceShield server not reachable"));
    });
    this.running = true;
    this.node.port.onmessage = (e) => {
      if (this.running && ws.readyState === WebSocket.OPEN) ws.send(toInt16(e.data));
    };
    this.src.connect(this.node);
    return ready;
  }

  stop() {
    if (!this.running) return;
    this.running = false;
    try { this.src.disconnect(); this.node.disconnect(); this.mute.disconnect(); } catch (e) { /* already gone */ }
    if (this.ws && this.ws.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify({ type: "stop" }));
  }
}

// Signalling channel to the other call page through the server (/ws/signal/<room>).
class Signal {
  constructor(room, role, onMessage) {
    this.ws = new WebSocket(wsUrl(`/ws/signal/${encodeURIComponent(room)}?role=${role}`));
    this.ws.onmessage = (e) => onMessage(JSON.parse(e.data));
    this.closed = new Promise((r) => { this.ws.onclose = r; });
  }
  send(msg) { if (this.ws.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify(msg)); }
  close() { try { this.ws.close(); } catch (e) { /* ignore */ } }
}

// Peer connection with sensible defaults for a local / same-Wi-Fi demo (no internet servers needed).
function newPeer(signal, onTrack, onState) {
  const pc = new RTCPeerConnection({ iceServers: [] });
  pc.onicecandidate = (e) => { if (e.candidate) signal.send({ type: "ice", candidate: e.candidate }); };
  pc.ontrack = (e) => onTrack(e.streams[0] || new MediaStream([e.track]));
  pc.onconnectionstatechange = () => onState(pc.connectionState);
  return pc;
}
