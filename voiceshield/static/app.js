"use strict";
// VoiceShield console: one page, seven views (hash routes). Streams the microphone or an audio file to
// /ws/stream exactly as before and renders the server's real messages; nothing here invents a score.
// Helpers $, esc, fmt, wsUrl, store, api, toInt16, rawMic, fillMicSelect come from vs-client.js.
const I = (n, c = "") => VSUI.icon(n, c);
const capital = (s) => (s ? s.charAt(0).toUpperCase() + s.slice(1) : "");
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const MAX_REC_S = 120;
const LEVEL_WORD = { PASS: "Authentic", VERIFY: "Suspicious", HOLD: "Spoofed" };
const LEVEL_ICON = { PASS: "shield-check", VERIFY: "alert", HOLD: "octagon" };

// one audio session at a time (mic, demo call or file), shared by every view
const S = { ws: null, ctx: null, node: null, mute: null, src: null, stream: null, running: false, mode: "monitor", origin: "live",
            profile: "default", claimed: "", enrollName: "", cfg: null, points: [], wins: [], clock: 0, speech: 0, level: null,
            calTimer: null, calN: 0, rec: [], recLen: 0, recRate: 16000, isMic: false, peak: 0, clipped: 0, chunks: 0, events: 0,
            procSeen: {}, lat: [] };
const LV = {};                                   // live widgets
const AN = { buffer: null, name: "", t0: 0, raf: 0, dur: 0, spec: null, samples: null, first: {} };
const VIEWS = {
  live: ["/#/live", "Command center"], analysis: ["/#/analysis", "Voice analysis"], history: ["/#/history", "Detection history"],
  call: ["/#/history", "Investigation"], analytics: ["/#/analytics", "Analytics"], incidents: ["/#/incidents", "Incidents"],
  settings: ["/#/settings", "Settings & enrolment"],
};
const fmtClock = (s) => { s = Math.max(0, s || 0); const m = Math.floor(s / 60); return `${m}:${(s - m * 60).toFixed(1).padStart(4, "0")}`; };
const fmtDate = (ts) => { const d = new Date(ts * 1000); return `${d.toLocaleDateString(undefined, { day: "2-digit", month: "short" })} · ${d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}`; };
const vbadge = (v) => `<span class="vbadge ${esc(v)}">${I(LEVEL_ICON[v] || "shield", "sm")}${esc(v)}</span>`;
function riskbar(v) {
  const n = clamp(Number(v) || 0, 0, 100);
  return `<div class="riskbar"><div class="track"><div class="fill ${n >= 75 ? "high" : n >= 45 ? "mid" : ""}" style="width:${n}%"></div></div><span class="num">${n.toFixed(0)}</span></div>`;
}

// ====================================================================== boot + router
async function init() {
  VSUI.mountTopbar("/#/live", "Command center");
  VSUI.hydrate();
  LV.chart = new VSUI.RiskChart($("chart"), { height: 210 });
  LV.wave = new VSUI.LiveWave($("wave"), { seconds: 8 });
  LV.lane = new VSUI.SegLane($("lane"));
  AN.lane = new VSUI.SegLane($("aLane"));
  $("sessionPill").onclick = () => { location.hash = S.origin === "analysis" ? "#/analysis" : S.origin === "settings" ? "#/settings" : "#/live"; };
  try {
    S.cfg = await api("/api/status");
    LV.chart.thresholds(S.cfg.risk.verify_at, S.cfg.risk.hold_at);
    fillProfiles(S.cfg.profiles);
  } catch (e) { VSUI.toast("crit", "Server not reachable", "Start it with  python run.py"); }
  bindLive(); bindAnalysis(); bindHistory(); bindSettings();
  $("iAll").checked = store.get("vs-sec-all") === "1";
  $("iAll").onchange = () => { store.set("vs-sec-all", $("iAll").checked ? "1" : "0"); INC.seen = null; refreshIncidents(); };
  $("anAll").checked = store.get("vs-an-all") !== "0";
  $("anAll").onchange = () => { store.set("vs-an-all", $("anAll").checked ? "1" : "0"); loadAnalytics(); };
  $("anRefresh").onclick = () => loadAnalytics();
  loadSamples(); loadVoiceprints();
  setVerdict("IDLE"); emptyTranscript(); renderResponse(null); setSignals({ voice: null, identity: null, scam: null });
  window.addEventListener("hashchange", route);
  route();
  pollBadge(); setInterval(pollBadge, 6000);
}

let CURRENT = "";
function route() {
  const h = location.hash.replace(/^#\/?/, "") || "live";
  let [name, arg] = h.split("/");
  if (!VIEWS[name]) name = "live";
  for (const k of Object.keys(VIEWS)) $(`v-${k}`).classList.toggle("hidden", k !== name);
  const el = $(`v-${name}`); el.classList.remove("view"); void el.offsetWidth; el.classList.add("view");
  const [href, title] = VIEWS[name];
  VSUI.setActive(href, name === "call" ? `Call #${arg || ""}` : title);
  document.title = `VoiceShield · ${name === "call" ? `Call #${arg}` : title}`;
  if (CURRENT !== name) window.scrollTo({ top: 0 });
  CURRENT = name;
  clearInterval(INC.timer);
  if (name === "live") requestAnimationFrame(() => { LV.wave.draw(); LV.lane.draw(); LV.chart.draw(); });
  if (name === "analysis") requestAnimationFrame(() => { redrawAnalysis(); });
  if (name === "history") loadHistory();
  if (name === "call") loadCall(arg);
  if (name === "analytics") loadAnalytics();
  if (name === "incidents") { refreshIncidents(); INC.timer = setInterval(refreshIncidents, 2000); }
  if (name === "settings") renderSettings();
}

// ====================================================================== shared lists
function fillProfiles(profiles, choose) {
  const opt = ([n, p]) => `<option value="${esc(n)}" title="${esc(p.source || "")}">${esc(n)} · cut-off ${p.auto ? "per audio type" : fmt(p.llr_threshold)}</option>`;
  for (const [id, key] of [["profile", "vs-profile"], ["aProfile", "vs-a-profile"]]) {
    const sel = $(id), keep = (id === "profile" && choose) || sel.value || store.get(key);
    sel.innerHTML = Object.entries(profiles).map(opt).join("");
    if (keep && profiles[keep]) sel.value = keep;
    sel.onchange = () => store.set(key, sel.value);
  }
  if (choose) store.set("vs-profile", choose);
}

let SAMPLES = [];
async function loadSamples() {
  try { SAMPLES = await api("/api/samples"); } catch (e) { SAMPLES = []; }
  $("samples").innerHTML = SAMPLES.map((s, i) => {
    const bad = s.kind !== "genuine";
    return `<button class="item" data-i="${i}"><span class="ic ${bad ? "crit" : "good"}">${I(bad ? "zap" : "user")}</span>` +
      `<span class="txt"><b>${esc(s.label)}</b><span>${esc(s.description || "")}</span></span><span class="go">${I("play", "sm")}</span></button>`;
  }).join("") || `<div class="empty"><i data-i="music"></i>No demo recordings in samples/.</div>`;
  VSUI.hydrate($("samples"));
  $("samples").querySelectorAll(".item").forEach((b) => {
    const s = SAMPLES[Number(b.dataset.i)];
    b.onclick = () => {
      if (s.profile && S.cfg && S.cfg.profiles[s.profile]) $("profile").value = s.profile;
      start({ kind: "url", url: "/samples/" + encodeURIComponent(s.file), label: s.file, origin: "live" });
    };
  });
  $("aSample").innerHTML = `<option value="">Choose a demo call…</option>` +
    SAMPLES.map((s, i) => `<option value="${i}">${esc(s.label)}</option>`).join("");
}

async function loadVoiceprints(select) {
  let vps = {};
  try { vps = await api("/api/voiceprints"); } catch (e) { return; }
  const names = Object.keys(vps).sort(), sel = $("claimed");
  const keep = select ?? (sel.value || store.get("vs-claimed") || "");
  sel.innerHTML = `<option value="">${names.length ? "Nobody (skip voice check)" : "Nobody enrolled"}</option>` +
    names.map((n) => `<option value="${esc(n)}">${esc(n)}</option>`).join("");
  if (names.includes(keep)) sel.value = keep;
  sel.onchange = () => { store.set("vs-claimed", sel.value); showClaim(); };
  showClaim();
  $("vpList").innerHTML = names.map((n) =>
    `<div class="item" style="cursor:default"><span class="ic brand">${I("fingerprint")}</span>` +
    `<span class="txt"><b>${esc(n)}</b><span>${fmt(vps[n].seconds, 0)} s of speech · enrolled ${esc(vps[n].created || "")}</span></span>` +
    `<button class="btn sm ghost" data-name="${esc(n)}" aria-label="Delete the voiceprint of ${esc(n)}">${I("trash", "sm")}</button></div>`).join("") ||
    `<div class="empty" style="padding:14px"><i data-i="users"></i>Nobody enrolled yet.</div>`;
  VSUI.hydrate($("vpList"));
  $("vpList").querySelectorAll("button[data-name]").forEach((b) => {
    b.onclick = async () => {
      if (!await VSUI.confirmDialog("Delete voiceprint?", `The voiceprint of ${b.dataset.name} is removed from this server. Calls can no longer be checked against it.`, "Delete", true)) return;
      await fetch("/api/voiceprints/" + encodeURIComponent(b.dataset.name), { method: "DELETE" });
      VSUI.toast("good", "Voiceprint deleted", b.dataset.name);
      loadVoiceprints(); VSUI.loadStatus();
    };
  });
}
function showClaim() {
  const n = $("claimed").value;
  $("sigIdS").textContent = n ? `Compared with ${n}'s enrolled voiceprint` : "Pick who the caller claims to be";
}

// ====================================================================== live view: display
function bindLive() {
  $("lang").value = store.get("vs-lang") || "";
  $("lang").onchange = () => store.set("vs-lang", $("lang").value);
  $("micBtn").onclick = () => start({ kind: "mic", origin: "live" });
  $("stopBtn").onclick = () => stop();
  $("fileInput").onchange = (e) => {
    const f = e.target.files[0];
    if (f) start({ kind: "file", file: f, label: f.name, origin: "live" });
    e.target.value = "";
  };
  $("saveBtn").onclick = saveRecording;
  fillMicSelect($("micSelect"), "vs-mic");
  if (navigator.mediaDevices) navigator.mediaDevices.addEventListener("devicechange", () => fillMicSelect($("micSelect"), "vs-mic"));
}

function setButtons() {
  $("micBtn").disabled = S.running; $("calBtn").disabled = S.running; $("enrollBtn").disabled = S.running;
  $("stopBtn").disabled = !S.running;
  $("aStop").disabled = !(S.running && S.origin === "analysis");
  $("aPlay").disabled = S.running || !AN.buffer;
  $("micBtn").innerHTML = S.running && S.isMic && S.mode === "monitor" ? `${I("live")}Monitoring…` : `${I("mic")}Start monitoring`;
}

function setVerdict(level, opts = {}) {
  const v = VSUI.VERDICT[level] || VSUI.VERDICT.IDLE;
  $("verdictBox").className = "verdict " + level;
  $("vIcon").innerHTML = I(v.icon, "sm");
  $("vState").textContent = opts.state || v.state;
  $("vTitle").textContent = opts.title || v.title;
  $("vWhy").textContent = opts.reasons && opts.reasons.length ? capital(opts.reasons.join("; ")) + "." : (opts.why || v.why);
  VSUI.setGauge($("gauge"), opts.risk ?? null, ["PASS", "VERIFY", "HOLD"].includes(level) ? level : "");
  $("verdict").innerHTML = opts.verdict ? vbadge(opts.verdict) : "–";
  if (S.running) LV.wave.setState(level === "CAL" ? "CAL" : level === "VERIFY" || level === "HOLD" ? level : "listen");
  const pill = $("wavePill");
  pill.className = "state-pill " + (S.running ? (["VERIFY", "HOLD"].includes(level) ? level : "listen") : "");
  pill.innerHTML = !S.running ? "Idle" : level === "CAL" ? `${I("sliders", "xs")}Calibrating` :
    level === "HOLD" ? `${I("octagon", "xs")}Threat · held` : level === "VERIFY" ? `${I("alert", "xs")}Suspicious` : `<span class="dot cyan pulse"></span>Listening`;
}

function setSignals(sig) {
  const one = (bar, val, v) => {
    if (v === null || v === undefined) { $(bar).style.width = "0"; $(val).textContent = "off"; $(val).className = "v off"; return; }
    $(bar).style.width = `${clamp(v, 0, 100)}%`;
    $(bar).className = "meter-fill" + (v >= 75 ? " high" : v >= 45 ? " mid" : "");
    $(val).textContent = Number(v).toFixed(0); $(val).className = "v";
  };
  one("sigVoice", "sigVoiceV", sig.voice); one("sigId", "sigIdV", sig.identity); one("sigScam", "sigScamV", sig.scam);
}

function renderResponse(verdict) {
  $("response").querySelectorAll(".act").forEach((a) => {
    const l = a.dataset.l, on = verdict === l;
    a.className = "act " + (verdict ? (on ? "on " + l : "off") : "");
  });
}

function sessionPill() {
  const p = $("sessionPill");
  if (!S.running) { p.classList.add("hidden"); return; }
  const lvl = S.level || (S.mode === "monitor" ? "PASS" : "");
  const what = S.mode === "calibrate" ? "Calibrating" : S.mode === "enroll" ? "Enrolling" : S.origin === "analysis" ? "Analysing" : "Live";
  p.className = "session-pill live " + (["VERIFY", "HOLD"].includes(lvl) ? lvl : "");
  p.innerHTML = `<span class="dot ${lvl === "HOLD" ? "crit pulse" : lvl === "VERIFY" ? "warn" : "cyan pulse"}"></span><span class="lbl">${what}</span><b>${fmtClock(S.clock)}</b>` +
    (S.mode === "monitor" ? `<span class="vbadge ${lvl}" style="height:20px">${lvl}</span>` : "");
}

function emptyTranscript(text) {
  $("transcript").innerHTML = `<div class="empty">${I("message")}${esc(text || "The caller's words appear here, with scam phrases highlighted.")}</div>`;
}
function highlight(text, hits) {
  let html = esc(text);
  for (const h of hits || []) {
    const p = esc(h.phrase).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    html = html.replace(new RegExp(p, "i"), (s) => `<mark title="${esc(h.label)}">${s}</mark>`);
  }
  return html;
}
function addEvent(cls, ic, title, detail, t) {
  $("eventsEmpty").classList.add("hidden");
  const li = document.createElement("li");
  li.className = cls;
  li.innerHTML = `${I(ic)}<div><b>${esc(title)}</b>${detail ? ` · ${esc(detail)}` : ""}<time>${t === undefined ? "" : fmt(t, 1) + " s into the session"}</time></div>`;
  $("events").prepend(li);
}

function resetSession(mode) {
  S.points = []; S.wins = []; S.clock = 0; S.speech = 0; S.calN = 0; S.level = null; S.procSeen = {}; S.lat = []; S.tactics = new Set();
  $("log").querySelector("tbody").innerHTML = ""; $("logEmpty").classList.remove("hidden");
  $("action").className = "alert hidden";
  $("events").innerHTML = ""; $("eventsEmpty").classList.remove("hidden");
  $("clock").textContent = "0.0 s"; $("speechS").textContent = "0.0 s"; $("speechDot").className = "speech-dot";
  $("mProb").textContent = "–"; $("mWin").textContent = "–"; $("lat").textContent = "–";
  const special = { calibrate: ["Calibrating", "Learning this line"], enroll: ["Enrolling", "Recording a voiceprint"] }[mode];
  if (special) setVerdict("CAL", { state: special[0], title: special[1], why: "Keep talking normally for about 25 s." });
  else setVerdict("PASS", { risk: 0, verdict: "PASS" });
  $("calStatus").textContent = "";
  emptyTranscript(mode === "monitor" ? "Listening…" : "No transcript while calibrating or enrolling.");
  setSignals({ voice: 0, identity: mode === "monitor" && S.claimed ? 0 : null, scam: 0 });
  renderResponse(mode === "monitor" ? "PASS" : null);
  LV.chart.set([], 0); LV.lane.set([], 0); LV.wave.clear();
  $("waveEmpty").classList.add("hidden");
}

function addRow(m, level) {
  $("logEmpty").classList.add("hidden");
  const tb = $("log").querySelector("tbody"), tr = document.createElement("tr");
  const sim = m.speaker ? fmt(m.speaker.similarity, 2) : "–";
  const tip = m.llr_original === undefined ? "" : ` title="base detector score ${fmt(m.llr_original)} · cut-off for this window ${fmt(m.window_threshold)}"`;
  tr.innerHTML = `<td>${fmt(m.t, 1)}</td><td${tip} class="ink">${fmt(m.llr)}</td><td>${fmt(m.p_fake, 4)}</td>` +
    `<td>${m.processing ? esc(m.processing) : "–"}</td><td>${sim}</td><td class="r">${m.risk === undefined ? "–" : fmt(m.risk, 0)}</td>` +
    `<td>${level ? vbadge(level) : '<span class="muted">calibration</span>'}</td><td class="r">${fmt(m.latency_ms, 0)}</td>`;
  tb.prepend(tr);
  while (tb.rows.length > 300) tb.deleteRow(-1);
}

// ====================================================================== audio engine
async function ensureCtx(resume = true) {
  if (!S.ctx) {
    try { S.ctx = new AudioContext({ sampleRate: 16000 }); } catch (e) { S.ctx = new AudioContext(); }
    await S.ctx.audioWorklet.addModule("/static/capture-worklet.js");
  }
  if (resume) await S.ctx.resume();                     // playback needs a click; decoding does not
  return S.ctx;
}

function trackLevel(f) {
  let peak = 0, sum = 0;
  for (let i = 0; i < f.length; i++) { const a = Math.abs(f[i]); if (a > peak) peak = a; sum += f[i] * f[i]; }
  const rms = Math.sqrt(sum / f.length);
  S.peak = Math.max(S.peak, peak); S.chunks += 1;
  if (peak >= 0.99) S.clipped += 1;
  if (S.chunks % 4 === 0) {
    $("levelInfo").textContent = `input ${(20 * Math.log10(rms + 1e-9)).toFixed(0)} dBFS · peak ${(20 * Math.log10(S.peak + 1e-9)).toFixed(1)} dBFS` +
      (S.clipped ? ` · clipping in ${S.clipped} chunks: move back from the mic` : "");
  }
}
function keepForRecording(i16) {
  if (S.recLen >= MAX_REC_S * S.recRate) return;
  S.rec.push(new Int16Array(i16)); S.recLen += i16.byteLength / 2;
}
function saveRecording() {
  if (!S.recLen) return;
  const data = new Int16Array(S.recLen);
  let o = 0;
  for (const c of S.rec) { data.set(c, o); o += c.length; }
  const hdr = new DataView(new ArrayBuffer(44));
  const w = (p, s) => [...s].forEach((ch, i) => hdr.setUint8(p + i, ch.charCodeAt(0)));
  w(0, "RIFF"); hdr.setUint32(4, 36 + data.byteLength, true); w(8, "WAVE"); w(12, "fmt ");
  hdr.setUint32(16, 16, true); hdr.setUint16(20, 1, true); hdr.setUint16(22, 1, true);
  hdr.setUint32(24, S.recRate, true); hdr.setUint32(28, S.recRate * 2, true); hdr.setUint16(32, 2, true);
  hdr.setUint16(34, 16, true); w(36, "data"); hdr.setUint32(40, data.byteLength, true);
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([hdr, data.buffer], { type: "audio/wav" }));
  a.download = `voiceshield_mic_${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}.wav`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}

function openSocket(source, sampleRate) {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(wsUrl("/ws/stream"));
    S.ws = ws;
    ws.onopen = () => ws.send(JSON.stringify({
      type: "start", sample_rate: sampleRate, source, mode: S.mode, profile: S.profile,
      save_as: S.mode === "enroll" ? S.enrollName : "mic", claimed: S.claimed || null, language: $("lang").value || null }));
    ws.onmessage = (e) => {
      const m = JSON.parse(e.data);
      if (m.type === "ready") resolve(m);
      if (S.ws === ws) onMessage(m);
    };
    ws.onerror = () => reject(new Error("Could not reach the server WebSocket."));
    ws.onclose = () => { if (S.ws === ws) S.ws = null; HIST.stale = true; };
  });
}

async function start(opts) {
  if (S.running) await stop();
  S.mode = opts.calibrate ? "calibrate" : opts.enroll ? "enroll" : "monitor";
  S.origin = opts.origin || "live";
  S.enrollName = opts.enroll || "";
  S.claimed = S.mode === "monitor" && S.origin === "live" ? $("claimed").value : "";
  S.profile = opts.profile || $("profile").value;
  resetSession(S.mode);
  if (S.origin === "analysis") resetAnalysisResults();
  try {
    const ctx = await ensureCtx();
    let source;
    S.isMic = opts.kind === "mic";
    S.peak = 0; S.clipped = 0; S.chunks = 0;
    S.mute = ctx.createGain(); S.mute.gain.value = 0; S.mute.connect(ctx.destination);
    if (S.isMic) {
      S.stream = await rawMic($("micSelect").value);
      S.src = ctx.createMediaStreamSource(S.stream);
      const track = S.stream.getAudioTracks()[0], st = track.getSettings ? track.getSettings() : {};
      const label = track.label || "unknown microphone";
      $("srcLabel").textContent = `${label} · ${st.sampleRate || "?"} Hz → ${ctx.sampleRate} Hz · echo-cancel ${st.echoCancellation ? "ON" : "off"} · ` +
        `noise suppression ${st.noiseSuppression ? "ON" : "off"} · auto-gain ${st.autoGainControl ? "ON" : "off"}`;
      source = `${{ calibrate: "mic-cal", enroll: "mic-enroll" }[S.mode] || "mic"}:${label}`.slice(0, 120);
      S.rec = []; S.recLen = 0; S.recRate = ctx.sampleRate;
      $("saveBtn").disabled = true;
      fillMicSelect($("micSelect"), "vs-mic");
    } else {
      let audio = opts.buffer;
      if (!audio) {
        const bytes = opts.kind === "file" ? await opts.file.arrayBuffer() : await (await fetch(opts.url)).arrayBuffer();
        audio = await ctx.decodeAudioData(bytes);
      }
      S.src = ctx.createBufferSource();
      S.src.buffer = audio;
      S.src.connect(ctx.destination);                                    // you hear the file while it is analysed
      S.src.onended = () => { if (S.running) stop(); };
      source = "file:" + opts.label;
      $("srcLabel").textContent = `Recording · ${opts.label} · ${audio.duration.toFixed(1)} s · played and streamed in real time`;
      $("levelInfo").textContent = "input level –";
    }
    S.node = new AudioWorkletNode(ctx, "capture", { processorOptions: { chunk: Math.round(ctx.sampleRate / 10) } });
    S.node.connect(S.mute);                                               // keeps the worklet running, silently
    await openSocket(source, ctx.sampleRate);
    LV.wave.attach(ctx);
    S.node.port.onmessage = (e) => {
      if (!(S.running && S.ws && S.ws.readyState === WebSocket.OPEN)) return;
      const pcm = toInt16(e.data);
      S.ws.send(pcm);
      LV.wave.feed(e.data);                              // the waveform shows exactly the audio sent to the detector
      trackLevel(e.data);
      if (S.isMic) keepForRecording(pcm);
    };
    S.src.connect(S.node);
    if (S.src.start) S.src.start();
    S.running = true;
    LV.wave.setState(S.mode === "monitor" ? "listen" : "CAL");
    setVerdict(S.mode === "monitor" ? "PASS" : "CAL", S.mode === "monitor" ? { risk: 0, verdict: "PASS" } :
      { state: S.mode === "enroll" ? "Enrolling" : "Calibrating", title: S.mode === "enroll" ? "Recording a voiceprint" : "Learning this line", why: "Keep talking normally for about 25 s." });
    if (S.origin === "analysis") { AN.t0 = ctx.currentTime; playhead(); }
    setButtons(); sessionPill();
    if (S.mode !== "monitor") countdown(25, S.mode === "enroll" ? "vpStatus" : "calStatus");
  } catch (e) {
    VSUI.toast("crit", "Could not start", e.message || String(e));
    S.running = true;
    await stop();
  }
}

async function stop() {
  if (!S.running) return;
  S.running = false;
  clearInterval(S.calTimer);
  try { if (S.src && S.src.buffer) S.src.stop(); } catch (e) { /* already ended */ }
  try { if (S.src) S.src.disconnect(); } catch (e) { /* ignore */ }
  if (S.stream) S.stream.getTracks().forEach((t) => t.stop());
  if (S.node) { S.node.port.onmessage = null; S.node.disconnect(); }
  LV.wave.detach();
  if (S.mute) S.mute.disconnect();
  S.src = S.stream = S.node = S.mute = null;
  if (S.ws && S.ws.readyState === WebSocket.OPEN) S.ws.send(JSON.stringify({ type: "stop" }));
  $("saveBtn").disabled = !S.recLen;
  cancelAnimationFrame(AN.raf);
  setButtons(); sessionPill();
  setVerdictPillIdle();
}
function setVerdictPillIdle() { const p = $("wavePill"); p.className = "state-pill"; p.textContent = "Stopped"; }

function countdown(sec, statusId) {
  const end = Date.now() + sec * 1000;
  const tick = () => {
    const left = Math.ceil((end - Date.now()) / 1000);
    $(statusId).textContent = left > 0 ? `Keep speaking normally… ${left} s left (${S.calN} windows analysed)` : "Finishing…";
    if (left <= 0) { clearInterval(S.calTimer); stop(); }
  };
  tick();
  S.calTimer = setInterval(tick, 500);
}

// ====================================================================== server messages
function onMessage(m) {
  switch (m.type) {
    case "ready":
      if (m.mode === "monitor") {
        $("asrInfo").textContent = m.transcript ?
          `Scam phrases highlighted · ${m.language ? { en: "English", hi: "Hindi", te: "Telugu" }[m.language] : "language auto-detected"}` : "Speech-to-text is off on the server";
        setSignals({ voice: 0, identity: m.speaker_check ? 0 : null, scam: 0 });
        addEvent("brand", "live", S.origin === "analysis" ? "Analysis started" : "Monitoring started",
          `profile ${m.profile}${m.auto_threshold ? " (cut-off per audio type)" : ` · cut-off ${fmt(m.threshold)}`}${m.claimed ? ` · caller claims to be ${m.claimed}` : ""}`, 0);
      }
      break;
    case "vad":
      S.clock = m.t; S.speech = m.speech_s;
      $("clock").textContent = `${fmt(m.t, 1)} s`; $("speechS").textContent = `${fmt(m.speech_s, 1)} s`;
      $("speechDot").className = "speech-dot" + (m.speech ? " on" : "");
      LV.wave.setSpeech(m.speech);
      LV.lane.set(S.wins, S.clock);
      sessionPill();
      break;
    case "score": {
      const win = { t: m.t, ws: m.window_score, level: m.level, llr: m.llr, orig: m.llr_original, p: m.p_fake, proc: m.processing,
                    cut: m.window_threshold, risk: m.risk, lat: m.latency_ms };
      S.wins.push(win); S.lat.push(m.latency_ms);
      if (m.processing) S.procSeen[m.processing] = (S.procSeen[m.processing] || 0) + 1;
      S.points.push({ t: m.t, risk: m.risk, s: 100 * m.window_score, llr: m.llr, level: m.level });
      S.level = m.verdict;
      showState(m);
      $("mProb").innerHTML = `${(100 * m.window_score).toFixed(0)}<small>%</small>`;
      $("mWin").innerHTML = `${m.suspicious_recent ?? "–"}<small>/ 5</small>`;
      $("lat").innerHTML = `${fmt(m.latency_ms, 0)}<small>ms</small>`;
      addRow(m, m.level);
      LV.chart.set(S.points, S.clock); LV.lane.set(S.wins, S.clock);
      if (S.origin === "analysis") analysisScore(m, win);
      sessionPill();
      break;
    }
    case "transcript": {
      if ($("transcript").querySelector(".empty")) $("transcript").innerHTML = "";
      const d = document.createElement("div");
      d.className = "utt" + ((m.hits || []).length ? " scam" : "");
      const chips = (m.hits || []).map((h) => `<span class="tag crit">${esc(h.label)}</span>`).join("");
      d.innerHTML = `<span class="ts">${fmt(m.t0, 0)}s</span><div><div class="bubble">${highlight(m.text, m.hits)}</div>` + (chips ? `<div class="tactics">${chips}</div>` : "") + `</div>`;
      $("transcript").appendChild(d);
      $("transcript").scrollTop = $("transcript").scrollHeight;
      for (const h of m.hits || []) {                     // each tactic once per session in the feed
        if (S.tactics.has(h.label)) continue;
        S.tactics.add(h.label); addEvent("crit", "message", "Scam tactic heard", h.label, m.t1);
      }
      S.level = m.verdict;
      showState(m); sessionPill();
      break;
    }
    case "calib_window":
      S.calN = m.n;
      $("lat").innerHTML = `${fmt(m.latency_ms, 0)}<small>ms</small>`;
      addRow(m, "");
      S.wins.push({ t: m.t, ws: 0, llr: m.llr, p: m.p_fake, proc: m.processing });
      LV.lane.set(S.wins, S.clock);
      break;
    case "enroll_window":
      S.calN = m.n;
      break;
    case "action": {
      const hold = m.level === "HOLD";
      $("action").className = "alert " + m.level;
      $("action").innerHTML = `${I(hold ? "octagon" : "alert", "lg")}<div><b>${hold ? "Hold" : "Verify"} · ${fmt(m.t, 1)} s</b>` +
        `<p>${esc(capital((m.reasons || []).join("; ")))}. ${esc(m.message)}</p></div>`;
      addEvent(hold ? "crit" : "warn", hold ? "octagon" : "alert", hold ? "HOLD · transactions held" : "VERIFY · out-of-band check", capital((m.reasons || []).join("; ")), m.t);
      VSUI.toast(hold ? "crit" : "warn", hold ? `HOLD at ${fmt(m.t, 1)} s` : `VERIFY at ${fmt(m.t, 1)} s`, capital((m.reasons || [])[0] || ""));
      if (S.origin === "analysis") AN.first[m.level] = AN.first[m.level] ?? m.t;
      break;
    }
    case "summary":
      setVerdict(m.verdict, { risk: m.max_risk, verdict: m.verdict, reasons: m.verdict === "PASS" ? [] : m.reasons, state: `Finished · ${LEVEL_WORD[m.verdict] || m.verdict}` });
      if (m.signals) setSignals(m.signals);
      renderResponse(m.verdict);
      addEvent(m.verdict === "HOLD" ? "crit" : m.verdict === "VERIFY" ? "warn" : "good", LEVEL_ICON[m.verdict] || "check",
        `Session ended · ${m.verdict}`, `max risk ${fmt(m.max_risk, 0)} · ${m.windows} windows · ${fmt(m.speech_s, 1)} s of speech`, m.audio_s);
      if (S.origin === "analysis") analysisSummary(m);
      setVerdictPillIdle();
      break;
    case "calibrated":
      setVerdict("PASS", { state: "Saved", title: `Line profile "${m.profile}" saved`, why: `Cut-off ${fmt(m.llr_threshold)} from ${m.n_windows} windows of your voice. It is now selected.` });
      $("calStatus").textContent = `Saved profile "${m.profile}": cut-off ${fmt(m.llr_threshold)} from ${m.n_windows} windows.`;
      VSUI.toast("good", `Profile "${m.profile}" saved`, `Cut-off ${fmt(m.llr_threshold)}`);
      api("/api/status").then((st) => { S.cfg = st; fillProfiles(st.profiles, m.profile); VSUI.loadStatus(); renderSettings(); });
      break;
    case "enrolled":
      setVerdict("PASS", { state: "Saved", title: `Voiceprint of ${m.name} saved`, why: `${fmt(m.seconds, 0)} s of speech. Selected under "Caller claims to be".` });
      $("vpStatus").textContent = `Saved the voiceprint of ${m.name}.`;
      VSUI.toast("good", "Voiceprint enrolled", m.name);
      $("vpName").value = "";
      loadVoiceprints(m.name).then(() => { store.set("vs-claimed", m.name); VSUI.loadStatus(); });
      break;
    case "warning":
      addEvent("warn", "alert", "Analysis warning", m.message, m.t);
      VSUI.toast("warn", "Analysis warning", m.message);
      break;
    case "error":
      $(S.mode === "enroll" ? "vpStatus" : "calStatus").textContent = m.message;
      setVerdict("IDLE", { state: "Error", title: "Not saved", why: m.message });
      VSUI.toast("crit", "Not saved", m.message);
      break;
  }
}
function showState(m) {
  setVerdict(m.level, { risk: m.risk, verdict: m.verdict, reasons: m.level === "PASS" ? [] : m.reasons });
  if (m.signals) setSignals(m.signals);
  renderResponse(m.verdict);
}

// ====================================================================== voice analysis view
function bindAnalysis() {
  $("aSample").onchange = async () => {
    const s = SAMPLES[Number($("aSample").value)];
    if (!s) return;
    if (s.profile && S.cfg && S.cfg.profiles[s.profile]) $("aProfile").value = s.profile;
    await loadAnalysisAudio(() => fetch("/samples/" + encodeURIComponent(s.file)).then((r) => r.arrayBuffer()), s.label, s.description || s.file);
  };
  $("aFile").onchange = async (e) => {
    const f = e.target.files[0];
    if (!f) return;
    $("aFileLbl").textContent = f.name; $("aSample").value = "";
    await loadAnalysisAudio(() => f.arrayBuffer(), f.name, `${(f.size / 1024).toFixed(0)} KB · uploaded`);
    e.target.value = "";
  };
  $("aPlay").onclick = () => {
    if (!AN.buffer) return;
    start({ kind: "buffer", buffer: AN.buffer, label: AN.name, origin: "analysis", profile: $("aProfile").value });
  };
  $("aStop").onclick = () => stop();
}

async function loadAnalysisAudio(getBytes, name, meta) {
  if (S.running && S.origin === "analysis") await stop();
  $("aName").textContent = name; $("aMeta").textContent = "Decoding…"; $("aPlay").disabled = true;
  try {
    const ctx = await ensureCtx(false);
    const audio = await ctx.decodeAudioData(await getBytes());
    AN.buffer = audio; AN.name = name; AN.dur = audio.duration;
    const ch = audio.getChannelData(0);
    AN.samples = ch;
    const step = audio.duration > 120 ? Math.ceil(audio.duration / 120) : 1;      // keep the spectrogram light for long files
    AN.spec = VSUI.spectrogram(step > 1 ? ch.filter((_, i) => i % step === 0) : ch, audio.sampleRate / step);
    $("aMeta").textContent = `${meta} · ${audio.duration.toFixed(1)} s · ${audio.sampleRate} Hz mono`;
    $("aDur").textContent = fmtClock(audio.duration); $("aTime").textContent = fmtClock(0);
    AN.lane.duration(audio.duration);
    resetAnalysisResults();
    redrawAnalysis();
    setButtons();
  } catch (e) {
    $("aMeta").textContent = "Could not decode this file"; VSUI.toast("crit", "Could not open the recording", e.message || String(e));
  }
}
function resetAnalysisResults() {
  AN.first = {};
  $("aLog").querySelector("tbody").innerHTML = "";
  $("aEmpty").classList.remove("hidden");
  $("aReasons").textContent = ""; $("aKv").innerHTML = "";
  $("aStatus").textContent = AN.buffer ? "Ready · press Analyse & play" : "Waiting for a recording";
  $("aScale").querySelectorAll("div").forEach((d) => d.classList.remove("on"));
  AN.lane.set([], 0);
}
function analysisRegions() {
  return S.origin !== "analysis" ? [] : S.wins.filter((w) => w.ws >= 0.5).map((w) => ({
    a: Math.max(0, w.t - 1) / AN.dur, b: w.t / AN.dur, color: w.level === "HOLD" ? "rgba(244,63,94,.18)" : "rgba(245,158,11,.16)" }));
}
function redrawAnalysis() {
  if (!AN.buffer || $("v-analysis").classList.contains("hidden")) return;
  VSUI.drawOverview($("aWave"), AN.samples, 112, analysisRegions());
  VSUI.drawImageFit($("aSpec"), AN.spec.image, 150);
  AN.lane.draw();
}
function playhead() {
  cancelAnimationFrame(AN.raf);
  const tick = () => {
    if (!(S.running && S.origin === "analysis")) { $("aHead").classList.add("hidden"); $("aHead2").classList.add("hidden"); return; }
    const t = clamp(S.ctx.currentTime - AN.t0, 0, AN.dur), pct = `${100 * t / AN.dur}%`;
    for (const id of ["aHead", "aHead2"]) { $(id).classList.remove("hidden"); $(id).style.left = pct; }
    $("aTime").textContent = fmtClock(t);
    AN.raf = requestAnimationFrame(tick);
  };
  AN.raf = requestAnimationFrame(tick);
}
function setScale(level) {
  $("aScale").querySelectorAll("div").forEach((d) => d.classList.toggle("on", d.classList.contains(level)));
}
function analysisScore(m, win) {
  $("aEmpty").classList.add("hidden");
  AN.lane.set(S.wins, 0);
  $("aStatus").textContent = `Analysing · ${S.wins.length} windows scored`;
  setScale(m.verdict);
  const tr = document.createElement("tr");
  tr.innerHTML = `<td>${fmt(m.t, 1)}</td><td class="ink">${fmt(m.llr)}</td><td>${fmt(m.p_fake, 4)}</td><td>${fmt(100 * m.window_score, 0)}%</td>` +
    `<td>${m.processing ? esc(m.processing) : "–"}</td><td class="r">${fmt(m.risk, 0)}</td><td>${vbadge(m.level)}</td>`;
  $("aLog").querySelector("tbody").prepend(tr);
  if (!$("v-analysis").classList.contains("hidden")) VSUI.drawOverview($("aWave"), AN.samples, 112, analysisRegions());
  void win;
}
function analysisSummary(m) {
  setScale(m.verdict);
  const w = S.wins, flagged = w.filter((x) => x.ws >= 0.5).length;
  const lat = [...S.lat].sort((a, b) => a - b), med = lat.length ? lat[Math.floor(lat.length / 2)] : null;
  const mean = w.length ? w.reduce((s, x) => s + x.llr, 0) / w.length : null;
  const types = Object.entries(S.procSeen).sort((a, b) => b[1] - a[1]).map(([k, n]) => `${k} (${n})`).join(", ");
  $("aStatus").textContent = `Finished · ${LEVEL_WORD[m.verdict]} (${m.verdict})`;
  $("aReasons").textContent = m.verdict === "PASS" ? "No window pattern, voiceprint or scam language reached an alert level." : capital((m.reasons || []).join("; ")) + ".";
  const row = (k, v) => `<dt>${k}</dt><dd>${v}</dd>`;
  $("aKv").innerHTML = row("Verdict", vbadge(m.verdict)) + row("Highest risk", `${fmt(m.max_risk, 0)} / 100`) +
    row("Windows flagged", `${flagged} of ${w.length}`) +
    row("Peak window probability", w.length ? `${(100 * Math.max(...w.map((x) => x.ws))).toFixed(0)}%` : "–") +
    row("First VERIFY", AN.first.VERIFY !== undefined ? `${fmt(AN.first.VERIFY, 1)} s` : "–") +
    row("HOLD at", AN.first.HOLD !== undefined ? `${fmt(AN.first.HOLD, 1)} s` : "–") +
    row("Mean detector score", fmt(mean)) + row("Line profile", esc(S.profile)) +
    (types ? row("Audio type seen", esc(types)) : "") + row("Median latency", med === null ? "–" : `${fmt(med, 0)} ms`);
}

// ====================================================================== history view
const HIST = { rows: [], page: 0, per: 25, sort: { k: "id", dir: -1 }, verdict: "", mode: "", q: "", stale: true };
function bindHistory() {
  $("hRefresh").onclick = () => { HIST.stale = true; loadHistory(); };
  $("hSearch").oninput = () => { HIST.q = $("hSearch").value.trim().toLowerCase(); HIST.page = 0; renderHistory(); };
  $("hMode").onchange = () => { HIST.mode = $("hMode").value; HIST.page = 0; renderHistory(); };
  $("hVerdict").querySelectorAll("button").forEach((b) => b.onclick = () => {
    $("hVerdict").querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    HIST.verdict = b.dataset.v; HIST.page = 0; renderHistory();
  });
  $("hTable").querySelectorAll("th.sortable").forEach((th) => th.onclick = () => {
    const k = th.dataset.k;
    HIST.sort = { k, dir: HIST.sort.k === k ? -HIST.sort.dir : -1 }; renderHistory();
  });
  $("hPrev").onclick = () => { HIST.page = Math.max(0, HIST.page - 1); renderHistory(); };
  $("hNext").onclick = () => { HIST.page += 1; renderHistory(); };
}
async function loadHistory() {
  if (!HIST.stale && HIST.rows.length) { renderHistory(); return; }
  $("hTable").querySelector("tbody").innerHTML = Array.from({ length: 6 }, () => `<tr><td colspan="10"><div class="skeleton" style="height:18px"></div></td></tr>`).join("");
  try { HIST.rows = await api("/api/calls?limit=200"); HIST.stale = false; } catch (e) { HIST.rows = []; VSUI.toast("crit", "Could not load history", e.message); }
  renderHistory();
}
function renderHistory() {
  let rows = HIST.rows.filter((r) =>
    (!HIST.verdict || r.verdict === HIST.verdict) && (!HIST.mode || r.mode === HIST.mode) &&
    (!HIST.q || `#${r.id} ${r.source} ${r.profile}`.toLowerCase().includes(HIST.q)));
  const { k, dir } = HIST.sort;
  rows.sort((a, b) => ((a[k] ?? -1) > (b[k] ?? -1) ? 1 : (a[k] ?? -1) < (b[k] ?? -1) ? -1 : 0) * dir);
  $("hTable").querySelectorAll("th.sortable .arr").forEach((s) => { s.textContent = s.parentElement.dataset.k === k ? (dir < 0 ? "↓" : "↑") : ""; });
  const pages = Math.max(1, Math.ceil(rows.length / HIST.per));
  HIST.page = Math.min(HIST.page, pages - 1);
  const slice = rows.slice(HIST.page * HIST.per, (HIST.page + 1) * HIST.per);
  $("hCount").textContent = `${rows.length} of ${HIST.rows.length} sessions`;
  $("hPageInfo").textContent = rows.length ? `Page ${HIST.page + 1} of ${pages}` : "";
  $("hPrev").disabled = HIST.page === 0; $("hNext").disabled = HIST.page >= pages - 1;
  $("hEmpty").classList.toggle("hidden", rows.length > 0);
  const modeTag = { monitor: '<span class="tag brand">monitoring</span>', calibrate: '<span class="tag">calibration</span>', enroll: '<span class="tag">enrolment</span>' };
  $("hTable").querySelector("tbody").innerHTML = slice.map((r) => {
    const v = r.mode === "monitor" ? r.verdict : r.verdict;
    return `<tr class="click" tabindex="0" data-id="${r.id}"><td class="mono ink">#${r.id}</td><td>${r.started ? fmtDate(r.started) : "–"}</td>` +
      `<td class="wrap" style="max-width:320px"><span class="ink" title="${esc(r.source)}">${esc((r.source || "").slice(0, 70))}</span>` +
      `<div class="xs muted">profile ${esc(r.profile)} · cut-off ${fmt(r.threshold)}</div></td>` +
      `<td>${modeTag[r.mode] || esc(r.mode)}</td><td>${["PASS", "VERIFY", "HOLD"].includes(v) ? vbadge(v) : `<span class="vbadge IDLE">${esc(v)}</span>`}</td>` +
      `<td>${r.mode === "monitor" ? riskbar(r.max_risk) : '<span class="muted">–</span>'}</td><td class="r">${fmt(r.speech_s, 1)} s</td>` +
      `<td class="r">${r.windows ?? "–"}</td><td class="r">${fmt(r.mean_llr)}</td><td class="muted">${I("chevron", "sm")}</td></tr>`;
  }).join("");
  $("hTable").querySelectorAll("tr.click").forEach((tr) => {
    tr.onclick = () => { location.hash = `#/call/${tr.dataset.id}`; };
    tr.onkeydown = (e) => { if (e.key === "Enter") location.hash = `#/call/${tr.dataset.id}`; };
  });
}

// ====================================================================== threat details view
let CALLCHART = null;
const ACTION = { TRANSFER_REQUESTED: ["", "Payment requested by the caller", "send"], TRANSFER_APPROVED: ["good", "Payment approved", "check"],
  TRANSFER_BLOCKED: ["crit", "Payment blocked automatically", "lock"], TRANSFER_DECLINED: ["", "Payment declined by the customer", "x"],
  OTP_SENT: ["warn", "OTP sent to the registered phone", "smartphone"], OTP_VERIFIED: ["good", "OTP verified", "check"],
  OTP_FAILED: ["crit", "OTP failed", "x"], CALLBACK_REQUESTED: ["warn", "Call-back on the official number", "phone"], CALL_ENDED: ["", "Call ended", "phone-off"] };
async function loadCall(id) {
  $("h-call").textContent = "Loading…"; $("cSub").textContent = ""; $("cKpis").innerHTML = Array.from({ length: 4 }, () => `<div class="skeleton" style="height:76px"></div>`).join("");
  let c;
  try { c = await api(`/api/calls/${encodeURIComponent(id)}`); } catch (e) {
    $("h-call").textContent = "Session not found"; $("cSub").textContent = `There is no session #${id} in the audit log.`; $("cKpis").innerHTML = ""; return;
  }
  const W = c.window_log || [], E = c.events || [], monitor = c.mode === "monitor";
  const flagged = W.filter((w) => (w.window_score ?? 0) >= 0.5).length;
  const firstOf = (k) => { const e = E.find((x) => x.kind === k); return e ? e.t : null; };
  const head = !monitor ? (c.mode === "calibrate" ? "Calibration session" : "Voiceprint enrolment") :
    c.verdict === "HOLD" ? "Likely cloned voice — call held" : c.verdict === "VERIFY" ? "Suspicious call — verification required" : "No threat found";
  $("h-call").textContent = head;
  $("cEyebrow").innerHTML = `${I("shield-alert", "xs")}Investigation · session #${c.id}`;
  const dur = c.audio_s ? `${fmt(c.audio_s, 1)} s audio, ${fmt(c.speech_s, 1)} s speech` : "duration not recorded";
  $("cSub").textContent = `${c.source} · ${c.started ? fmtDate(c.started) : ""} · ${dur}`;
  $("cBadge").innerHTML = monitor ? `<span style="transform:scale(1.25);display:inline-block;transform-origin:right">${vbadge(c.verdict)}</span>` : "";
  const kpi = (ic, cls, label, big, sub) => `<div class="kpi"><span class="ki ${cls}">${I(ic, "lg")}</span><div><div class="label">${label}</div><div class="big">${big}</div>${sub ? `<div class="sub">${sub}</div>` : ""}</div></div>`;
  $("cKpis").innerHTML =
    kpi("gauge", c.verdict === "HOLD" ? "crit" : c.verdict === "VERIFY" ? "warn" : "good", "Highest risk", monitor ? `${fmt(c.max_risk, 0)}<small>/ 100</small>` : "–", "noisy-OR of voice, identity and scam signals") +
    kpi("layers", flagged ? "warn" : "", "Windows flagged", `${flagged}<small>of ${W.length}</small>`, "window probability ≥ 50% on this line") +
    kpi("clock", firstOf("HOLD") !== null ? "crit" : firstOf("VERIFY") !== null ? "warn" : "", "First alert",
      firstOf("VERIFY") !== null || firstOf("HOLD") !== null ? `${fmt(Math.min(...["VERIFY", "HOLD"].map(firstOf).filter((x) => x !== null)), 1)}<small>s</small>` : "–",
      firstOf("HOLD") !== null ? `HOLD at ${fmt(firstOf("HOLD"), 1)} s` : "no HOLD") +
    kpi("sliders", "brand", "Line profile", esc(c.profile), `cut-off ${fmt(c.threshold)} on the detector score`);
  // 1. model prediction
  scoreChart($("cScore"), W, c.threshold);
  const pmax = W.length ? Math.max(...W.map((w) => w.p_fake || 0)) : null;
  const sorted = W.map((w) => w.llr).sort((a, b) => a - b), med = sorted.length ? sorted[Math.floor(sorted.length / 2)] : null;
  const line = (k, v) => `<div class="stat-line"><span>${k}</span><b>${v}</b></div>`;
  $("cModel").innerHTML = line("Median detector score", fmt(med)) + line("Highest p(fake)", pmax === null ? "–" : fmt(pmax, 4)) +
    line("Windows above the cut-off", `${flagged} / ${W.length}`) +
    `<p class="xs muted" style="margin-top:8px">The detector scores each 4 s window independently. The decision uses the running average and how many of the last 5 windows agree.</p>`;
  // 2. supporting signals
  const sims = W.map((w) => w.similarity).filter((x) => x !== null && x !== undefined);
  const claimed = (c.source.match(/claims: (.+)$/) || [])[1];
  const tactics = E.filter((e) => e.kind === "SCAM").flatMap((e) => e.detail.split("; "));
  const sMean = sims.length ? sims.reduce((a, b) => a + b, 0) / sims.length : null;
  $("cSignals").innerHTML =
    `<div class="section-label">Voiceprint</div>` +
    (claimed && sims.length ? `<div class="stat-line"><span>Claimed identity</span><b>${esc(claimed)}</b></div>` +
      `<div class="stat-line"><span>Mean similarity</span><b class="${sMean < 0.45 ? "crit-ink" : "good-ink"}">${fmt(sMean, 2)}</b></div>` +
      `<div class="stat-line"><span>Lowest similarity</span><b>${fmt(Math.min(...sims), 2)}</b></div>` +
      `<p class="xs muted" style="margin-top:6px">${sMean < 0.45 ? `${I("x", "xs")} Does not match the enrolled voice (cut-off 0.45).` : `${I("check", "xs")} Matches the enrolled voice. A match never lowers the risk.`}</p>`
      : `<p class="small muted" style="margin:6px 0 4px">No identity was claimed in this session.</p>`) +
    `<div class="section-label" style="margin-top:16px">Scam language</div>` +
    (tactics.length ? `<div style="margin-top:6px">${tactics.map((t) => `<span class="tag crit">${I("message", "xs")}${esc(t)}</span>`).join("")}</div>` +
      `<p class="xs muted" style="margin-top:6px">Tactic names only; the transcript itself is never stored.</p>`
      : `<p class="small muted" style="margin-top:6px">No scam tactics recognised.</p>`);
  // 3. actions
  const items = E.map((e) => {
    if (e.kind === "VERIFY" || e.kind === "HOLD") return [e.kind === "HOLD" ? "crit" : "warn", e.kind === "HOLD" ? "octagon" : "alert",
      `Decision · ${e.kind}`, e.detail.split(". ")[0].replace(/^(VERIFY|HOLD): /, ""), e.t];
    if (e.kind === "SCAM") return ["", "message", "Signal · scam tactic", e.detail, e.t];
    const a = ACTION[e.kind]; if (a) return [a[0] || "brand", a[2], `Action · ${a[1]}`, e.detail, e.t];
    return ["", "info", e.kind, e.detail, e.t];
  });
  $("cActions").innerHTML = items.length ? items.map(([cls, ic, t, d, at]) =>
    `<li class="${cls}"><span class="node">${I(ic)}</span><div class="what"><b>${esc(t)}</b>${d ? `<p>${esc(capital(d))}</p>` : ""}</div><time>${fmt(at, 1)}s</time></li>`).join("")
    : `<div class="empty" style="padding:16px">${I("shield-check")}<b>No action needed</b>No alert was raised in this session.</div>`;
  // risk chart + log
  if (!CALLCHART) CALLCHART = new VSUI.RiskChart($("cChart"), { height: 200, verifyAt: S.cfg?.risk?.verify_at ?? 45, holdAt: S.cfg?.risk?.hold_at ?? 75 });
  CALLCHART.set(W.filter((w) => w.risk !== null).map((w) => ({ t: w.t, risk: w.risk, s: 100 * (w.window_score ?? 0), llr: w.llr, level: w.level })), c.audio_s || 0);
  $("cLog").querySelector("tbody").innerHTML = W.map((w) =>
    `<tr><td>${fmt(w.t, 1)}</td><td class="ink">${fmt(w.llr)}</td><td>${fmt(w.p_fake, 4)}</td><td>${w.window_score === null ? "–" : fmt(100 * w.window_score, 0) + "%"}</td>` +
    `<td>${w.similarity === null ? "–" : fmt(w.similarity, 2)}</td><td class="r">${w.risk === null ? "–" : fmt(w.risk, 0)}</td>` +
    `<td>${w.level ? vbadge(w.level) : '<span class="muted">–</span>'}</td><td class="r">${fmt(w.latency_ms, 0)}</td></tr>`).join("") ||
    `<tr><td colspan="8" class="muted">No windows were analysed.</td></tr>`;
}

// detector score per window against the line cut-off (SVG, dots carry their own tooltips)
function scoreChart(el, W, cut) {
  if (!W.length) { el.innerHTML = `<div class="empty" style="padding:16px">${I("layers")}No windows.</div>`; return; }
  const Wd = el.clientWidth || 360, H = 170, pad = { l: 30, r: 8, t: 10, b: 22 };
  const lo = Math.min(-12, ...W.map((w) => w.llr), cut ?? 0) - 1, hi = Math.max(6, ...W.map((w) => w.llr), cut ?? 0) + 1;
  const tmax = Math.max(...W.map((w) => w.t), 5);
  const X = (t) => pad.l + (Wd - pad.l - pad.r) * t / tmax, Y = (v) => pad.t + (H - pad.t - pad.b) * (hi - v) / (hi - lo);
  let s = `<svg class="chart" viewBox="0 0 ${Wd} ${H}" height="${H}" role="img" aria-label="Detector score per window"><g class="grid">`;
  for (const v of [Math.ceil(lo / 5) * 5, 0, Math.floor(hi / 5) * 5].filter((v, i, a) => a.indexOf(v) === i))
    s += `<line x1="${pad.l}" x2="${Wd - pad.r}" y1="${Y(v)}" y2="${Y(v)}"/><text x="${pad.l - 6}" y="${Y(v) + 4}" text-anchor="end">${v}</text>`;
  s += `</g>`;
  if (cut !== null && cut !== undefined) s += `<line x1="${pad.l}" x2="${Wd - pad.r}" y1="${Y(cut)}" y2="${Y(cut)}" stroke="${VSUI.css("--warn")}" stroke-dasharray="4 4" opacity=".8"/>` +
    `<text x="${Wd - pad.r}" y="${Y(cut) - 5}" text-anchor="end" style="fill:${VSUI.css("--warn-ink")}">cut-off ${fmt(cut)}</text>`;
  s += `<polyline fill="none" stroke="${VSUI.css("--series-1")}" stroke-width="2" stroke-linejoin="round" points="${W.map((w) => `${X(w.t)},${Y(w.llr)}`).join(" ")}"/>`;
  for (const w of W) {
    const bad = (w.window_score ?? 0) >= 0.5;
    s += `<circle cx="${X(w.t)}" cy="${Y(w.llr)}" r="4" fill="${bad ? VSUI.css("--chart-warn") : VSUI.css("--series-1")}" stroke="${VSUI.css("--chart-surface")}" stroke-width="2">` +
      `<title>${fmt(w.t, 1)} s · score ${fmt(w.llr)} · p(fake) ${fmt(w.p_fake, 3)}${bad ? " · suspicious" : ""}</title></circle>`;
  }
  s += `<text x="${pad.l}" y="${H - 4}">0 s</text><text x="${Wd - pad.r}" y="${H - 4}" text-anchor="end">${fmt(tmax, 0)} s</text></svg>`;
  el.innerHTML = s + `<div class="spark-legend" style="margin:6px 0 4px"><span><span class="sw" style="display:inline-block;width:14px;height:2px;background:var(--series-1);vertical-align:middle;margin-right:6px"></span>detector score</span>` +
    `<span><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--chart-warn);margin-right:6px"></span>above the cut-off</span></div>`;
}

// ====================================================================== analytics view
async function loadAnalytics() {
  let calls = [], inc = [];
  try { [calls, inc] = await Promise.all([api("/api/calls?limit=200"), api("/api/incidents?limit=200")]); }
  catch (e) { VSUI.toast("crit", "Could not load analytics", e.message); return; }
  const all = $("anAll").checked;
  const keep = (r) => r.mode === "monitor" && (all || (r.source || "").startsWith("call:"));
  const mon = calls.filter(keep), incs = inc.filter(keep);
  const n = mon.length, by = (v) => mon.filter((r) => r.verdict === v).length;
  const firstAlert = incs.map((r) => { const e = r.events.find((x) => x.kind === "VERIFY" || x.kind === "HOLD"); return e ? e.t : null; }).filter((x) => x !== null).sort((a, b) => a - b);
  const pct = (k) => (n ? ` · ${Math.round(100 * k / n)}%` : "");
  const kpi = (ic, cls, label, big, sub) => `<div class="kpi"><span class="ki ${cls}">${I(ic, "lg")}</span><div><div class="label">${label}</div><div class="big">${big}</div><div class="sub">${sub}</div></div></div>`;
  $("anKpis").innerHTML =
    kpi("activity", "brand", "Sessions monitored", n, all ? "calls and lab-console tests" : "calls through the demo app") +
    kpi("octagon", "crit", "Held · likely spoofed", by("HOLD"), `HOLD${pct(by("HOLD"))}`) +
    kpi("alert", "warn", "Suspicious", by("VERIFY"), `VERIFY${pct(by("VERIFY"))}`) +
    kpi("clock", "", "Median time to first alert", firstAlert.length ? `${fmt(firstAlert[Math.floor(firstAlert.length / 2)], 1)}<small>s</small>` : "–",
      firstAlert.length ? `over ${firstAlert.length} alerted sessions` : "no alerts yet");
  // outcomes by day
  const days = new Map();
  for (const r of [...mon].reverse()) {
    const d = new Date(r.started * 1000), key = d.toLocaleDateString(undefined, { day: "2-digit", month: "short" });
    if (!days.has(key)) days.set(key, { label: key, values: { PASS: 0, VERIFY: 0, HOLD: 0 } });
    days.get(key).values[r.verdict] = (days.get(key).values[r.verdict] || 0) + 1;
  }
  const ser = [{ key: "PASS", name: "Pass", color: VSUI.css("--chart-good") }, { key: "VERIFY", name: "Verify", color: VSUI.css("--chart-warn") }, { key: "HOLD", name: "Hold", color: VSUI.css("--chart-crit") }];
  const dayEl = $("anDays"); dayEl.classList.remove("skeleton"); dayEl.style.height = "";
  if (days.size) VSUI.barChart(dayEl, [...days.values()].slice(-14), ser, { height: 220, label: "Outcomes by day" });
  else dayEl.innerHTML = `<div class="empty">${I("chart")}<b>No monitored sessions yet</b></div>`;
  // risk distribution
  const bins = Array.from({ length: 10 }, (_, i) => ({ label: `${i * 10}`, tip: `Max risk ${i * 10}–${i * 10 + (i === 9 ? 10 : 9)}`, values: { n: 0 } }));
  mon.forEach((r) => { bins[Math.min(9, Math.floor((r.max_risk || 0) / 10))].values.n += 1; });
  const rEl = $("anRisk"); rEl.classList.remove("skeleton"); rEl.style.height = "";
  if (n) VSUI.barChart(rEl, bins, [{ key: "n", name: "Sessions", color: VSUI.css("--series-1") }], { height: 220, label: "Risk distribution" });
  else rEl.innerHTML = `<div class="empty">${I("gauge")}<b>No data yet</b></div>`;
  // tactics + actions
  const tally = (arr) => { const m = new Map(); arr.forEach((x) => m.set(x, (m.get(x) || 0) + 1)); return [...m.entries()].sort((a, b) => b[1] - a[1]); };
  barList($("anTactics"), tally(incs.flatMap((r) => r.events.filter((e) => e.kind === "SCAM").flatMap((e) => e.detail.split("; ")))), "var(--chart-crit)", "No scam tactics heard yet.");
  barList($("anActions"), tally(incs.flatMap((r) => r.events.filter((e) => ACTION[e.kind]).map((e) => ACTION[e.kind][1]))), "var(--series-2)", "No prevention actions recorded yet. They come from the customer phone.");
  renderModelCard();
  // latency from the stored window logs of recent sessions
  const recent = mon.filter((r) => r.windows > 0).slice(0, 16);
  const details = await Promise.all(recent.map((r) => api(`/api/calls/${r.id}`).catch(() => null)));
  const lats = [], per = [];
  details.forEach((d) => { if (!d) return; const l = d.window_log.map((w) => w.latency_ms).filter((x) => x).sort((a, b) => a - b); if (!l.length) return; lats.push(...l); per.push({ label: `#${d.id}`, tip: `Session #${d.id}`, values: { ms: Math.round(l[Math.floor(l.length / 2)]) } }); });
  const lEl = $("anLat"); lEl.classList.remove("skeleton"); lEl.style.height = "";
  if (per.length) {
    VSUI.barChart(lEl, per.reverse(), [{ key: "ms", name: "Median ms", color: VSUI.css("--series-2") }], { height: 180, label: "Detector latency", fmt: (v) => `${v} ms` });
    lats.sort((a, b) => a - b);
    $("anLatStats").innerHTML = `<div class="stat-line"><span>Median over ${lats.length} windows</span><b>${Math.round(lats[Math.floor(lats.length / 2)])} ms</b></div>` +
      `<div class="stat-line"><span>95th percentile</span><b>${Math.round(lats[Math.floor(lats.length * 0.95)])} ms</b></div>` +
      `<div class="stat-line"><span>Real-time budget</span><b>1000 ms per 1 s hop</b></div>`;
  } else { lEl.innerHTML = `<div class="empty">${I("cpu")}<b>No windows yet</b></div>`; $("anLatStats").innerHTML = ""; }
}
function barList(el, items, color, empty) {
  if (!items.length) { el.innerHTML = `<div class="empty" style="padding:18px 8px">${I("info")}${esc(empty)}</div>`; return; }
  const max = items[0][1];
  el.innerHTML = items.slice(0, 8).map(([k, v]) => `<div style="margin-bottom:10px"><div class="row" style="justify-content:space-between;font-size:12.5px">` +
    `<span class="ink-2">${esc(k)}</span><b class="tnum">${v}</b></div><div class="level" style="height:6px;margin-top:5px">` +
    `<div style="width:${100 * v / max}%;background:${color}"></div></div></div>`).join("");
}
// Measured offline on Kaggle (Step 8c, 20 Nov 2026): EER % on unseen test sets. Source: docs/SIH26104_Research_and_Build_Plan.md
const MODEL_EVAL = [
  ["Hindi/Telugu clones · clean", 0.0, 0.0], ["… phone line", 2.0, 0.0], ["… noise suppression", 5.0, 1.33], ["… Bluetooth CVSD", 10.0, 4.0],
  ["… Opus (VoIP)", 4.67, 1.33], ["Premium clones (ElevenLabs, Kokoro) · clean", 8.23, 2.95], ["… noise suppression", 33.97, 12.66],
  ["… Opus (VoIP)", 45.15, 16.67], ["In-the-Wild deepfakes", 4.67, 4.67], ["Mean of 13 test sets", 9.73, 4.05]];
function renderModelCard() {
  const max = 50, bar = (v, c) => `<div class="level" style="height:6px;width:100%"><div style="width:${100 * v / max}%;background:${c}"></div></div>`;
  $("anModel").innerHTML = `<div class="legend" style="margin-bottom:12px"><span><span class="sw bar" style="background:var(--series-2)"></span>Original detector</span>` +
    `<span><span class="sw bar" style="background:var(--series-1)"></span>With the Step 8c back-end (in use)</span></div>` +
    `<div class="table-wrap" style="max-height:none"><table class="table"><thead><tr><th>Test set (never used for training)</th><th class="r">Original EER</th><th style="width:28%"></th><th class="r">Step 8c EER</th><th style="width:28%"></th></tr></thead><tbody>` +
    MODEL_EVAL.map(([name, a, b], i) => `<tr${i === MODEL_EVAL.length - 1 ? ' style="font-weight:650"' : ""}><td class="ink">${esc(name)}</td>` +
      `<td class="r">${a.toFixed(1)}%</td><td>${bar(a, "var(--series-2)")}</td><td class="r ink">${b.toFixed(1)}%</td><td>${bar(b, "var(--series-1)")}</td></tr>`).join("") +
    `</tbody></table></div><p class="xs muted" style="margin-top:10px">EER: the error rate where false alarms equal missed fakes. Processed test sets use simulated processing; ` +
    `on real laptop recordings Step 8c turned a premium ElevenLabs clone from VERIFY into HOLD at 5.6 s while genuine callers still passed.</p>`;
}

// ====================================================================== incidents view (security operations)
const INC = { seen: null, timer: null, ICON: { PASS: "shield-check", VERIFY: "alert", HOLD: "octagon" } };
async function refreshIncidents() {
  let rows;
  try { rows = await api("/api/incidents?limit=100"); } catch (e) { $("iUpdated").textContent = "Server offline"; return; }
  if (!$("iAll").checked) rows = rows.filter((r) => (r.source || "").startsWith("call:"));
  rows = rows.slice(0, 50);
  const holds = new Set(rows.filter((r) => r.verdict === "HOLD").map((r) => r.id));
  $("kCalls").textContent = rows.length; $("kHold").textContent = holds.size;
  $("kVerify").textContent = rows.filter((r) => r.verdict === "VERIFY").length;
  $("kBlocked").textContent = rows.reduce((n, r) => n + r.events.filter((e) => e.kind === "TRANSFER_BLOCKED").length, 0);
  $("emptyInc").classList.toggle("hidden", rows.length > 0);
  if (INC.seen) for (const id of holds) if (!INC.seen.has(id)) VSUI.toast("crit", `New HOLD · call #${id}`, "Likely cloned voice or scam. Payments on this call are blocked.");
  $("inc").querySelector("tbody").innerHTML = rows.map((r) => {
    const alerts = r.events.filter((e) => e.kind === "VERIFY" || e.kind === "HOLD").map((e) =>
      `<div><span class="tag ${e.kind === "HOLD" ? "crit" : "warn"}">${e.kind} · ${fmt(e.t, 1)} s</span> <span class="small ink-2">${esc(e.detail.split(". ")[0].replace(/^(VERIFY|HOLD): /, ""))}</span></div>`).join("");
    const scams = r.events.filter((e) => e.kind === "SCAM").flatMap((e) => e.detail.split("; ")).map((d) => `<span class="tag">${esc(d)}</span>`).join("");
    const acts = r.events.filter((e) => ACTION[e.kind]).map((e) => `<span class="tag ${ACTION[e.kind][0]}" title="${esc(e.detail)}">${esc(ACTION[e.kind][1])}</span>`).join("");
    const live = r.ended ? "" : ' <span class="chip live" style="height:20px"><span class="dot"></span>live</span>';
    const fresh = INC.seen && r.verdict === "HOLD" && !INC.seen.has(r.id) ? " flash" : "";
    const v = r.verdict || "PASS";
    return `<tr class="click${fresh}" tabindex="0" data-id="${r.id}"><td>${vbadge(v)}</td>` +
      `<td><b>#${r.id}</b><div class="xs muted">${new Date(r.started * 1000).toLocaleTimeString()}</div></td>` +
      `<td class="wrap" style="min-width:110px"><span class="ink">${esc(r.source)}</span>${live}</td><td>${riskbar(r.max_risk)}</td>` +
      `<td class="wrap" style="min-width:230px">${alerts || '<span class="muted">–</span>'}</td>` +
      `<td class="wrap" style="min-width:170px">${scams || '<span class="muted">–</span>'}</td>` +
      `<td class="wrap" style="min-width:130px">${acts || '<span class="muted">–</span>'}</td></tr>`;
  }).join("");
  $("inc").querySelectorAll("tr.click").forEach((tr) => {
    tr.onclick = () => { location.hash = `#/call/${tr.dataset.id}`; };
    tr.onkeydown = (e) => { if (e.key === "Enter") location.hash = `#/call/${tr.dataset.id}`; };
  });
  INC.seen = holds;
  const maxHold = Math.max(0, ...holds); store.set("vs-inc-seen", String(maxHold)); pollBadge();
  $("iUpdated").textContent = "Live · " + new Date().toLocaleTimeString();
}
async function pollBadge() {
  const b = $("incBadge"); if (!b) return;
  try {
    const rows = await api("/api/incidents?limit=30"), seen = Number(store.get("vs-inc-seen") || 0);
    const n = rows.filter((r) => r.verdict === "HOLD" && r.id > seen && (store.get("vs-sec-all") === "1" || (r.source || "").startsWith("call:"))).length;
    b.textContent = String(n); b.classList.toggle("hidden", n === 0 || CURRENT === "incidents");
  } catch (e) { b.classList.add("hidden"); }
}

// ====================================================================== settings view
function bindSettings() {
  $("calBtn").onclick = () => start({ kind: "mic", calibrate: true, origin: "settings" });
  $("enrollBtn").onclick = () => {
    const name = $("vpName").value.trim();
    if (!name) { $("vpStatus").textContent = "Type a name first."; $("vpName").focus(); return; }
    start({ kind: "mic", enroll: name, origin: "settings" });
  };
}
async function renderSettings() {
  let st = S.cfg;
  try { st = await api("/api/status"); S.cfg = st; } catch (e) { /* keep last */ }
  if (!st) return;
  $("profTable").querySelector("tbody").innerHTML = Object.entries(st.profiles).map(([n, p]) =>
    `<tr><td><b>${esc(n)}</b></td><td class="r ink">${p.auto ? "per type" : fmt(p.llr_threshold)}</td>` +
    `<td class="wrap small">${esc(p.method || "")}${p.source ? `<div class="xs muted">${esc(p.source)}</div>` : ""}</td></tr>`).join("");
  const r = st.risk, row = (k, v) => `<dt>${k}</dt><dd>${v}</dd>`;
  $("sysKv").innerHTML =
    row("Detector", st.detector_mode === "step8c" ? "NII AntiDeepfake wav2vec 2.0 Large (frozen) + Step 8c back-end" : "NII AntiDeepfake wav2vec 2.0 Large") +
    row("Runs on", esc((st.detector || "").replace("ExecutionProvider", ""))) +
    row("Analysis window", `${st.window_s} s of speech, re-checked every ${st.hop_s} s (first after ${st.min_speech_s} s)`) +
    row("VERIFY when", `voice score ≥ ${r.verify_at}, or 2 of the last ${r.hold_n} windows flagged, or identity ≥ ${r.identity_at}, or scam ≥ ${r.scam_at}`) +
    row("HOLD when", `voice score ≥ ${r.hold_at} with ${r.hold_k} of the last ${r.hold_n} windows flagged, or scam language plus a suspicious voice or identity`) +
    row("Speech-to-text", esc(st.asr || "off")) + row("Voiceprints enrolled", (st.voiceprints || []).length) +
    row("Stored", "scores, decisions, scam-tactic names and prevention events only");
}

init();
