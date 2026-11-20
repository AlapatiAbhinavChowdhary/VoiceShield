"use strict";
// Customer page: answers the call, plays the caller, streams the caller's audio to VoiceShield,
// and enforces the prevention steps in the mock bank app.
const U = { sig: null, pc: null, offer: null, ice: [], remote: null, ctx: null, an: null, session: 0,
            callId: null, mode: "monitor", verdict: "PASS", level: "IDLE", points: [], clock: 0, request: null, otp: null,
            talk: null, speaker: false, balance: 125000, cfg: null, calTimer: null, started: 0, tick: null, chart: null };
const INR = (n) => "₹" + Number(n).toLocaleString("en-IN");
const I = (n, c = "") => VSUI.icon(n, c);

// ------------------------------------------------------------------ boot
async function init() {
  VSUI.mountTopbar("/customer");
  VSUI.hydrate();
  U.chart = new VSUI.RiskChart($("chart"), { height: 190 });
  $("room").value = new URLSearchParams(location.search).get("room") || store.get("vs-room") || "demo";
  $("lang").value = store.get("vs-cust-lang") || "";
  $("lang").onchange = () => store.set("vs-cust-lang", $("lang").value);
  await loadSettings();
  $("joinBtn").onclick = goOnline;
  $("answerBtn").onclick = answer;
  $("rejectBtn").onclick = reject;
  $("endBtn").onclick = () => endCall("You ended the call", true);
  $("speakerBtn").onclick = toggleSpeaker;
  $("talkBtn").onclick = toggleTalk;
  $("calBtn").onclick = calibrate;
  setCall("offline");
  setVerdict("IDLE");
  emptyTranscript();
  renderRequest();
  addActivity("info", "Bank app opened · VoiceShield protection is on");
}

async function loadSettings(chooseProfile) {
  try { U.cfg = await api("/api/status"); } catch (e) { setCall("offline", "VoiceShield server not reachable"); return; }
  U.chart.thresholds(U.cfg.risk.verify_at, U.cfg.risk.hold_at);
  const profs = U.cfg.profiles;
  const want = chooseProfile || store.get("vs-cust-profile") || (profs.call ? "call" : profs.mic ? "mic" : "default");
  $("profile").innerHTML = Object.entries(profs).map(([n, p]) =>
    `<option value="${esc(n)}">${esc(n)} · cut-off ${p.auto ? "per audio type" : fmt(p.llr_threshold)}</option>`).join("");
  $("profile").value = profs[want] ? want : "default";
  $("profile").onchange = () => store.set("vs-cust-profile", $("profile").value);
  const names = U.cfg.voiceprints || [];
  const keep = store.get("vs-cust-claimed") || "";
  $("claimed").innerHTML = '<option value="">Unknown caller</option>' + names.map((n) => `<option value="${esc(n)}">${esc(n)}</option>`).join("");
  if (names.includes(keep)) $("claimed").value = keep;
  $("claimed").onchange = () => { store.set("vs-cust-claimed", $("claimed").value); showWho(); };
  showWho();
}

function showWho() {
  const name = $("claimed").value;
  $("who").textContent = name || "Unknown caller";
  const av = $("avatar");
  av.classList.toggle("known", !!name);
  av.innerHTML = name ? esc(name.split(/\s+/).map((w) => w[0]).slice(0, 2).join("").toUpperCase()) : I("user", "xl");
  $("sigIdS").textContent = name ? `Compared with ${name}'s voiceprint` : "Pick who the caller claims to be";
}

// ------------------------------------------------------------------ call card state
function setCall(state, text) {
  const labels = { offline: "Offline", online: "Online · waiting for a call", ringing: "Incoming call…",
                   incall: "In call", ended: "Call ended" };
  $("status").textContent = text || labels[state];
  $("statusDot").className = "speech-dot" + (state === "incall" || state === "online" ? " on" : "");
  $("offlineCtl").classList.toggle("hidden", state !== "offline");
  $("ringCtl").classList.toggle("hidden", state !== "ringing");
  $("callCtl").classList.toggle("hidden", state !== "incall");
  $("avatar").classList.toggle("ringing", state === "ringing");
  $("calBtn").disabled = state !== "incall";
  clearInterval(U.tick);
  if (state === "incall") {
    U.started = Date.now();
    U.tick = setInterval(() => {
      const s = Math.floor((Date.now() - U.started) / 1000);
      $("timer").textContent = `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
    }, 500);
  } else if (state !== "ended") {
    $("timer").textContent = state === "ringing" ? "ringing" : "00:00";
  }
}

// ------------------------------------------------------------------ signalling
function goOnline() {
  const room = $("room").value.trim();
  if (!/^[A-Za-z0-9]{1,32}$/.test(room)) { alert("Room: letters and digits only"); return; }
  store.set("vs-room", room);
  U.sig = new Signal(room, "customer", onSignal);
  U.sig.closed.then(() => { U.sig = null; setCall("offline", "Offline · server connection closed"); });
  setCall("online");
}

async function onSignal(m) {
  switch (m.type) {
    case "joined": setCall("online", m.peer ? "Online · caller connected" : "Online · waiting for a call"); break;
    case "peer-joined": if (!U.pc) setCall("online", "Online · caller connected"); break;
    case "offer":
      U.offer = m.sdp; U.ice = [];
      showWho();
      setCall("ringing");
      break;
    case "ice":
      if (U.pc && U.pc.remoteDescription) { try { await U.pc.addIceCandidate(m.candidate); } catch (e) { /* late */ } }
      else U.ice.push(m.candidate);
      break;
    case "app":
      if (m.action === "transfer_request") showRequest(m.amount, m.payee);
      break;
    case "bye":
    case "peer-left":
      if (U.pc) endCall("The caller hung up", false);
      else if (U.sig) setCall("online");
      break;
  }
}

async function answer() {
  U.ctx = await VSAnalyzer.prepare(U.ctx);              // inside the click: unlocks audio
  U.pc = newPeer(U.sig, onRemote, (state) => {
    if (state === "failed") setCall("incall", "Call connection failed");
  });
  await U.pc.setRemoteDescription({ type: "offer", sdp: U.offer });
  for (const c of U.ice) { try { await U.pc.addIceCandidate(c); } catch (e) { /* ignore */ } }
  U.ice = [];
  const tr = U.pc.getTransceivers()[0];
  if (tr) tr.direction = "sendrecv";                   // lets "Talk" add the customer's mic later without renegotiation
  const ans = await U.pc.createAnswer();
  ans.sdp = hqOpus(ans.sdp);
  await U.pc.setLocalDescription(ans);
  U.sig.send({ type: "answer", sdp: ans.sdp });
  setCall("incall");
  addActivity("phone", `Call answered${$("claimed").value ? ` · caller says they are ${$("claimed").value}` : ""}`);
}

function reject() {
  U.sig.send({ type: "app", action: "declined" });
  setCall("online", "Call declined · still online");
}

function onRemote(stream) {
  U.remote = stream;
  const a = $("remoteAudio");
  a.srcObject = stream;                                // also needed so Chrome feeds remote audio to Web Audio
  a.muted = !U.speaker;
  startAnalysis("monitor");
}

async function startAnalysis(mode) {
  const session = ++U.session;
  U.mode = mode;
  if (U.an) U.an.stop();
  resetView(mode);
  U.an = new VSAnalyzer((m) => { if (session === U.session) onAnalysis(m); }, U.ctx);
  try {
    const ready = await U.an.start(U.remote, {
      mode, source: `call:${$("room").value.trim()}`, profile: $("profile").value,
      claimed: mode === "monitor" ? ($("claimed").value || null) : null,
      language: $("lang").value || null, save_as: "call" });
    if (mode === "monitor") {
      U.callId = ready.call_id;
      $("asrInfo").textContent = ready.transcript ? "Live transcript · scam phrases highlighted" : "Speech-to-text is off on the server";
    }
  } catch (e) {
    setCall("incall", e.message);
  }
}

function toggleSpeaker() {
  U.speaker = !U.speaker;
  $("remoteAudio").muted = !U.speaker;
  $("speakerBtn").className = "round" + (U.speaker ? " on" : "");
  $("speakerBtn").innerHTML = I(U.speaker ? "volume" : "volume-x", "lg");
  $("speakerLbl").textContent = U.speaker ? "Speaker on" : "Speaker off";
}

async function toggleTalk() {
  if (!U.talk) {
    try {   // the customer's own voice is not analysed, so normal call processing (echo cancellation) is fine
      const s = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
      U.talk = s.getAudioTracks()[0];
      U.talk.enabled = false;
      await U.pc.getTransceivers()[0].sender.replaceTrack(U.talk);
    } catch (e) { alert("Microphone not available: " + e.message); return; }
  }
  U.talk.enabled = !U.talk.enabled;
  $("talkBtn").className = "round" + (U.talk.enabled ? " on" : "");
  $("talkBtn").innerHTML = I(U.talk.enabled ? "mic" : "mic-off", "lg");
  $("talkLbl").textContent = U.talk.enabled ? "Talking" : "Muted";
}

function endCall(text, tellPeer) {
  if (tellPeer && U.sig) U.sig.send({ type: "bye" });
  U.session++;
  if (U.an) {
    const an = U.an; U.an = null;
    an.onMessage = (m) => { if (m.type === "summary") addActivity(m.verdict, `Call ended · VoiceShield verdict ${m.verdict}`); };
    an.stop();
  }
  if (U.pc) { U.pc.close(); U.pc = null; }
  if (U.talk) { U.talk.stop(); U.talk = null; }
  $("talkBtn").className = "round"; $("talkBtn").innerHTML = I("mic-off", "lg"); $("talkLbl").textContent = "Muted";
  clearInterval(U.calTimer);
  setCall(U.sig ? "ended" : "offline", text + (U.sig ? " · still online" : ""));
  setTimeout(() => { if (!U.pc && U.sig) setCall("online"); }, 4000);
  if (U.request && U.request.state === "pending") { U.request.state = "callended"; renderRequest(); }
}

// ------------------------------------------------------------------ calibration of the call line
function calibrate() {
  if (!U.remote) return;
  if (!confirm("Calibrate with the CALLER's voice?\nOnly do this when the genuine caller is on the line. They should talk normally for 25 s.")) return;
  startAnalysis("calibrate");
  const end = Date.now() + 25000;
  clearInterval(U.calTimer);
  U.calTimer = setInterval(() => {
    const left = Math.ceil((end - Date.now()) / 1000);
    $("calStatus").textContent = left > 0 ? `Caller keeps talking… ${left} s left` : "Saving…";
    if (left <= 0) { clearInterval(U.calTimer); U.an.stop(); }
  }, 500);
}

// ------------------------------------------------------------------ analysis display
function setVerdict(level, opts = {}) {
  const v = VSUI.VERDICT[level] || VSUI.VERDICT.IDLE;
  U.level = level;
  $("verdictBox").className = "verdict " + level;
  $("vIcon").innerHTML = I(v.icon, "sm");
  $("vState").textContent = v.state;
  $("vTitle").textContent = v.title;
  $("vWhy").textContent = opts.reasons && opts.reasons.length ? capital(opts.reasons.join("; ")) + "." : v.why;
  VSUI.setGauge($("gauge"), opts.risk ?? null, ["PASS", "VERIFY", "HOLD"].includes(level) ? level : "");
  $("verdict").innerHTML = opts.verdict ? `<span class="vbadge ${opts.verdict}">${opts.verdict}</span>` : "–";
}
const capital = (s) => s.charAt(0).toUpperCase() + s.slice(1);

function resetView(mode) {
  U.points = []; U.clock = 0;
  if (mode === "monitor") U.verdict = "PASS";
  setVerdict(mode === "monitor" ? "PASS" : "CAL", { risk: mode === "monitor" ? 0 : null, verdict: mode === "monitor" ? "PASS" : null });
  $("clock").textContent = "0.0 s";
  $("action").className = "alert hidden";
  if (mode === "monitor") emptyTranscript();
  else $("transcript").innerHTML = `<div class="empty">${I("sliders")}Calibrating the line: no transcript.</div>`;
  setSignals({ voice: 0, identity: mode === "monitor" && $("claimed").value ? 0 : null, scam: 0 });
  U.chart.set([], 0);
}

function emptyTranscript() {
  $("transcript").innerHTML = `<div class="empty">${I("message")}The caller's words appear here during the call.</div>`;
}

function setSignals(sig, tactics) {
  const one = (bar, val, v) => {
    if (v === null || v === undefined) { $(bar).style.width = "0"; $(val).textContent = "off"; $(val).className = "v off"; return; }
    $(bar).style.width = `${Math.max(0, Math.min(100, v))}%`;
    $(bar).className = "meter-fill" + (v >= 75 ? " high" : v >= 45 ? " mid" : "");
    $(val).textContent = Number(v).toFixed(0); $(val).className = "v";
  };
  one("sigVoice", "sigVoiceV", sig.voice);
  one("sigId", "sigIdV", sig.identity);
  one("sigScam", "sigScamV", sig.scam);
  if (tactics && tactics.length) $("sigScamS").textContent = `${tactics.length} tactic${tactics.length > 1 ? "s" : ""} seen`;
}

function showState(m) {
  setVerdict(m.level, { risk: m.risk, verdict: m.verdict, reasons: m.level === "PASS" ? [] : m.reasons });
  if (m.signals) setSignals(m.signals);
  if (m.verdict !== U.verdict) {
    U.verdict = m.verdict;
    if (U.request && U.request.state === "verify" && U.verdict === "HOLD") block(m.reasons);
    renderRequest();
  }
}

function highlight(text, hits) {
  let html = esc(text);
  for (const h of hits || []) {
    const p = esc(h.phrase).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    html = html.replace(new RegExp(p, "i"), (s) => `<mark title="${esc(h.label)}">${s}</mark>`);
  }
  return html;
}

function onAnalysis(m) {
  switch (m.type) {
    case "vad": U.clock = m.t; $("clock").textContent = `${fmt(m.t, 1)} s`; break;
    case "score":
      U.points.push({ t: m.t, risk: m.risk, s: 100 * m.window_score, llr: m.llr, level: m.level });
      showState(m);
      U.chart.set(U.points, U.clock);
      break;
    case "transcript": {
      if ($("transcript").querySelector(".empty")) $("transcript").innerHTML = "";
      const d = document.createElement("div");
      d.className = "utt" + ((m.hits || []).length ? " scam" : "");
      const chips = (m.hits || []).map((h) => `<span class="tag crit">${esc(h.label)}</span>`).join("");
      d.innerHTML = `<span class="ts">${fmt(m.t0, 0)}s</span><div><div class="bubble">${highlight(m.text, m.hits)}</div>` +
        (chips ? `<div class="tactics">${chips}</div>` : "") + `</div>`;
      $("transcript").appendChild(d);
      $("transcript").scrollTop = $("transcript").scrollHeight;
      showState(m);
      break;
    }
    case "action": {
      const hold = m.level === "HOLD";
      $("action").className = "alert " + m.level;
      $("action").innerHTML = `${I(hold ? "octagon" : "alert", "lg")}<div><b>${hold ? "Hold" : "Verify"} · ${fmt(m.t, 1)} s into the call</b>` +
        `<p>${esc(capital((m.reasons || []).join("; ")))}. ${esc(m.message)}</p></div>`;
      addActivity(m.level, `VoiceShield ${m.level}: ${(m.reasons || []).join("; ")}`);
      if (hold && U.request && ["pending", "verify"].includes(U.request.state)) block(m.reasons);
      break;
    }
    case "calib_window": $("calStatus").textContent = `Caller keeps talking… (${m.n} windows analysed)`; break;
    case "calibrated":
      $("calStatus").textContent = `Saved line profile "call": cut-off ${fmt(m.llr_threshold)} from ${m.n_windows} windows. Monitoring again.`;
      loadSettings("call").then(() => { store.set("vs-cust-profile", "call"); if (U.remote && U.pc) startAnalysis("monitor"); });
      break;
    case "error":
      $("calStatus").textContent = m.message;
      if (U.mode === "calibrate" && U.remote && U.pc) startAnalysis("monitor");
      break;
  }
}

// ------------------------------------------------------------------ bank app + prevention
const FEED_ICON = { HOLD: ["octagon", "crit"], VERIFY: ["alert", "warn"], TRANSFER_BLOCKED: ["lock", "crit"],
  TRANSFER_APPROVED: ["check", "good"], OTP_VERIFIED: ["check", "good"], OTP_SENT: ["smartphone", "warn"],
  OTP_FAILED: ["x", "crit"], CALLBACK_REQUESTED: ["phone", "warn"], TRANSFER_REQUESTED: ["rupee", ""],
  TRANSFER_DECLINED: ["x", ""], PASS: ["shield-check", "good"], phone: ["phone", ""], info: ["shield", ""] };

function addActivity(kind, text) {
  const [ic, cls] = FEED_ICON[kind] || ["info", ""];
  const li = document.createElement("li");
  li.className = cls;
  li.innerHTML = `${I(ic)}<div>${esc(text)}<time>${new Date().toLocaleTimeString()}</time></div>`;
  $("activity").prepend(li);
  while ($("activity").children.length > 30) $("activity").lastChild.remove();
}

function postEvent(kind, detail) {
  if (!U.callId) return;
  fetch(`/api/calls/${U.callId}/events`, { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ kind, t: U.clock, detail: String(detail).slice(0, 300) }) }).catch(() => {});
}

function showRequest(amount, payee) {
  U.request = { amount: Math.max(1, Math.round(Number(amount) || 0)), payee: String(payee || "unknown").slice(0, 40),
                state: "pending", reasons: [] };
  addActivity("TRANSFER_REQUESTED", `Payment request: ${INR(U.request.amount)} to ${U.request.payee}`);
  postEvent("TRANSFER_REQUESTED", `${INR(U.request.amount)} to ${U.request.payee}`);
  $("phone").classList.add("hidden");
  if (U.verdict === "HOLD") block(["the call was already on HOLD"]);
  renderRequest();
}

function pay() {
  if (U.verdict === "HOLD") return block();
  if (U.verdict === "VERIFY") { U.request.state = "verify"; renderRequest(); return; }
  approve("approved by you; the call looked genuine");
}

function approve(how) {
  U.balance -= U.request.amount;
  $("balance").textContent = U.balance.toLocaleString("en-IN");
  U.request.state = "done";
  addActivity("TRANSFER_APPROVED", `Paid ${INR(U.request.amount)} to ${U.request.payee} (${how})`);
  postEvent("TRANSFER_APPROVED", `${INR(U.request.amount)} to ${U.request.payee}: ${how}`);
  renderRequest();
}

function block(reasons) {
  if (!U.request || ["blocked", "done"].includes(U.request.state)) return;
  U.request.state = "blocked";
  U.request.reasons = reasons || [];
  addActivity("TRANSFER_BLOCKED", `Blocked ${INR(U.request.amount)} to ${U.request.payee} · security team alerted`);
  postEvent("TRANSFER_BLOCKED", `${INR(U.request.amount)} to ${U.request.payee}`);
  $("phone").classList.add("hidden");
  renderRequest();
}

function decline() {
  U.request.state = "declined";
  addActivity("TRANSFER_DECLINED", "You declined the payment request");
  postEvent("TRANSFER_DECLINED", `${INR(U.request.amount)} to ${U.request.payee}`);
  renderRequest();
}

function sendOtp() {
  U.otp = String(Math.floor(100000 + Math.random() * 900000));
  const ph = $("phone");
  ph.classList.remove("hidden");
  ph.innerHTML = `<div class="from"><span>MESSAGES · MyBank</span><span>now</span></div>` +
    `<span class="code">${U.otp}</span> is your OTP to pay ${INR(U.request.amount)} to ${esc(U.request.payee)}. ` +
    `<b>Never share it with anyone on a call.</b> MyBank will never ask for it.`;
  addActivity("OTP_SENT", "OTP sent to your registered phone (not to the caller)");
  postEvent("OTP_SENT", "OTP sent to the registered phone");
  U.request.otpSent = true;
  renderRequest();
}

function confirmOtp() {
  const v = ($("otpIn").value || "").trim();
  if (v && v === U.otp) {
    postEvent("OTP_VERIFIED", "customer confirmed in the bank app");
    addActivity("OTP_VERIFIED", "OTP confirmed in the bank app");
    $("phone").classList.add("hidden");
    approve("approved after OTP check in the bank app");
  } else {
    postEvent("OTP_FAILED", "wrong OTP entered");
    addActivity("OTP_FAILED", "Wrong OTP: payment not sent");
    $("otpMsg").textContent = "Wrong OTP. Payment not sent.";
  }
}

function callBack() {
  postEvent("CALLBACK_REQUESTED", "customer hung up to call the bank on its official number");
  addActivity("CALLBACK_REQUESTED", "You hung up to call the bank back on its official number");
  U.request.state = "callback";
  endCall("You hung up to call the bank back", true);
  renderRequest();
}

function renderRequest() {
  const box = $("request");
  const r = U.request;
  if (!r) { box.innerHTML = `<div class="req"><div class="empty" style="padding:14px">${I("rupee")}No payment requests</div></div>`; return; }
  const head = `<div class="top"><span class="headicon">${I("send", "sm")}</span><div><b>Requested by the caller</b>` +
    `<div class="xs muted">during this call</div></div></div><div class="amt">${INR(r.amount)}</div><div class="to">to ${esc(r.payee)}</div>`;
  const note = U.verdict === "PASS" ? ["shield-check", "good", "VoiceShield: this call looks genuine."] :
    U.verdict === "VERIFY" ? ["alert", "warn", "VoiceShield: suspicious call. Extra check needed."] :
    ["octagon", "crit", "VoiceShield: likely cloned voice."];
  let html = "", cls = "req";
  if (r.state === "pending") {
    html = head + `<div class="msg"><span class="tag ${note[1]}">${I(note[0], "sm")}${esc(note[2])}</span></div>` +
      `<div class="btn-row"><button id="payBtn" class="btn primary">Pay now</button><button id="decBtn" class="btn ghost">Decline</button></div>`;
  } else if (r.state === "verify") {
    cls += " verify";
    html = head + `<div class="ribbon">${I("alert")}Verification required</div>` +
      `<div class="msg">This call was flagged as suspicious. Confirm the payment in a way the caller cannot control.</div>` +
      `<div class="btn-row"><button id="otpBtn" class="btn sm">${I("smartphone", "sm")}Send OTP to my phone</button>` +
      `<button id="cbBtn" class="btn sm">${I("phone", "sm")}Hang up &amp; call bank</button></div>` +
      (r.otpSent ? `<div class="otp"><input id="otpIn" class="input" maxlength="6" inputmode="numeric" placeholder="••••••">` +
        `<button id="otpOk" class="btn primary">Confirm</button></div><p id="otpMsg" class="small" style="color:var(--crit-ink)"></p>` : "");
  } else if (r.state === "blocked") {
    cls += " blocked";
    html = head + `<div class="ribbon">${I("lock")}Payment blocked by VoiceShield</div>` +
      `<div class="msg">This call shows signs of a cloned or impersonated voice${r.reasons && r.reasons.length ? ": " + esc(r.reasons.join("; ")) : ""}.</div>` +
      `<div class="msg"><b>Security team alerted${U.callId ? ` · incident #${U.callId}` : ""}.</b> Do not share any OTP or PIN on this call.</div>`;
  } else if (r.state === "done") {
    cls += " done";
    html = head + `<div class="ribbon">${I("check")}Paid</div>`;
  } else if (r.state === "declined") {
    html = head + `<div class="msg">You declined this request.</div>`;
  } else if (r.state === "callback") {
    cls += " verify";
    html = head + `<div class="ribbon">${I("phone")}Payment on hold</div><div class="msg">Call the number printed on your card ` +
      `(1800-XXX-XXXX) to confirm. Never call back a number the caller gives you.</div>`;
  } else if (r.state === "callended") {
    html = head + `<div class="msg">The call ended before you acted. Request expired.</div>`;
  }
  box.innerHTML = `<div class="${cls}">${html}</div>`;
  const on = (id, fn) => { if ($(id)) $(id).onclick = fn; };
  on("payBtn", pay); on("decBtn", decline); on("otpBtn", sendOtp); on("cbBtn", callBack); on("otpOk", confirmOtp);
}

init();
