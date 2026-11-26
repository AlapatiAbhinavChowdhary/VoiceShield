"use strict";
// VoiceShield UI shell: icons, sidebar + top bar, toasts, dialogs, gauge, risk chart, the live waveform,
// the segment lane, spectrogram + overview renderers and small SVG charts. Exposes window.VSUI only.
// Everything is drawn from real data handed in by the pages; nothing here invents signal or scores.
(function () {
  // ---------------------------------------------------------------- theme (before first paint: no flash)
  const THEME_KEY = "vs-theme";
  const savedTheme = (() => { try { return localStorage.getItem(THEME_KEY); } catch (e) { return null; } })();
  if (savedTheme === "light") document.documentElement.dataset.theme = "light";
  function setTheme(t) {
    if (t === "light") document.documentElement.dataset.theme = "light"; else delete document.documentElement.dataset.theme;
    try { localStorage.setItem(THEME_KEY, t); } catch (e) { /* private mode: theme lasts for this page only */ }
    const b = document.getElementById("themeBtn");
    if (b) { b.innerHTML = icon(t === "light" ? "moon" : "sun"); b.setAttribute("aria-label", t === "light" ? "Switch to dark theme" : "Switch to light theme"); b.title = b.getAttribute("aria-label"); }
    window.dispatchEvent(new CustomEvent("vs-theme", { detail: t }));
  }
  const theme = () => (document.documentElement.dataset.theme === "light" ? "light" : "dark");
  // ---------------------------------------------------------------- icons (24x24 stroke, Lucide style)
  const SH = "M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z";
  const P = {
    shield: [SH], "shield-check": [SH, "m9 12 2 2 4-4"], "shield-alert": [SH, "M12 8v4", "M12 16h.01"],
    sun: ["M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8z", "M12 2v2", "M12 20v2", "m4.93 4.93 1.41 1.41", "m17.66 17.66 1.41 1.41", "M2 12h2", "M20 12h2", "m6.34 17.66-1.41 1.41", "m19.07 4.93-1.41 1.41"],
    moon: ["M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9z"],
    flag: ["M4 22V4", "M4 4h12l-2 4 2 4H4"],
    maximize: ["M8 3H5a2 2 0 0 0-2 2v3", "M21 8V5a2 2 0 0 0-2-2h-3", "M3 16v3a2 2 0 0 0 2 2h3", "M16 21h3a2 2 0 0 0 2-2v-3"],
    "arrow-right": ["M5 12h14", "m13 6 6 6-6 6"],
    flask: ["M9 3h6", "M10 3v6.5L4.5 19a1.5 1.5 0 0 0 1.3 2.25h12.4A1.5 1.5 0 0 0 19.5 19L14 9.5V3", "M7 15h10"],
    logo: [SH, "M8 12.5h1.4l1.3-2.8 1.9 5.3 1.4-3.3H16"],
    alert: ["m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3", "M12 9v4", "M12 17h.01"],
    octagon: ["M2.59 16.73A2 2 0 0 1 2 15.31V8.69a2 2 0 0 1 .59-1.42l4.68-4.68A2 2 0 0 1 8.69 2h6.62a2 2 0 0 1 1.42.59l4.68 4.68A2 2 0 0 1 22 8.69v6.62a2 2 0 0 1-.59 1.42l-4.68 4.68a2 2 0 0 1-1.42.59H8.69a2 2 0 0 1-1.42-.59z", "m15 9-6 6", "m9 9 6 6"],
    lock: ["M5 11h14a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2z", "M7 11V7a5 5 0 0 1 10 0v4"],
    phone: ["M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z"],
    "phone-off": ["M10.68 13.31a16 16 0 0 0 3.41 2.6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7 2 2 0 0 1 1.72 2v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.42 19.42 0 0 1-3.33-2.67m-2.67-3.34a19.79 19.79 0 0 1-3.07-8.63A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91", "M22 2 2 22"],
    mic: ["M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z", "M19 10v2a7 7 0 0 1-14 0v-2", "M12 19v3"],
    "mic-off": ["M2 2l20 20", "M18.89 13.23A7.12 7.12 0 0 0 19 12v-2", "M5 10v2a7 7 0 0 0 12 5", "M15 9.34V5a3 3 0 0 0-5.68-1.33", "M9 9v3a3 3 0 0 0 5.12 2.12", "M12 19v3"],
    volume: ["M11 4.7a.7.7 0 0 0-1.2-.5L6.41 7.59A1.4 1.4 0 0 1 5.42 8H3a1 1 0 0 0-1 1v6a1 1 0 0 0 1 1h2.42a1.4 1.4 0 0 1 .99.41l3.39 3.39a.7.7 0 0 0 1.2-.5z", "M16 9a5 5 0 0 1 0 6", "M19.36 18.36a9 9 0 0 0 0-12.73"],
    "volume-x": ["M11 4.7a.7.7 0 0 0-1.2-.5L6.41 7.59A1.4 1.4 0 0 1 5.42 8H3a1 1 0 0 0-1 1v6a1 1 0 0 0 1 1h2.42a1.4 1.4 0 0 1 .99.41l3.39 3.39a.7.7 0 0 0 1.2-.5z", "m22 9-6 6", "m16 9 6 6"],
    wave: ["M2 12h2.5l2-6 3 12 3-15 3 15 2-6H22"],
    "audio-lines": ["M2 10v3", "M6 6v11", "M10 3v18", "M14 8v7", "M18 5v13", "M22 10v3"],
    fingerprint: ["M12 10a2 2 0 0 0-2 2c0 1.02-.1 2.51-.26 4", "M14 13.12c0 2.38 0 6.38-1 8.88", "M17.29 21.02c.12-.6.43-2.3.5-3.02", "M2 12a10 10 0 0 1 18-6", "M2 16h.01", "M21.8 16c.2-2 .13-5.35 0-6", "M5 19.5C5.5 18 6 15 6 12a6 6 0 0 1 .34-2", "M8.65 22c.21-.66.45-1.32.57-2", "M9 6.8a6 6 0 0 1 9 5.2v2"],
    message: ["M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z", "M12 7v3.5", "M12 13.5h.01"],
    bank: ["M10 18v-7", "M11.12 2.2a2 2 0 0 1 1.76 0l7.87 3.85c.47.23.31.95-.22.95H3.47c-.53 0-.7-.72-.22-.95z", "M14 18v-7", "M18 18v-7", "M3 22h18", "M6 18v-7"],
    send: ["M14.54 21.69a.5.5 0 0 0 .94-.03l6.5-19a.5.5 0 0 0-.64-.63l-19 6.5a.5.5 0 0 0-.02.93l7.93 3.18a2 2 0 0 1 1.11 1.11z", "m21.85 2.15-10.94 10.94"],
    play: ["M6 3.5 20 12 6 20.5Z"], pause: ["M6 4h4v16H6z", "M14 4h4v16h-4z"], stop: ["M6 6h12v12H6z"],
    upload: ["M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4", "M17 8l-5-5-5 5", "M12 3v12"],
    download: ["M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4", "m7 10 5 5 5-5", "M12 15V3"],
    printer: ["M6 9V2h12v7", "M6 18H4a2 2 0 0 1-2-2v-5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v5a2 2 0 0 1-2 2h-2", "M6 14h12v8H6z"],
    user: ["M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2", "M12 11a4 4 0 1 0 0-8 4 4 0 0 0 0 8z"],
    users: ["M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2", "M9 11a4 4 0 1 0 0-8 4 4 0 0 0 0 8z", "M22 21v-2a4 4 0 0 0-3-3.87", "M16 3.13a4 4 0 0 1 0 7.75"],
    sliders: ["M4 21v-7", "M4 10V3", "M12 21v-9", "M12 8V3", "M20 21v-5", "M20 12V3", "M2 14h4", "M10 8h4", "M18 16h4"],
    live: ["M12 14a2 2 0 1 0 0-4 2 2 0 0 0 0 4z", "M16.24 7.76a6 6 0 0 1 0 8.49", "M7.76 16.24a6 6 0 0 1 0-8.49", "M19.07 4.93a10 10 0 0 1 0 14.14", "M4.93 19.07a10 10 0 0 1 0-14.14"],
    clock: ["M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20z", "M12 6v6l4 2"],
    check: ["M22 11.08V12a10 10 0 1 1-5.93-9.14", "m9 11 3 3L22 4"],
    x: ["M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20z", "m15 9-6 6", "m9 9 6 6"], close: ["M18 6 6 18", "m6 6 12 12"],
    zap: ["M4 14a1 1 0 0 1-.78-1.63l9.9-10.2a.5.5 0 0 1 .86.46l-1.92 6.02A1 1 0 0 0 13 10h7a1 1 0 0 1 .78 1.63l-9.9 10.2a.5.5 0 0 1-.86-.46l1.92-6.02A1 1 0 0 0 11 14z"],
    trash: ["M3 6h18", "M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6", "M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"],
    rupee: ["M6 3h12", "M6 8h12", "m6 13 8.5 8", "M6 13h3", "M9 13c6.67 0 6.67-10 0-10"],
    smartphone: ["M17 2H7a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V4a2 2 0 0 0-2-2z", "M12 18h.01"],
    activity: ["M22 12h-4l-3 9L9 3l-3 9H2"],
    info: ["M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20z", "M12 16v-4", "M12 8h.01"],
    chevron: ["m9 18 6-6-6-6"], "chevron-down": ["m6 9 6 6 6-6"], "arrow-left": ["m12 19-7-7 7-7", "M19 12H5"],
    target: ["M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20z", "M12 18a6 6 0 1 0 0-12 6 6 0 0 0 0 12z", "M12 14a2 2 0 1 0 0-4 2 2 0 0 0 0 4z"],
    refresh: ["M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8", "M21 3v5h-5", "M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16", "M8 16H3v5"],
    radar: ["M19.07 4.93A10 10 0 0 0 6.99 3.34", "M4 6h.01", "M2.29 9.62A10 10 0 1 0 21.31 8.35", "M16.24 7.76A6 6 0 1 0 8.23 16.67", "M12 18h.01", "M17.99 11.66A6 6 0 0 1 15.77 16.67", "M12 14a2 2 0 1 0 0-4 2 2 0 0 0 0 4z", "m13.41 10.59 5.66-5.66"],
    history: ["M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8", "M3 3v5h5", "M12 7v5l4 2"],
    chart: ["M3 3v16a2 2 0 0 0 2 2h16", "M7 16V11", "M11 16V7", "M15 16v-4", "M19 16V9"],
    search: ["M11 19a8 8 0 1 0 0-16 8 8 0 0 0 0 16z", "m21 21-4.3-4.3"],
    settings: ["M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z", "M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6z"],
    external: ["M15 3h6v6", "M10 14 21 3", "M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"],
    menu: ["M4 6h16", "M4 12h16", "M4 18h16"],
    cpu: ["M18 4H6a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V6a2 2 0 0 0-2-2z", "M9 9h6v6H9z", "M9 1v3", "M15 1v3", "M9 20v3", "M15 20v3", "M20 9h3", "M20 14h3", "M1 9h3", "M1 14h3"],
    layers: ["m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z", "m22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65", "m22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65"],
    gauge: ["m12 14 4-4", "M3.34 19a10 10 0 1 1 17.32 0"],
    file: ["M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z", "M14 2v4a2 2 0 0 0 2 2h4", "M8 13v3", "M11 11v7", "M14 13v3"],
    eye: ["M2.06 12.35a1 1 0 0 1 0-.7 10.75 10.75 0 0 1 19.88 0 1 1 0 0 1 0 .7 10.75 10.75 0 0 1-19.88 0", "M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6z"],
    database: ["M12 8c4.97 0 9-1.34 9-3s-4.03-3-9-3-9 1.34-9 3 4.03 3 9 3z", "M3 5v14a9 3 0 0 0 18 0V5", "M3 12a9 3 0 0 0 18 0"],
    music: ["M9 18V5l12-2v13", "M6 21a3 3 0 1 0 0-6 3 3 0 0 0 0 6z", "M18 19a3 3 0 1 0 0-6 3 3 0 0 0 0 6z"],
    flask: ["M10 2v7.53a2 2 0 0 1-.21.9L4.72 20.55a1 1 0 0 0 .9 1.45h12.76a1 1 0 0 0 .9-1.45l-5.07-10.12A2 2 0 0 1 14 9.53V2", "M8.5 2h7", "M7 16h10"],
  };
  function icon(name, cls = "") {
    const paths = P[name] || P.info;
    return `<svg class="i ${cls}" viewBox="0 0 24 24" aria-hidden="true">${paths.map((d) => `<path d="${d}"/>`).join("")}</svg>`;
  }
  function hydrate(root = document) {
    root.querySelectorAll("i[data-i]").forEach((el) => { el.outerHTML = icon(el.dataset.i, el.className); });
  }
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const reduced = () => matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ---------------------------------------------------------------- sidebar + top bar
  const NAV = [
    ["Monitor"],
    ["/#/live", "radar", "Command center"],
    ["/#/analysis", "audio-lines", "Voice analysis"],
    ["Investigate"],
    ["/#/incidents", "shield-alert", "Incidents", "incBadge"],
    ["/#/history", "history", "Detection history"],
    ["/#/analytics", "chart", "Analytics"],
    ["/#/eval", "gauge", "Evaluation"],
    ["Call demo"],
    ["/customer", "smartphone", "Customer phone", null, true],
    ["/caller", "phone", "Caller", null, true],
    ["/#/lab", "flask", "Scenario lab"],
    ["System"],
    ["/#/settings", "settings", "Settings & enrolment"],
  ];

  function mountTopbar(active, title) {
    const top = document.getElementById("topbar");
    if (!top) return;
    document.body.classList.add("shell");
    const side = document.createElement("aside");
    side.className = "sidebar";
    side.setAttribute("aria-label", "Main navigation");
    side.innerHTML =
      `<a class="sb-brand" href="/#/live"><span class="logo">${icon("logo")}</span>` +
      `<span class="sb-name">VoiceShield<small>Voice-clone defence</small></span></a>` +
      `<nav class="sb-nav">${NAV.map(([href, ic, label, badge, page]) => !ic ? `<div class="sb-group">${href}</div>` :
        `<a class="sb-link" href="${href}" data-href="${href}" title="${label}">${icon(ic)}<span>${label}</span>` +
        (badge ? `<em id="${badge}" class="badge hidden"></em>` : "") +
        (page ? `<em class="ext" data-open="${href}" title="Open in a new window">${icon("external", "xs")}</em>` : "") + `</a>`).join("")}</nav>` +
      `<div class="sb-foot" id="sys"><div class="sys-row"><span class="dot"></span><span class="lbl">Connecting…</span></div></div>`;
    document.body.prepend(side);
    const scrim = document.createElement("div");
    scrim.className = "scrim";
    scrim.onclick = () => document.body.classList.remove("nav-open");
    document.body.prepend(scrim);
    side.querySelectorAll("[data-open]").forEach((el) => el.addEventListener("click", (e) => {
      e.preventDefault(); e.stopPropagation(); window.open(el.dataset.open, "_blank", "noopener");
    }));
    side.querySelectorAll(".sb-link").forEach((a) => a.addEventListener("click", () => document.body.classList.remove("nav-open")));
    top.className = "topbar";
    top.innerHTML =
      `<button class="iconbtn menu" id="menuBtn" aria-label="Open navigation">${icon("menu")}</button>` +
      `<div class="crumbs" id="crumbs"></div><div class="spacer"></div>` +
      `<button class="session-pill hidden" id="sessionPill" title="Go to the live session"></button>` +
      `<button class="iconbtn" id="themeBtn" type="button"></button>`;
    document.getElementById("menuBtn").onclick = () => document.body.classList.toggle("nav-open");
    setTheme(theme());
    document.getElementById("themeBtn").onclick = () => setTheme(theme() === "light" ? "dark" : "light");
    setActive(active, title);
    loadStatus();
    setInterval(loadStatus, 15000);
  }

  function setActive(href, title) {
    document.querySelectorAll(".sb-link").forEach((a) => a.classList.toggle("active", a.dataset.href === href));
    const link = [...document.querySelectorAll(".sb-link")].find((a) => a.dataset.href === href);
    const group = link ? link.previousElementSibling : null;
    let g = group; while (g && !g.classList.contains("sb-group")) g = g.previousElementSibling;
    const c = document.getElementById("crumbs");
    if (c) c.innerHTML = `<span>${esc(g ? g.textContent : "VoiceShield")}</span>${icon("chevron", "xs")}<b>${esc(title || (link ? link.title : ""))}</b>`;
  }

  async function loadStatus() {
    const box = document.getElementById("sys");
    if (!box) return null;
    try {
      const st = await (await fetch("/api/status")).json();
      const det = (st.detector || "").replace("ExecutionProvider", "");
      const row = (cls, text, tip) => `<div class="sys-row" title="${esc(tip)}"><span class="dot ${cls}"></span><span class="lbl">${esc(text)}</span></div>`;
      box.innerHTML =
        row("good", `${st.detector_mode === "step8c" ? "Detector + Step 8c" : "Detector"} · ${det}`,
          st.detector_mode === "step8c" ? "NII AntiDeepfake wav2vec2-Large (frozen) + Step 8c back-end on its 25 layers" : "NII AntiDeepfake wav2vec2-Large") +
        row(st.asr ? "good" : "", st.asr || "Speech-to-text off", "Speech-to-text (Whisper)") +
        row(st.speaker ? "good" : "", `Voiceprints · ${(st.voiceprints || []).length} enrolled`, "WeSpeaker ResNet34 voiceprints");
      return st;
    } catch (e) {
      box.innerHTML = `<div class="sys-row"><span class="dot crit"></span><span class="lbl">Server offline</span></div>`;
      return null;
    }
  }

  // ---------------------------------------------------------------- toasts + confirm dialog
  function toast(kind, title, msg = "", ms = 4200) {
    let host = document.querySelector(".toasts");
    if (!host) { host = document.createElement("div"); host.className = "toasts"; host.setAttribute("role", "status"); host.setAttribute("aria-live", "polite"); document.body.appendChild(host); }
    const ic = { crit: "octagon", warn: "alert", good: "check", brand: "info" }[kind] || "info";
    const el = document.createElement("div");
    el.className = `toast ${kind}`;
    el.innerHTML = `${icon(ic)}<div><b>${esc(title)}</b>${msg ? `<div>${esc(msg)}</div>` : ""}</div>`;
    host.appendChild(el);
    setTimeout(() => { el.classList.add("out"); setTimeout(() => el.remove(), 220); }, ms);
  }
  function confirmDialog(title, body, ok = "Confirm", danger = false) {
    return new Promise((resolve) => {
      const d = document.createElement("dialog");
      d.className = "modal";
      d.innerHTML = `<div class="mh"><h3>${esc(title)}</h3></div><div class="mb">${esc(body)}</div>` +
        `<div class="mf"><button class="btn ghost" value="no">Cancel</button><button class="btn ${danger ? "danger" : "primary"}" value="yes">${esc(ok)}</button></div>`;
      document.body.appendChild(d);
      d.querySelectorAll("button").forEach((b) => { b.onclick = () => d.close(b.value); });
      d.addEventListener("close", () => { resolve(d.returnValue === "yes"); d.remove(); });
      d.showModal();
      d.querySelector(".btn.ghost").focus();
    });
  }

  // ---------------------------------------------------------------- verdict helpers
  const VERDICT = {
    IDLE: { icon: "shield", state: "Standing by", title: "Not monitoring", why: "Waiting for a call, the microphone or a recording to analyse." },
    PASS: { icon: "shield-check", state: "Pass · authentic", title: "Voice appears authentic", why: "No sign of a cloned voice, impersonation or scam script so far." },
    VERIFY: { icon: "alert", state: "Verify · suspicious", title: "Suspicious: verify before acting", why: "Something is off. Confirm through a channel the caller does not control." },
    HOLD: { icon: "octagon", state: "Hold · likely spoofed", title: "Likely cloned voice: transactions held", why: "Requests made on this call are blocked and the security team is alerted." },
    CAL: { icon: "sliders", state: "Calibrating", title: "Learning this line", why: "Keep talking normally. Nothing is judged while calibrating." },
  };
  function gaugeSVG() {
    return `<svg viewBox="0 0 100 100"><circle class="track" cx="50" cy="50" r="43" fill="none" stroke-width="7"/>
      <circle class="arc" cx="50" cy="50" r="43" fill="none" stroke-width="7" stroke-linecap="round" stroke-dasharray="270.2" stroke-dashoffset="270.2"/></svg>
      <div class="val"><div><b>–</b><small>risk</small></div></div>`;
  }
  function setGauge(el, value, level) {
    if (!el.querySelector("svg")) el.innerHTML = gaugeSVG();
    el.className = "gauge " + (level || "");
    const v = value === null || value === undefined ? null : Math.max(0, Math.min(100, value));
    el.querySelector(".arc").setAttribute("stroke-dashoffset", String(270.2 * (1 - (v || 0) / 100)));
    el.querySelector("b").textContent = v === null ? "–" : Math.round(v);
    el.setAttribute("role", "img");
    el.setAttribute("aria-label", v === null ? "risk not available" : `risk ${Math.round(v)} of 100${level ? ", " + level : ""}`);
  }

  // ---------------------------------------------------------------- canvas helper
  function fitCanvas(c, h) {
    const dpr = window.devicePixelRatio || 1, w = c.clientWidth || c.parentElement.clientWidth || 600;
    const H = h || c.clientHeight || 100;
    if (h) c.style.height = h + "px";
    if (c.width !== Math.round(w * dpr) || c.height !== Math.round(H * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(H * dpr); }
    const g = c.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { g, w, h: H };
  }
  function tipPos(tip, host, x, y) {
    const hw = host.clientWidth;
    tip.style.left = Math.min(Math.max(x + 14, 0), hw - tip.offsetWidth - 4) + "px";
    tip.style.top = Math.max(0, y - tip.offsetHeight - 12) + "px";
  }

  // ---------------------------------------------------------------- risk chart (canvas, with hover)
  class RiskChart {
    constructor(canvas, opts = {}) {
      this.c = canvas; this.h = opts.height || 200; this.verifyAt = opts.verifyAt ?? 45; this.holdAt = opts.holdAt ?? 75;
      this.points = []; this.clock = 0; this.hover = null;
      this.tip = document.createElement("div"); this.tip.className = "charttip hidden";
      canvas.parentElement.appendChild(this.tip);
      canvas.addEventListener("mousemove", (e) => this.move(e));
      canvas.addEventListener("mouseleave", () => { this.hover = null; this.tip.classList.add("hidden"); this.draw(); });
      if (window.ResizeObserver) new ResizeObserver(() => this.draw()).observe(canvas);
    }
    thresholds(v, h) { this.verifyAt = v; this.holdAt = h; this.draw(); }
    set(points, clock) { this.points = points; this.clock = clock || 0; this.draw(); }
    geom() {
      const w = this.c.clientWidth || 600, h = this.h, pad = { l: 34, r: 62, t: 12, b: 24 };
      const tmax = Math.max(20, Math.ceil(Math.max(this.clock, ...this.points.map((p) => p.t), 0) / 5) * 5);
      return { w, h, pad, tmax, X: (t) => pad.l + (w - pad.l - pad.r) * t / tmax, Y: (v) => pad.t + (h - pad.t - pad.b) * (1 - v / 100) };
    }
    draw() {
      if (!this.c.clientWidth) return;
      const { g, w, h } = fitCanvas(this.c, this.h);
      const { pad, tmax, X, Y } = this.geom();
      g.clearRect(0, 0, w, h);
      g.font = `11px ${css("--font")}`; g.lineWidth = 1;
      for (const v of [0, 25, 50, 75, 100]) {
        g.strokeStyle = v === 0 ? css("--axis") : css("--grid");
        g.beginPath(); g.moveTo(pad.l, Math.round(Y(v)) + 0.5); g.lineTo(w - pad.r, Math.round(Y(v)) + 0.5); g.stroke();
        g.fillStyle = css("--muted"); g.textAlign = "right"; g.fillText(String(v), pad.l - 8, Y(v) + 4);
      }
      const step = tmax > 120 ? 30 : tmax > 60 ? 10 : 5;
      g.textAlign = "center";
      for (let t = 0; t <= tmax; t += step) g.fillText(`${t}s`, X(t), h - 6);
      for (const [v, name, label] of [[this.verifyAt, "--warn", "VERIFY"], [this.holdAt, "--crit", "HOLD"]]) {
        g.setLineDash([4, 4]); g.strokeStyle = css(name); g.globalAlpha = 0.7; g.lineWidth = 1;
        g.beginPath(); g.moveTo(pad.l, Y(v)); g.lineTo(w - pad.r, Y(v)); g.stroke(); g.globalAlpha = 1;
        g.setLineDash([]); g.fillStyle = css(name); g.fillRect(w - pad.r + 6, Y(v) - 1, 6, 2);
        g.fillStyle = css("--ink-2"); g.textAlign = "left"; g.fillText(label, w - pad.r + 16, Y(v) + 4);
      }
      const pts = this.points;
      if (pts.length) {
        const s1 = css("--cyan");
        const grad = g.createLinearGradient(0, Y(100), 0, Y(0));
        grad.addColorStop(0, "rgba(34,211,238,.22)"); grad.addColorStop(1, "rgba(34,211,238,0)");
        g.beginPath(); g.moveTo(X(pts[0].t), Y(0));
        pts.forEach((p) => g.lineTo(X(p.t), Y(p.risk)));
        g.lineTo(X(pts[pts.length - 1].t), Y(0)); g.closePath(); g.fillStyle = grad; g.fill();
        g.strokeStyle = s1; g.lineWidth = 2; g.lineJoin = "round"; g.beginPath();
        pts.forEach((p, i) => (i ? g.lineTo(X(p.t), Y(p.risk)) : g.moveTo(X(p.t), Y(p.risk)))); g.stroke();
        g.lineWidth = 2; g.strokeStyle = css("--chart-surface"); g.fillStyle = css("--series-2");
        for (const p of pts) {
          if (p.s === undefined || p.s === null) continue;
          g.beginPath(); g.arc(X(p.t), Y(p.s), 4, 0, 2 * Math.PI); g.fill(); g.stroke();
        }
      }
      if (this.hover !== null && pts[this.hover]) {
        const p = pts[this.hover];
        g.strokeStyle = css("--line-3"); g.lineWidth = 1;
        g.beginPath(); g.moveTo(X(p.t) + 0.5, pad.t); g.lineTo(X(p.t) + 0.5, h - pad.b); g.stroke();
        g.lineWidth = 2; g.strokeStyle = css("--chart-surface"); g.fillStyle = css("--cyan");
        g.beginPath(); g.arc(X(p.t), Y(p.risk), 5, 0, 2 * Math.PI); g.fill(); g.stroke();
      }
    }
    move(e) {
      if (!this.points.length) return;
      const r = this.c.getBoundingClientRect(), { X, Y } = this.geom(), x = e.clientX - r.left;
      let best = 0, bd = Infinity;
      this.points.forEach((p, i) => { const d = Math.abs(X(p.t) - x); if (d < bd) { bd = d; best = i; } });
      if (bd > 40) { this.hover = null; this.tip.classList.add("hidden"); this.draw(); return; }
      this.hover = best;
      const p = this.points[best], f = (v, d = 0) => (v === undefined || v === null ? "–" : Number(v).toFixed(d));
      this.tip.innerHTML = `<div class="t">${f(p.t, 1)} s into the call</div>` +
        `<div class="r"><span class="k"><i style="background:${css("--cyan")}"></i>Overall risk</span><b>${f(p.risk)}</b></div>` +
        `<div class="r"><span class="k"><i style="background:${css("--series-2")};border-radius:50%"></i>Window score</span><b>${f(p.s)}</b></div>` +
        (p.llr !== undefined ? `<div class="r"><span>Detector score</span><b>${f(p.llr, 2)}</b></div>` : "") +
        (p.level ? `<div class="r"><span>Decision</span><b>${p.level}</b></div>` : "");
      this.tip.classList.remove("hidden");
      tipPos(this.tip, this.c.parentElement, X(p.t), Y(p.risk));
      this.draw();
    }
  }

  // ---------------------------------------------------------------- live waveform (real signal only)
  // Fed with the very same ~100 ms PCM chunks that are streamed to the detector (feed()). Each chunk is cut
  // into 12.5 ms blocks whose peak and RMS become one column, stamped with AUDIO time, so the picture is exact
  // even when the browser renders few frames. Drawing runs on animation frames, only while the stage is visible.
  // Columns keep the decision that was current when they were captured, so the history shows when a threat began.
  const STATE_RGB = { listen: [34, 211, 238], PASS: [34, 211, 238], CAL: [129, 140, 248], VERIFY: [245, 158, 11], HOLD: [244, 63, 94], idle: [100, 116, 139] };
  class LiveWave {
    constructor(canvas, opts = {}) {
      this.c = canvas; this.seconds = opts.seconds || 8; this.cols = []; this.state = "idle"; this.speech = false;
      this.live = false; this.raf = 0; this.audioT = 0; this.wall = 0; this.sr = 16000;
      if (window.ResizeObserver) new ResizeObserver(() => this.draw()).observe(canvas);
      document.addEventListener("visibilitychange", () => { if (!document.hidden && this.live) this.start(); });
    }
    attach(ctx) { this.sr = ctx.sampleRate; this.cols = []; this.audioT = 0; this.wall = performance.now(); this.live = true; this.start(); }
    detach() { this.live = false; cancelAnimationFrame(this.raf); this.raf = 0; this.draw(); }
    clear() { this.cols = []; this.audioT = 0; this.draw(); }
    setState(s) { this.state = s; }
    setSpeech(b) { this.speech = !!b; }
    feed(x) {
      const block = Math.max(32, Math.round(this.sr / 80));
      for (let o = 0; o < x.length; o += block) {
        const e = Math.min(x.length, o + block);
        let peak = 0, sum = 0;
        for (let i = o; i < e; i++) { const a = Math.abs(x[i]); if (a > peak) peak = a; sum += a * a; }
        this.audioT += (e - o) / this.sr;
        this.cols.push({ t: this.audioT, p: peak, r: Math.sqrt(sum / (e - o)), s: this.state, sp: this.speech });
      }
      this.wall = performance.now();
      const keep = this.audioT - this.seconds - 0.3;
      while (this.cols.length && this.cols[0].t < keep) this.cols.shift();
    }
    start() { if (!this.raf) this.raf = requestAnimationFrame(() => this.tick()); }
    tick() {
      this.raf = 0;
      if (!this.live || document.hidden) return;
      if (this.c.offsetParent !== null) this.draw();
      this.start();
    }
    static db(x) { return Math.max(0, Math.min(1, (20 * Math.log10(x + 1e-6) + 58) / 58)); }
    draw() {
      if (!this.c.clientWidth) return;
      const { g, w, h } = fitCanvas(this.c);
      // smooth scroll between chunks: advance with the wall clock for at most one chunk, lagging one chunk behind
      const now = this.audioT + (this.live ? Math.min(0.1, (performance.now() - this.wall) / 1000) - 0.1 : 0);
      g.clearRect(0, 0, w, h);
      const mid = Math.round(h / 2) + 0.5, half = h / 2 - 14, pxms = w / (this.seconds * 1000), step = 3;
      g.strokeStyle = "rgba(148,163,184,.14)"; g.lineWidth = 1;
      g.beginPath(); g.moveTo(0, mid); g.lineTo(w, mid); g.stroke();
      // seconds ticks along the bottom edge
      g.fillStyle = "rgba(148,163,184,.4)"; g.font = `10px ${css("--mono")}`; g.textAlign = "center";
      for (let s = 1; s < this.seconds; s++) { const x = w - s * 1000 * pxms; g.fillRect(Math.round(x), h - 6, 1, 4); if (s % 2 === 0) g.fillText(`-${s}s`, x, h - 9); }
      if (!this.cols.length) return;
      // bucket columns into 2px bars with 1px gaps
      const buckets = new Map();
      for (const c of this.cols) {
        const x = w - (now - c.t) * 1000 * pxms; if (x < -step || x > w) continue;
        const k = Math.floor(x / step), b = buckets.get(k);
        if (!b) buckets.set(k, { p: c.p, r: c.r, s: c.s, sp: c.sp });
        else { b.p = Math.max(b.p, c.p); b.r = Math.max(b.r, c.r); if (c.s === "HOLD" || (c.s === "VERIFY" && b.s !== "HOLD")) b.s = c.s; b.sp = b.sp || c.sp; }
      }
      const groups = new Map();
      for (const [k, b] of buckets) {
        const key = `${b.s}|${b.sp ? 1 : 0}`;
        if (!groups.has(key)) groups.set(key, { outer: new Path2D(), inner: new Path2D(), s: b.s, sp: b.sp });
        const gr = groups.get(key), x = k * step;
        const ho = Math.max(1, LiveWave.db(b.p) * half), hi = Math.max(1, LiveWave.db(b.r) * half * 0.92);
        gr.outer.rect(x, mid - ho, step - 1, ho * 2);
        gr.inner.rect(x, mid - hi, step - 1, hi * 2);
      }
      const glow = !reduced();
      for (const gr of groups.values()) {
        const [r, gg, bb] = STATE_RGB[gr.s] || STATE_RGB.listen, a = gr.sp ? 1 : 0.42;
        g.fillStyle = `rgba(${r},${gg},${bb},${0.22 * a})`; g.fill(gr.outer);
        g.save();
        if (glow && gr.sp) { g.shadowColor = `rgba(${r},${gg},${bb},.55)`; g.shadowBlur = 10; }
        g.fillStyle = `rgba(${r},${gg},${bb},${0.95 * a})`; g.fill(gr.inner);
        g.restore();
      }
      // "now" cursor at the right edge while audio is flowing
      if (this.live) {
        const [r, gg, bb] = STATE_RGB[this.state] || STATE_RGB.listen;
        const grad = g.createLinearGradient(0, 0, 0, h);
        grad.addColorStop(0, `rgba(${r},${gg},${bb},0)`); grad.addColorStop(0.5, `rgba(${r},${gg},${bb},.8)`); grad.addColorStop(1, `rgba(${r},${gg},${bb},0)`);
        g.fillStyle = grad; g.fillRect(w - 2, 0, 2, h);
      }
    }
  }

  // ---------------------------------------------------------------- segment lane: one cell per analysed window
  class SegLane {
    constructor(canvas, opts = {}) {
      this.c = canvas; this.wins = []; this.tmax = 20; this.hop = opts.hop || 1; this.hover = null; this.fixed = opts.duration || null;
      this.tip = document.createElement("div"); this.tip.className = "charttip hidden";
      canvas.parentElement.style.position = "relative";
      canvas.parentElement.parentElement.appendChild(this.tip);
      canvas.addEventListener("mousemove", (e) => this.move(e));
      canvas.addEventListener("mouseleave", () => { this.hover = null; this.tip.classList.add("hidden"); this.draw(); });
      if (window.ResizeObserver) new ResizeObserver(() => this.draw()).observe(canvas);
    }
    set(wins, clock) { this.wins = wins; this.tmax = this.fixed || Math.max(20, Math.ceil(Math.max(clock || 0, ...wins.map((w) => w.t), 0) / 5) * 5); this.draw(); }
    duration(d) { this.fixed = d; this.tmax = d; this.draw(); }
    color(w) {
      if (w.ws >= 0.5) return w.level === "HOLD" ? [244, 63, 94] : [245, 158, 11];
      return [16, 185, 129];
    }
    draw() {
      if (!this.c.clientWidth) return;
      const { g, w, h } = fitCanvas(this.c);
      g.clearRect(0, 0, w, h);
      const X = (t) => w * t / this.tmax;
      g.fillStyle = "rgba(148,163,184,.12)";
      for (let t = 5; t < this.tmax; t += 5) g.fillRect(Math.round(X(t)), 0, 1, h);
      this.wins.forEach((win, i) => {
        const [r, gg, b] = this.color(win), x0 = X(Math.max(0, win.t - this.hop)), x1 = X(win.t);
        const a = win.ws >= 0.5 ? 0.45 + 0.5 * win.ws : 0.18 + 0.3 * (1 - win.ws);
        g.fillStyle = `rgba(${r},${gg},${b},${i === this.hover ? Math.min(1, a + 0.25) : a})`;
        g.fillRect(x0 + 0.5, 3, Math.max(1.5, x1 - x0 - 1), h - 6);
        if (win.ws >= 0.5) { g.fillStyle = `rgb(${r},${gg},${b})`; g.fillRect(x0 + 0.5, h - 3, Math.max(1.5, x1 - x0 - 1), 3); }
      });
    }
    move(e) {
      if (!this.wins.length) return;
      const r = this.c.getBoundingClientRect(), t = (e.clientX - r.left) / r.width * this.tmax;
      let best = null, bd = Infinity;
      this.wins.forEach((w, i) => { const d = Math.abs(w.t - 0.5 - t); if (d < bd) { bd = d; best = i; } });
      if (best === null || bd > 1.5) { this.hover = null; this.tip.classList.add("hidden"); this.draw(); return; }
      this.hover = best; this.draw();
      const w = this.wins[best], f = (v, d = 2) => (v === undefined || v === null ? "–" : Number(v).toFixed(d));
      const verdictWord = w.ws >= 0.5 ? (w.level === "HOLD" ? "suspicious · call held" : "suspicious window") : "consistent with a real voice";
      this.tip.innerHTML = `<div class="t">Window ending ${f(w.t, 1)} s</div>` +
        `<div class="r"><span>Reading</span><b>${verdictWord}</b></div>` +
        `<div class="r"><span>AI-voice probability (this line)</span><b>${f(100 * w.ws, 0)}%</b></div>` +
        `<div class="r"><span>Detector score</span><b>${f(w.llr)}</b></div>` +
        (w.cut !== undefined && w.cut !== null ? `<div class="r"><span>Line cut-off</span><b>${f(w.cut)}</b></div>` : "") +
        (w.orig !== undefined && w.orig !== null ? `<div class="r"><span>Base detector score</span><b>${f(w.orig)}</b></div>` : "") +
        (w.proc ? `<div class="r"><span>Audio type</span><b>${esc(w.proc)}</b></div>` : "") +
        (w.risk !== undefined ? `<div class="r"><span>Call risk after it</span><b>${f(w.risk, 0)}</b></div>` : "");
      this.tip.classList.remove("hidden");
      const host = this.tip.parentElement, hr = host.getBoundingClientRect();
      tipPos(this.tip, host, e.clientX - hr.left, r.top - hr.top);
    }
  }

  // ---------------------------------------------------------------- offline renderers (file analysis)
  function fft(re, im) {                                   // in-place radix-2
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const ang = -2 * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang);
      for (let i = 0; i < n; i += len) {
        let cr = 1, ci = 0;
        for (let k = 0; k < len / 2; k++) {
          const a = i + k, b = a + len / 2;
          const tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr;
          re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
          const ncr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = ncr;
        }
      }
    }
  }
  const SPEC_STOPS = [[0, [10, 15, 25]], [0.32, [12, 46, 66]], [0.55, [14, 116, 144]], [0.78, [34, 211, 238]], [1, [236, 254, 255]]];
  function specColor(v) {
    for (let i = 1; i < SPEC_STOPS.length; i++) {
      const [p1, c1] = SPEC_STOPS[i], [p0, c0] = SPEC_STOPS[i - 1];
      if (v <= p1) { const f = (v - p0) / (p1 - p0); return c0.map((c, k) => Math.round(c + (c1[k] - c) * f)); }
    }
    return SPEC_STOPS[SPEC_STOPS.length - 1][1];
  }
  // Short-time spectrum of the decoded audio: 512-point frames, 10 ms hop, dB scaled, drawn into an image once.
  function spectrogram(samples, sr) {
    const N = 512, hop = Math.max(1, Math.round(sr / 100)), frames = Math.max(1, Math.floor((samples.length - N) / hop));
    const bins = N / 2, win = new Float32Array(N), out = new Float32Array(frames * bins);
    for (let i = 0; i < N; i++) win[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (N - 1));
    const re = new Float32Array(N), im = new Float32Array(N);
    let max = -Infinity;
    for (let f = 0; f < frames; f++) {
      const o = f * hop;
      for (let i = 0; i < N; i++) { re[i] = (samples[o + i] || 0) * win[i]; im[i] = 0; }
      fft(re, im);
      for (let b = 0; b < bins; b++) {
        const v = 10 * Math.log10(re[b] * re[b] + im[b] * im[b] + 1e-12);
        out[f * bins + b] = v; if (v > max) max = v;
      }
    }
    const img = new ImageData(frames, bins), floor = max - 80;
    for (let f = 0; f < frames; f++) for (let b = 0; b < bins; b++) {
      const v = Math.max(0, Math.min(1, (out[f * bins + b] - floor) / 80)), [r, g, bl] = specColor(v), p = ((bins - 1 - b) * frames + f) * 4;
      img.data[p] = r; img.data[p + 1] = g; img.data[p + 2] = bl; img.data[p + 3] = 255;
    }
    const cv = document.createElement("canvas"); cv.width = frames; cv.height = bins;
    cv.getContext("2d").putImageData(img, 0, 0);
    return { image: cv, frames, bins, nyquist: sr / 2 };
  }
  function drawImageFit(canvas, image, height) {
    const { g, w, h } = fitCanvas(canvas, height);
    g.clearRect(0, 0, w, h); g.imageSmoothingEnabled = true; g.drawImage(image, 0, 0, w, h);
  }
  // min/max per pixel of the whole file
  function drawOverview(canvas, samples, height, regions) {
    const { g, w, h } = fitCanvas(canvas, height);
    g.clearRect(0, 0, w, h);
    const mid = h / 2, per = Math.max(1, Math.floor(samples.length / w));
    let peak = 1e-6;
    for (let i = 0; i < samples.length; i += 7) peak = Math.max(peak, Math.abs(samples[i]));
    for (const r of regions || []) { g.fillStyle = r.color; g.fillRect(w * r.a, 0, Math.max(1, w * (r.b - r.a)), h); }
    g.strokeStyle = "rgba(148,163,184,.16)"; g.beginPath(); g.moveTo(0, mid + 0.5); g.lineTo(w, mid + 0.5); g.stroke();
    const path = new Path2D();
    for (let x = 0; x < w; x++) {
      let mn = 0, mx = 0;
      for (let i = x * per, e = Math.min(samples.length, i + per); i < e; i++) { const v = samples[i]; if (v < mn) mn = v; if (v > mx) mx = v; }
      const top = mid - (mx / peak) * (mid - 4), bot = mid - (mn / peak) * (mid - 4);
      path.rect(x, top, 1, Math.max(1, bot - top));
    }
    const grad = g.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, "rgba(103,232,249,.95)"); grad.addColorStop(0.5, "rgba(34,211,238,.9)"); grad.addColorStop(1, "rgba(103,232,249,.95)");
    g.fillStyle = grad; g.fill(path);
  }

  // ---------------------------------------------------------------- small SVG bar chart with hover
  // data: [{label, values: {seriesKey: number}, tip?}], series: [{key, name, color}]
  function barChart(el, data, series, opts = {}) {
    const H = opts.height || 220, W = el.clientWidth || 600, pad = { l: 36, r: 8, t: 10, b: 26 };
    const stacked = opts.stacked !== false;
    const totals = data.map((d) => stacked ? series.reduce((s, k) => s + (d.values[k.key] || 0), 0) : Math.max(...series.map((k) => d.values[k.key] || 0)));
    const top = Math.max(1, ...totals), step = niceStep(top / 4), nt = Math.max(1, Math.ceil(top / step)), ymax = step * nt;
    const iw = W - pad.l - pad.r, ih = H - pad.t - pad.b, bw = iw / Math.max(1, data.length);
    const barW = Math.max(3, Math.min(42, bw * 0.62)), Y = (v) => pad.t + ih * (1 - v / ymax);
    let s = `<svg class="chart" viewBox="0 0 ${W} ${H}" height="${H}" role="img" aria-label="${esc(opts.label || "bar chart")}"><g class="grid">`;
    for (let i = 0; i <= nt; i++) { const v = step * i, y = Y(v); s += `<line x1="${pad.l}" x2="${W - pad.r}" y1="${y}" y2="${y}"/><text x="${pad.l - 8}" y="${y + 4}" text-anchor="end">${fmtTick(v)}</text>`; }
    s += `</g>`;
    const every = Math.ceil(data.length / Math.max(1, Math.floor(iw / 64)));
    data.forEach((d, i) => {
      const cx = pad.l + bw * i + bw / 2;
      let y0 = Y(0), bars = "";
      series.forEach((k, j) => {
        const v = d.values[k.key] || 0; if (!v) return;
        if (stacked) {                                   // 2px surface gap between stacked segments
          const top = y0 - (ih * v / ymax);
          bars += `<rect class="bar" x="${cx - barW / 2}" y="${top + 1}" width="${barW}" height="${Math.max(1, y0 - top - 2)}" rx="2" fill="${k.color}"/>`;
          y0 = top;
        } else {
          const n = series.length, w2 = barW / n, x = cx - barW / 2 + j * w2;
          bars += `<rect class="bar" x="${x + 1}" y="${Y(v)}" width="${Math.max(1, w2 - 2)}" height="${Y(0) - Y(v)}" rx="2" fill="${k.color}"/>`;
        }
      });
      s += `<g data-i="${i}"><rect class="hit" x="${pad.l + bw * i}" y="${pad.t}" width="${bw}" height="${ih}"/>${bars}</g>`;
      if (i % every === 0) s += `<text x="${cx}" y="${H - 8}" text-anchor="middle">${esc(d.label)}</text>`;
    });
    s += `<g class="axis"><line x1="${pad.l}" x2="${W - pad.r}" y1="${Y(0) + .5}" y2="${Y(0) + .5}"/></g></svg>`;
    el.style.position = "relative";
    el.innerHTML = s + `<div class="charttip hidden"></div>`;
    const tip = el.querySelector(".charttip");
    el.querySelectorAll("g[data-i]").forEach((gEl) => {
      gEl.addEventListener("mouseenter", () => {
        const d = data[Number(gEl.dataset.i)];
        tip.innerHTML = `<div class="t">${esc(d.tip || d.label)}</div>` + series.map((k) =>
          `<div class="r"><span class="k"><i style="background:${k.color}"></i>${esc(k.name)}</span><b>${fmtNum(d.values[k.key] || 0, opts.fmt)}</b></div>`).join("");
        tip.classList.remove("hidden");
        const i = Number(gEl.dataset.i), sc = el.clientWidth / W;
        tipPos(tip, el, (pad.l + bw * i + bw / 2) * sc, Y(totals[i]) * sc);
      });
      gEl.addEventListener("mouseleave", () => tip.classList.add("hidden"));
    });
  }
  function niceStep(v) {                                 // 1, 2, 5 x 10^k, never below 1 (counts and ms)
    if (v <= 1) return 1;
    const m = Math.pow(10, Math.floor(Math.log10(v))), n = v / m;
    return (n <= 1 ? 1 : n <= 2 ? 2 : n <= 5 ? 5 : 10) * m;
  }
  function fmtTick(v) { return v >= 1000 ? `${(v / 1000).toFixed(v >= 10000 ? 0 : 1)}k` : Number.isInteger(v) ? String(v) : v.toFixed(1); }
  function fmtNum(v, f) { return f ? f(v) : (Number.isInteger(v) ? String(v) : Number(v).toFixed(1)); }

  window.VSUI = { setTheme, theme, icon, hydrate, mountTopbar, setActive, loadStatus, toast, confirmDialog, VERDICT, setGauge, RiskChart,
                  LiveWave, SegLane, spectrogram, drawImageFit, drawOverview, fitCanvas, barChart, css, esc };
})();
