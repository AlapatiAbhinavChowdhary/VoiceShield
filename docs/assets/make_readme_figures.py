"""Builds the README graphics (SVG) from real project data.

    voiceshield\\.venv\\Scripts\\python docs\\assets\\make_readme_figures.py

- banner.svg      : the real waveform of the demo scam call, coloured by the verdict the live system gave it
- stats.svg       : headline results
- chart-step8c.svg: EER before / after the Step 8c back-end on unseen test sets (Kaggle, 20 Nov 2026)
- chart-layers.svg: linear-probe EER for every layer of the frozen detector (recordings/step8c_layers_from_log.csv)
Every SVG carries its own dark card, so it reads the same on GitHub's light and dark themes.
Chart colours: validated categorical slots on the dark card (cyan #0EA3BE, indigo #7B7FF2); text never uses series colours.
"""
import csv
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
FONT = "'Segoe UI', Inter, -apple-system, 'Helvetica Neue', Arial, sans-serif"
MONO = "'Cascadia Mono', 'JetBrains Mono', Consolas, monospace"
BG, CARD, LINE = "#080b12", "#0f1521", "rgba(148,163,184,0.16)"
INK, INK2, MUTED = "#f8fafc", "#a3b1c6", "#7c8ba1"
CYAN, INDIGO, GOOD, WARN, CRIT = "#0ea3be", "#7b7ff2", "#10b981", "#f59e0b", "#f43f5e"
SHIELD = ("M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72"
          "a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z")


def card(w, h, body, rx=18):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" font-family="{FONT}">'
            f'<defs><linearGradient id="bgg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#0d1320"/>'
            f'<stop offset="1" stop-color="{BG}"/></linearGradient>'
            f'<radialGradient id="glow" cx="0.12" cy="0" r="0.7"><stop offset="0" stop-color="#22d3ee" stop-opacity="0.13"/>'
            f'<stop offset="1" stop-color="#22d3ee" stop-opacity="0"/></radialGradient></defs>'
            f'<rect x="0.5" y="0.5" width="{w - 1}" height="{h - 1}" rx="{rx}" fill="url(#bgg)" stroke="{LINE}"/>'
            f'<rect x="0.5" y="0.5" width="{w - 1}" height="{h - 1}" rx="{rx}" fill="url(#glow)"/>{body}</svg>')


def text(x, y, s, size=14, fill=INK, weight=400, anchor="start", family=None, ls=0):
    fam = f' font-family="{family}"' if family else ""
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" font-weight="{weight}" text-anchor="{anchor}"'
            f' letter-spacing="{ls}"{fam}>{s}</text>')


# ---------------------------------------------------------------- banner: real waveform of the demo scam call
def banner():
    wav, sr = sf.read(ROOT / "voiceshield/samples/scam_call_tts.wav", dtype="float32", always_2d=True)
    x = wav.mean(1)[: int(sr * 14)]
    n, w, h = 150, 1280, 340
    seg = np.array_split(np.abs(x), n)
    env = np.array([s.max() for s in seg])
    env = env / env.max()
    x0, x1, mid, amp = 640, 1230, 200, 92
    bw = (x1 - x0) / n
    bars = []
    for i, e in enumerate(env):
        t = (i + 0.5) / n * 14
        col = CRIT if t >= 4.4 else WARN if t >= 3.3 else "#22d3ee"      # verdict the live system gave at that time
        hgt = max(2, e * amp)
        bars.append(f'<rect x="{x0 + i * bw:.1f}" y="{mid - hgt:.1f}" width="{bw * 0.62:.1f}" height="{2 * hgt:.1f}" rx="1" fill="{col}" opacity="0.92"/>')
    marks = ""
    for t, lab, col in [(3.3, "VERIFY 3.3 s", WARN), (4.4, "HOLD 4.4 s", CRIT)]:
        xx = x0 + t / 14 * (x1 - x0)
        marks += (f'<line x1="{xx:.1f}" x2="{xx:.1f}" y1="{mid - amp - 16}" y2="{mid + amp + 6}" stroke="{col}" stroke-dasharray="3 4" opacity="0.8"/>'
                  + text(xx + 6, mid - amp - 6 if lab.startswith("HOLD") else mid - amp - 22, lab, 12, INK2, 600, family=MONO))
    logo = (f'<g transform="translate(64,70)"><rect width="64" height="64" rx="16" fill="#22d3ee"/>'
            f'<path transform="translate(10,10) scale(1.83)" d="{SHIELD}" fill="none" stroke="#04131a" stroke-width="1.9" stroke-linejoin="round"/>'
            f'<path transform="translate(10,10) scale(1.83)" d="M8 12.5h1.4l1.3-2.8 1.9 5.3 1.4-3.3H16" fill="none" stroke="#04131a" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"/></g>')
    body = (logo + text(148, 112, "VoiceShield", 46, INK, 700, ls=-1)
            + text(66, 186, "Real-time detection &amp; prevention", 26, INK, 600)
            + text(66, 220, "of voice-cloning scams in phone calls", 26, INK2, 500)
            + text(66, 268, "Smart India Hackathon 2026 · Problem SIH26104", 14, MUTED, 600)
            + text(66, 290, "AI-voice detector · voiceprint · scam-language risk engine", 14, MUTED, 500)
            + f'<rect x="{x0 - 20}" y="58" width="{x1 - x0 + 40}" height="268" rx="14" fill="#0a0f19" stroke="{LINE}"/>'
            + f'<line x1="{x0}" x2="{x1}" y1="{mid}" y2="{mid}" stroke="rgba(148,163,184,0.18)"/>' + "".join(bars) + marks
            + text(x0, 314, "Real waveform of the demo scam call, coloured by the live verdict · listening → suspicious → held", 12, MUTED, 500))
    (OUT / "banner.svg").write_text(card(w, h, body, 22), encoding="utf-8")


# ---------------------------------------------------------------- headline numbers
def stats():
    items = [("4.4 s", "to HOLD a robot-voice", "scam call, payment blocked", CRIT),
             ("9.7% → 4.1%", "mean error on 13 unseen", "test sets (Step 8c)", "#22d3ee"),
             ("VERIFY → HOLD", "real ElevenLabs clone,", "held after 5.6 s", WARN),
             ("0 bytes", "of call audio or transcript", "ever stored on disk", GOOD)]
    w, h, cw = 1280, 150, 1280 / 4
    body = ""
    for i, (big, l1, l2, col) in enumerate(items):
        x = i * cw
        if i:
            body += f'<line x1="{x}" x2="{x}" y1="28" y2="{h - 28}" stroke="{LINE}"/>'
        body += (f'<rect x="{x + 36}" y="36" width="4" height="78" rx="2" fill="{col}"/>'
                 + text(x + 54, 72, big, 30, INK, 700, ls=-0.5) + text(x + 54, 98, l1, 14, INK2, 500) + text(x + 54, 117, l2, 14, INK2, 500))
    (OUT / "stats.svg").write_text(card(w, h, body), encoding="utf-8")


# ---------------------------------------------------------------- Step 8c before / after (paired horizontal bars)
EVAL = [("Hindi/Telugu clones · clean", 0.0, 0.0), ("Hindi/Telugu clones · phone line", 2.0, 0.0),
        ("Hindi/Telugu clones · noise suppression", 5.0, 1.33), ("Hindi/Telugu clones · Bluetooth CVSD", 10.0, 4.0),
        ("Hindi/Telugu clones · Opus (VoIP)", 4.67, 1.33), ("Premium clones · clean", 8.23, 2.95),
        ("Premium clones · noise suppression", 33.97, 12.66), ("Premium clones · Opus (VoIP)", 45.15, 16.67),
        ("In-the-Wild deepfakes", 4.67, 4.67), ("Mean of 13 test sets", 9.73, 4.05)]


def chart_step8c():
    w, top, row, lx, x0, x1 = 1280, 118, 44, 56, 420, 1150
    h = top + row * len(EVAL) + 70
    X = lambda v: x0 + (x1 - x0) * v / 50            # noqa: E731
    body = (text(56, 50, "Everyday audio processing breaks the detector, and Step 8c repairs most of it", 22, INK, 700)
            + text(56, 78, "Equal error rate (%) on unseen test sets, lower is better · Kaggle T4 · detector frozen, 1,050-number back-end on its 25 layers", 14, INK2, 500)
            + f'<rect x="{x1 - 360}" y="92" width="12" height="12" rx="3" fill="{INDIGO}"/>' + text(x1 - 342, 103, "Original detector", 13, INK2, 500)
            + f'<rect x="{x1 - 200}" y="92" width="12" height="12" rx="3" fill="{CYAN}"/>' + text(x1 - 182, 103, "With Step 8c (in use)", 13, INK2, 500))
    for v in range(0, 51, 10):
        body += (f'<line x1="{X(v)}" x2="{X(v)}" y1="{top}" y2="{top + row * len(EVAL)}" stroke="rgba(148,163,184,{0.22 if v == 0 else 0.08})"/>'
                 + text(X(v), top + row * len(EVAL) + 22, f"{v}%", 12, MUTED, 500, "middle"))
    for i, (name, a, b) in enumerate(EVAL):
        y = top + i * row + 8
        last = i == len(EVAL) - 1
        if last:
            body += f'<rect x="{lx - 12}" y="{y - 8}" width="{x1 - lx + 70}" height="{row}" rx="8" fill="rgba(34,211,238,0.06)"/>'
        body += text(lx, y + 19, name, 14, INK, 700 if last else 500)
        for j, (v, col) in enumerate([(a, INDIGO), (b, CYAN)]):
            yy = y + j * 15
            body += f'<rect x="{x0}" y="{yy + 2}" width="{max(2, X(v) - x0):.1f}" height="11" rx="3" fill="{col}"/>'
            body += text(X(v) + 8, yy + 12, f"{v:.2f}%" if last else f"{v:.1f}%", 12, INK if j else INK2, 600 if j else 500, family=MONO)
    body += text(56, h - 22, "Premium clones = ElevenLabs + Kokoro, never seen in training. Processed test sets use simulated processing; see the paper for the full table.", 12, MUTED, 500)
    (OUT / "chart-step8c.svg").write_text(card(w, h, body), encoding="utf-8")


# ---------------------------------------------------------------- layer analysis (one axis, three series + head markers)
def chart_layers():
    rows = list(csv.DictReader(open(ROOT / "recordings/step8c_layers_from_log.csv", encoding="utf-8")))
    lay = [r for r in rows if r["layer"].startswith("layer")]
    head = next(r for r in rows if r["layer"] == "original head")
    series = [("dev mean", "dev mean", CYAN, ""), ("test mean", "test mean", INDIGO, "6 5"),
              ("held-out Bluetooth CVSD", "Bluetooth CVSD (dev)", WARN, "2 4")]
    w, h, pl, pr, pt, pb = 1280, 470, 96, 150, 120, 64
    X = lambda l: pl + (w - pl - pr) * l / 26           # noqa: E731
    Y = lambda v: pt + (h - pt - pb) * (1 - v / 27)     # noqa: E731
    body = (text(56, 50, "Where the evidence survives processing: inside the frozen detector", 22, INK, 700)
            + text(56, 78, "EER (%) of a simple probe on each of the 25 layers · the middle layers keep the clues that the final read-out loses", 14, INK2, 500))
    for v in (0, 5, 10, 15, 20, 25):
        body += f'<line x1="{pl}" x2="{X(26)}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="rgba(148,163,184,{0.22 if v == 0 else 0.08})"/>' + text(pl - 12, Y(v) + 4, f"{v}%", 12, MUTED, 500, "end")
    for l in range(0, 25, 4):
        body += text(X(l), h - pb + 24, str(l), 12, MUTED, 500, "middle")
    body += text(X(26), h - pb + 24, "head", 12, MUTED, 600, "middle") + text((pl + X(24)) / 2, h - 16, "layer (0 = first block input, 24 = last block)", 12, MUTED, 500, "middle")
    body += f'<line x1="{X(25)}" x2="{X(25)}" y1="{pt}" y2="{h - pb}" stroke="rgba(148,163,184,0.25)"/>'
    body += f'<rect x="{X(5) - 10}" y="{pt}" width="{X(11) - X(5) + 20}" height="{h - pt - pb}" rx="8" fill="rgba(34,211,238,0.06)"/>'
    body += text((X(5) + X(11)) / 2, pt + 18, "most robust", 12, INK2, 600, "middle")
    for key, label, col, dash in series:
        pts = " ".join(f"{X(i):.1f},{Y(float(r[key])):.1f}" for i, r in enumerate(lay))
        body += f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.4" stroke-linejoin="round" stroke-dasharray="{dash}"/>'
        for i, r in enumerate(lay):
            body += f'<circle cx="{X(i):.1f}" cy="{Y(float(r[key])):.1f}" r="3.2" fill="{col}" stroke="{CARD}" stroke-width="1.5"/>'
        hv = float(head[key])
        body += f'<circle cx="{X(26)}" cy="{Y(hv):.1f}" r="5.5" fill="{col}" stroke="{CARD}" stroke-width="2"/>'
        body += text(X(26) + 12, Y(hv) + 4, f"{hv:.1f}%", 12, INK2, 600, family=MONO)
    lx = w - pr - 360
    for i, (_, label, col, dash) in enumerate(series):
        body += (f'<line x1="{lx + i * 130}" x2="{lx + i * 130 + 22}" y1="99" y2="99" stroke="{col}" stroke-width="2.4" stroke-dasharray="{dash}"/>'
                 + text(lx + i * 130 + 28, 103, label, 12, INK2, 500))
    (OUT / "chart-layers.svg").write_text(card(w, h, body), encoding="utf-8")


if __name__ == "__main__":
    banner(); stats(); chart_step8c(); chart_layers()
    for f in sorted(OUT.glob("*.svg")):
        print(f.name, f.stat().st_size // 1024, "KB")
