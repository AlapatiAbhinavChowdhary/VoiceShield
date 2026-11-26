"""Replay the robustness suite's stored detector windows through decision rules (no models needed).

Each recording x condition is one call: the voice-only decision of the risk engine at the default threshold.
The baseline is the shipped RiskEngine. Candidates change only HOW MUCH CONSISTENT evidence a HOLD needs,
never the threshold or the window score, so a clean clone call still produces exactly the same windows.
Writes eval/robustness_report.json (used by the console's Evaluation view).
"""
import json
import sys
from dataclasses import replace
from pathlib import Path
from statistics import median

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from server.config import load_config  # noqa: E402
from server.models import LayerBackend  # noqa: E402
from server.risk import RiskEngine  # noqa: E402

DEGRADED = ("fair", "poor")


class Base(RiskEngine):
    name, note = "baseline", "shipped rule: HOLD when voice risk >= 75 and 3 of the last 5 windows are flagged"

    def feed(self, llr, q):
        self.update(llr)


class Persist(Base):
    """In degraded audio (most of the last 5 windows fair/poor), a voice HOLD needs 4 of 5 flagged windows."""
    name, note = "quality-aware persistence", "degraded audio: HOLD needs 4 of 5 flagged windows (else 3 of 5)"

    def __init__(self, cfg, thr):
        super().__init__(cfg, thr)
        self.base_cfg, self.strict = cfg, replace(cfg, hold_k=cfg.hold_k + 1)
        self.qs = []

    def feed(self, llr, q):
        self.qs.append(q)
        self.update(llr)

    def state(self):
        last = self.qs[-self.base_cfg.hold_n:]
        self.cfg = self.strict if sum(x in DEGRADED for x in last) * 2 > len(last) else self.base_cfg
        try:
            return super().state()
        finally:
            self.cfg = self.base_cfg


class Confirm(Base):
    """A voice HOLD must be confirmed by the next window too (one extra second)."""
    name, note = "confirmed HOLD", "voice HOLD condition must hold on two consecutive windows"

    def __init__(self, cfg, thr):
        super().__init__(cfg, thr)
        self.base_cfg, self.block = cfg, replace(cfg, hold_at=101.0)
        self.prev = False

    def state(self):
        c = self.base_cfg
        now = self.voice >= c.hold_at and sum(self.recent) >= c.hold_k
        self.cfg = self.base_cfg if (now and self.prev) or self.verdict == 2 else self.block
        self.prev = now
        try:
            return super().state()
        finally:
            self.cfg = self.base_cfg


class Median3(Base):
    """Median of the last 3 window scores (llr) before the window score: damps single-window spikes."""
    name, note = "median-of-3 smoothing", "each window uses the median score of itself and the 2 before it"

    def __init__(self, cfg, thr):
        super().__init__(cfg, thr)
        self.hist = []

    def feed(self, llr, q):
        self.hist.append(llr)
        self.update(float(np.median(self.hist[-3:])))


class PersistConfirm(Persist):
    name, note = "persistence + confirmation", "both of the above"

    def __init__(self, cfg, thr):
        super().__init__(cfg, thr)
        self.prev = False

    def state(self):
        last = self.qs[-self.base_cfg.hold_n:]
        cfg = self.strict if sum(x in DEGRADED for x in last) * 2 > len(last) else self.base_cfg
        now = self.voice >= cfg.hold_at and sum(self.recent) >= cfg.hold_k
        self.cfg = cfg if (now and self.prev) or self.verdict == 2 else replace(cfg, hold_at=101.0)
        self.prev = now
        try:
            return RiskEngine.state(self)
        finally:
            self.cfg = self.base_cfg


class CapDegraded(Base):
    """Degraded audio (most of the last 5 windows fair/poor): voice evidence alone escalates to VERIFY, not HOLD.
    Scam language or an identity mismatch can still raise HOLD; clean audio is unchanged."""
    name, note = "degraded audio caps voice-only at VERIFY", "noisy audio: voice alone gives VERIFY (OTP / call-back), not HOLD"

    def __init__(self, cfg, thr):
        super().__init__(cfg, thr)
        self.base_cfg, self.block = cfg, replace(cfg, hold_at=101.0)
        self.qs = []

    def feed(self, llr, q):
        self.qs.append(q)
        self.update(llr)

    def state(self):
        last = self.qs[-self.base_cfg.hold_n:]
        degraded = sum(x in DEGRADED for x in last) * 2 > len(last)
        self.cfg = self.block if degraded and self.verdict < 2 else self.base_cfg
        try:
            return super().state()
        finally:
            self.cfg = self.base_cfg


RULES = [Base, Persist, Confirm, Median3, PersistConfirm, CapDegraded]


def simulate(rule, cfg, thr, wins):
    e = rule(cfg.risk, thr)
    first = {}
    for w in wins:
        e.feed(w["llr"], w["q"])
        st = e.state()
        for lv in ("VERIFY", "HOLD"):
            if st["verdict"] in (lv, "HOLD") and lv not in first:
                first[lv] = w["t"]
    return {"verdict": ["PASS", "VERIFY", "HOLD"][e.verdict], "first_hold": first.get("HOLD"), "first_alert": first.get("VERIFY")}


def roc(gen, spf):
    """Window-level ROC from detector scores (higher = more likely spoof); EER by linear interpolation."""
    g, s = np.sort(np.asarray(gen)), np.sort(np.asarray(spf))
    ts = np.unique(np.concatenate([g, s]))
    far = np.array([np.mean(g >= t) for t in ts])                # genuine flagged
    frr = np.array([np.mean(s < t) for t in ts])                 # spoof missed
    i = int(np.argmin(np.abs(far - frr)))
    auc = float(np.mean([np.mean(x > g) + 0.5 * np.mean(x == g) for x in s]))
    step = max(1, len(ts) // 60)
    return {"eer": round(float((far[i] + frr[i]) / 2) * 100, 2), "auc": round(auc, 4),
            "points": [[round(float(a), 4), round(float(b), 4)] for a, b in zip(far[::step], frr[::step])]}


def report(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cfg = load_config()
    thr = LayerBackend(cfg.models / cfg.backend_file).threshold
    conds = data["conditions"]
    out = {"built": data["built"], "detector_mode": data["detector_mode"], "threshold": round(thr, 3),
           "recordings": [{k: r[k] for k in ("id", "label", "kind")} for r in data["recordings"]],
           "conditions": conds, "window_level": {}, "rules": []}
    for c in conds:
        gen = [w["llr"] for r in data["recordings"] if r["label"] == "genuine" for w in r["conditions"][c]]
        spf = [w["llr"] for r in data["recordings"] if r["label"] == "spoof" for w in r["conditions"][c]]
        q = [w["q"] for r in data["recordings"] for w in r["conditions"][c]]
        out["window_level"][c] = {**roc(gen, spf), "n_genuine": len(gen), "n_spoof": len(spf),
                                  "far_at_threshold": round(100 * float(np.mean(np.asarray(gen) >= thr)), 2),
                                  "frr_at_threshold": round(100 * float(np.mean(np.asarray(spf) < thr)), 2),
                                  "quality": {k: q.count(k) for k in ("good", "fair", "poor")}}
    for rule in RULES:
        calls = []
        for r in data["recordings"]:
            for c in conds:
                res = simulate(rule, cfg, thr, r["conditions"][c])
                calls.append({"rec": r["id"], "label": r["label"], "cond": c, **res})
        g = [x for x in calls if x["label"] == "genuine"]
        s = [x for x in calls if x["label"] == "spoof"]
        holds = [x["first_hold"] for x in s if x["first_hold"] is not None]
        out["rules"].append({"name": rule.name, "note": rule.note,
                             "genuine_calls": len(g), "false_hold": sum(x["verdict"] == "HOLD" for x in g),
                             "false_alert": sum(x["verdict"] != "PASS" for x in g),
                             "spoof_calls": len(s), "spoof_hold": sum(x["verdict"] == "HOLD" for x in s),
                             "spoof_stopped": sum(x["verdict"] != "PASS" for x in s),
                             "median_time_to_hold_s": round(median(holds), 1) if holds else None, "calls": calls})
    (ROOT / "eval" / "robustness_report.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"threshold {thr:.3f} · {len(data['recordings'])} recordings x {len(conds)} conditions\n")
    print(f"{'condition':14s} {'EER%':>6} {'AUC':>6} {'FAR@thr':>8} {'FRR@thr':>8}  quality g/f/p")
    for c, v in out["window_level"].items():
        print(f"{c:14s} {v['eer']:6.2f} {v['auc']:6.3f} {v['far_at_threshold']:8.2f} {v['frr_at_threshold']:8.2f}  "
              f"{v['quality']['good']}/{v['quality']['fair']}/{v['quality']['poor']}")
    print(f"\n{'rule':28s} {'falseHOLD':>9} {'falseAlert':>10} {'spoofHOLD':>9} {'spoofStopped':>12} {'t_hold':>7}")
    for r in out["rules"]:
        print(f"{r['name']:28s} {r['false_hold']:4d}/{r['genuine_calls']:<4d} {r['false_alert']:5d}/{r['genuine_calls']:<4d}"
              f" {r['spoof_hold']:4d}/{r['spoof_calls']:<4d} {r['spoof_stopped']:7d}/{r['spoof_calls']:<4d} {r['median_time_to_hold_s']}")
    # honest split: choose on dev-split recordings only, then check on the held-out recordings (demo, mic, clones)
    dev = lambda rid: rid.startswith(("dev_", "hume_"))  # noqa: E731
    out["split"] = []
    for r in out["rules"]:
        row = {"name": r["name"]}
        for part, sel in (("dev", True), ("heldout", False)):
            cs = [x for x in r["calls"] if dev(x["rec"]) == sel]
            row[part] = {"genuine": sum(x["label"] == "genuine" for x in cs),
                         "false_hold": sum(x["label"] == "genuine" and x["verdict"] == "HOLD" for x in cs),
                         "spoof": sum(x["label"] == "spoof" for x in cs),
                         "spoof_hold": sum(x["label"] == "spoof" and x["verdict"] == "HOLD" for x in cs),
                         "spoof_stopped": sum(x["label"] == "spoof" and x["verdict"] != "PASS" for x in cs)}
        out["split"].append(row)
    (ROOT / "eval" / "robustness_report.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("\nsplit: choose on dev recordings -> check on held-out recordings")
    for row in out["split"]:
        d, h = row["dev"], row["heldout"]
        print(f"  {row['name']:42s} dev falseHOLD {d['false_hold']}/{d['genuine']} spoofHOLD {d['spoof_hold']}/{d['spoof']}"
              f" | held-out falseHOLD {h['false_hold']}/{h['genuine']} spoofHOLD {h['spoof_hold']}/{h['spoof']} stopped {h['spoof_stopped']}/{h['spoof']}")
    base = {(x["rec"], x["cond"]): x["verdict"] for x in out["rules"][0]["calls"]}
    for r in out["rules"][1:]:
        lost = [f"{x['rec']}/{x['cond']} {base[(x['rec'], x['cond'])]}->{x['verdict']}" for x in r["calls"]
                if x["label"] == "spoof" and x["verdict"] != base[(x["rec"], x["cond"])]]
        fixed = [f"{x['rec']}/{x['cond']}" for x in r["calls"] if x["label"] == "genuine"
                 and base[(x["rec"], x["cond"])] == "HOLD" and x["verdict"] != "HOLD"]
        print(f"\n{r['name']}: spoof calls weakened: {lost or 'none'}\n  genuine false HOLDs removed: {fixed or 'none'}")


if __name__ == "__main__":
    report(ROOT / "eval" / "robustness_windows.json")
