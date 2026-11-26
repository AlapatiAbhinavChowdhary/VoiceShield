"""Evaluation center data: only measured results, each labelled with the kind of evidence it is.

Classes (shown in the console):
  heldout     offline test sets never used for training, calibration or model selection (Step 8c, Kaggle)
  controlled  controlled suites built from real recordings with synthetic conditions (robustness suite, lab)
  dev         development sanity checks written / used while designing (scam_eval.json)
  blind       written before the method was coded, evaluated once (scam_eval_blind.json)
  live        measured on this server during operation (audit log)
"""
import csv
import json
from pathlib import Path

import numpy as np

from .config import ROOT

EVAL = ROOT / "eval"
STEP8C = ROOT.parent / "recordings" / "step8c_results_from_log.csv"


def _det(points):
    """ROC points [far, frr] (0-1) -> DET coordinates (probit), clipped to 0.1% .. 99.9%."""
    from scipy.stats import norm
    out = []
    for far, frr in points:
        a, b = min(max(far, 1e-3), 0.999), min(max(frr, 1e-3), 0.999)
        out.append([round(float(norm.ppf(a)), 3), round(float(norm.ppf(b)), 3)])
    return out


def step8c_table():
    if not STEP8C.exists():
        return None
    rows = {}
    with STEP8C.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d = rows.setdefault(r["test set"], {"set": r["test set"]})
            d[r["scorer"]] = {"eer": float(r["EER_%"]), "auc": float(r["AUC"]),
                              "far": float(r["FAR_single_%"]), "frr": float(r["MR_single_%"])}
    return {"class": "heldout", "source": "recordings/step8c_results_from_log.csv (Step 8c, 20 Nov 2026)",
            "rows": list(rows.values())}


def _clean(x):
    """NaN / inf (e.g. a partial suite with no spoof windows yet) -> None, so the response stays valid JSON."""
    if isinstance(x, float):
        return x if np.isfinite(x) else None
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_clean(v) for v in x]
    return x


def robustness():
    p = EVAL / "robustness_report.json"
    if not p.exists():
        return None
    r = _clean(json.loads(p.read_text(encoding="utf-8")))
    for v in r["window_level"].values():
        v["det"] = _det([pt for pt in v["points"] if None not in pt])
    for rule in r["rules"]:
        rule.pop("calls", None)
    r["class"] = "controlled"
    return r


def scam():
    p = EVAL / "scam_report.json"
    if not p.exists():
        return None
    sets = json.loads(p.read_text(encoding="utf-8"))
    klass = {"scam_eval.json": "dev", "scam_eval_heldout.json": "heldout-seen", "scam_eval_blind.json": "blind"}
    out = []
    for s in sets:
        out.append({"set": s["set"], "class": klass.get(s["set"], "dev"), "about": s["about"],
                    "n_scam": s["n_scam"], "n_benign": s["n_benign"],
                    "engines": {k: {m: v[m] for m in ("tp", "fp", "fn", "tn", "precision", "recall", "f1")}
                                for k, v in s["engines"].items()}})
    return out


def live_latency(audit) -> dict:
    with audit.lock:
        rows = audit.db.execute("SELECT latency_ms FROM windows WHERE latency_ms IS NOT NULL ORDER BY rowid DESC LIMIT 3000").fetchall()
        n_calls = audit.db.execute("SELECT COUNT(*) FROM calls WHERE mode='monitor'").fetchone()[0]
    lat = np.array([r[0] for r in rows], float)
    if not lat.size:
        return {"class": "live", "windows": 0, "calls": n_calls}
    return {"class": "live", "windows": int(lat.size), "calls": n_calls, "median_ms": round(float(np.median(lat)), 1),
            "p95_ms": round(float(np.percentile(lat, 95)), 1), "max_ms": round(float(lat.max()), 1)}
