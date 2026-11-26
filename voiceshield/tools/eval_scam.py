"""Evaluate the scam-language rules on the labelled call set (tests/data/scam_eval.json).

    python tools/eval_scam.py            prints precision / recall / F1 and every disagreement
    python tools/eval_scam.py --json     the same numbers as JSON (used by the console's model page)

Engines compared: the original keyword rules (from git tag baseline-2026-11-26, if available), the current
keyword rules (with the added Hinglish phrasings), and the current rules with the context layer.
"""
import argparse
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from server import scam  # noqa: E402

DATA = ROOT / "tests" / "data" / "scam_eval.json"
FLAG_AT = 0.5


def run(calls, make) -> dict:
    rows, tp = [], 0
    for c in calls:
        eng = make()
        for u in c["utterances"]:
            r = eng.add(u)
        flagged = r["score"] >= FLAG_AT
        rows.append({"id": c["id"], "label": c["label"], "kind": c["kind"], "score": round(r["score"], 3), "flagged": flagged,
                     "patterns": r.get("patterns", []), "suppressed": [s["phrase"] for s in r.get("suppressed", [])]})
    tp = sum(r["flagged"] and r["label"] == "scam" for r in rows)
    fp = sum(r["flagged"] and r["label"] == "benign" for r in rows)
    fn = sum(not r["flagged"] and r["label"] == "scam" for r in rows)
    tn = sum(not r["flagged"] and r["label"] == "benign" for r in rows)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3),
            "accuracy": round((tp + tn) / len(rows), 3), "rows": rows}


def original_engine():
    """The keyword engine as it was at the baseline tag (None if git or the tag is unavailable)."""
    try:
        src = subprocess.run(["git", "show", "baseline-2026-11-26:voiceshield/server/scam.py"], cwd=ROOT,
                             capture_output=True, text=True, encoding="utf-8", check=True).stdout
    except Exception:
        return None
    mod = types.ModuleType("scam_baseline")
    exec(compile(src, "scam_baseline", "exec"), mod.__dict__)
    return mod


def evaluate() -> dict:
    data = json.loads(DATA.read_text(encoding="utf-8"))
    calls = data["calls"]
    out = {"about": data["about"], "n_scam": sum(c["label"] == "scam" for c in calls),
           "n_benign": sum(c["label"] == "benign" for c in calls), "flag_at": FLAG_AT, "engines": {}}
    base = original_engine()
    if base is not None:
        out["engines"]["original keywords"] = run(calls, base.CallScam)
    out["engines"]["keywords + Hinglish"] = run(calls, lambda: scam.CallScam(context=False))
    out["engines"]["keywords + Hinglish + context"] = run(calls, lambda: scam.CallScam(context=True))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    res = evaluate()
    if a.json:
        print(json.dumps(res, ensure_ascii=False, indent=1))
        sys.exit(0)
    print(f"{res['n_scam']} scam and {res['n_benign']} benign calls; flagged = scam score >= {FLAG_AT}\n")
    print(f"{'engine':34s} {'TP':>3} {'FP':>3} {'FN':>3} {'TN':>3} {'prec':>6} {'recall':>6} {'F1':>6}")
    for name, r in res["engines"].items():
        print(f"{name:34s} {r['tp']:3d} {r['fp']:3d} {r['fn']:3d} {r['tn']:3d} {r['precision']:6.2f} {r['recall']:6.2f} {r['f1']:6.2f}")
    print("\nerrors of the context engine:")
    for r in res["engines"]["keywords + Hinglish + context"]["rows"]:
        if r["flagged"] != (r["label"] == "scam"):
            print(f"  {r['id']} {r['label']:6s} score {r['score']:.2f}  {r['kind']}")
