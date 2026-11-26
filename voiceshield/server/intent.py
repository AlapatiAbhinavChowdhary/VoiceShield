"""Scam-intent classifier (shadow mode): hashed word + character n-grams and an L2 logistic regression.

Why: the keyword rules only fire on known phrases. Indirect scams ("read the code you will receive", "scan this QR
code to receive the money") use none of them. A small n-gram model generalises from phrasing, works for romanised
Hinglish and Devanagari / Telugu script, needs no new dependency (NumPy + SciPy) and trains in well under a second
from server/intent_train.json at start-up.

Status: SHADOW. Its probability is reported next to the rules but never changes a decision (config scam_semantic:
"shadow" or "off"). See tools/eval_scam.py for how it compares with the rules on a held-out set.
"""
import json
import re
import unicodedata
import zlib
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix

DIM = 2 ** 18
TRAIN = Path(__file__).with_name("intent_train.json")


def _norm(t: str) -> str:
    t = unicodedata.normalize("NFC", t).lower()
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def features(text: str) -> dict[int, float]:
    words = _norm(text).split()
    grams = ["w:" + w for w in words] + ["b:" + a + "_" + b for a, b in zip(words, words[1:])]
    for w in words:
        ww = f" {w} "
        grams += ["c:" + ww[i:i + n] for n in (3, 4, 5) for i in range(len(ww) - n + 1)]
    f: dict[int, float] = {}
    for g in grams:
        h = zlib.crc32(g.encode("utf-8")) % DIM
        f[h] = f.get(h, 0.0) + 1.0
    if not f:
        return f
    v = {k: 1.0 + np.log(c) for k, c in f.items()}             # sublinear term frequency
    norm = np.sqrt(sum(x * x for x in v.values()))
    return {k: x / norm for k, x in v.items()}


def _matrix(texts) -> csr_matrix:
    rows, cols, vals = [], [], []
    for i, t in enumerate(texts):
        for k, x in features(t).items():
            rows.append(i); cols.append(k); vals.append(x)
    return csr_matrix((vals, (rows, cols)), shape=(len(texts), DIM))


class IntentModel:
    def __init__(self, data=None, l2: float = 0.3):
        data = data or json.loads(TRAIN.read_text(encoding="utf-8"))["data"]
        y = np.array([d[0] for d in data], float)
        X = _matrix([d[1] for d in data])
        used = np.unique(X.indices)                              # optimise only the features that occur
        Xu = X[:, used]

        def loss(wb):
            w, b = wb[:-1], wb[-1]
            z = Xu @ w + b
            p = 1 / (1 + np.exp(-z))
            ll = -np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)) + l2 * np.dot(w, w) / 2
            g = Xu.T @ (p - y) / len(y)
            return ll, np.concatenate([g + l2 * w, [np.mean(p - y)]])

        res = minimize(loss, np.zeros(len(used) + 1), jac=True, method="L-BFGS-B", options={"maxiter": 500})
        self.w = np.zeros(DIM)
        self.w[used] = res.x[:-1]
        self.b = float(res.x[-1])
        self.n_train = len(data)

    def proba(self, text: str) -> float:
        f = features(text)
        z = sum(self.w[k] * x for k, x in f.items()) + self.b
        return float(1 / (1 + np.exp(-z)))


class CallIntent:
    """Conversation level: the highest intent seen in one utterance or in two consecutive utterances together."""

    def __init__(self, model: IntentModel):
        self.model, self.prev, self.call_p = model, "", 0.0

    def add(self, text: str) -> dict:
        p = self.model.proba(text)
        pair = self.model.proba(self.prev + " " + text) if self.prev else p
        self.prev = text
        self.call_p = max(self.call_p, p, pair)
        return {"p": round(p, 3), "call_p": round(self.call_p, 3), "mode": "shadow"}
