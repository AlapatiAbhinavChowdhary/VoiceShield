"""Tamper-evident, signed incident evidence.

An evidence file = the full audit record of one call + the SHA-256 of its canonical JSON (since 26 Nov 2026) +
an Ed25519 signature by this server over (format, call id, time, digest). Anyone with the server's public key
(GET /api/evidence/public-key) can check offline that the file came from this server and was not edited
(tools/verify_evidence.py). Files made before signing existed still verify by digest; they report "unsigned".

The signing key is generated on first use and kept in data/evidence_key.json (git-ignored, never sent out).
"""
import hashlib
import json
import secrets
import time
from pathlib import Path

from . import ed25519
from .config import write_text_atomic

FORMAT = "voiceshield-evidence/1"


def canonical(call: dict) -> str:
    return json.dumps(call, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(call: dict) -> str:
    return hashlib.sha256(canonical(call).encode("utf-8")).hexdigest()


def message(call_id, generated_at: str, sha256: str) -> bytes:
    return f"{FORMAT}|{call_id}|{generated_at}|{sha256}".encode("utf-8")


class EvidenceSigner:
    def __init__(self, path: Path):
        self.path = Path(path)
        if self.path.exists():
            k = json.loads(self.path.read_text(encoding="utf-8"))
            self._seed = bytes.fromhex(k["seed_hex"])
            self.created = k.get("created", "")
        else:
            self._seed = secrets.token_bytes(32)
            self.created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.public = ed25519.public_key(self._seed)
        self.key_id = hashlib.sha256(self.public).hexdigest()[:16]
        if not self.path.exists():
            write_text_atomic(self.path, json.dumps({"alg": "Ed25519", "seed_hex": self._seed.hex(),
                                                     "public_key": self.public.hex(), "key_id": self.key_id,
                                                     "created": self.created}, indent=1))

    def info(self) -> dict:
        return {"alg": "Ed25519", "public_key": self.public.hex(), "key_id": self.key_id, "created": self.created}

    def sign(self, call_id, generated_at: str, sha256: str) -> dict:
        sig = ed25519.sign(self._seed, message(call_id, generated_at, sha256))
        return {"alg": "Ed25519", "key_id": self.key_id, "public_key": self.public.hex(), "value": sig.hex(),
                "signed": f"{FORMAT}|<call id>|<generated_at>|<sha256>"}


def check(ev: dict, trusted_public_hex: str | None = None) -> dict:
    """Verify an evidence file. trusted_public_hex: this server's key, to say whether it signed the file."""
    call = ev.get("call")
    if not isinstance(call, dict) or not isinstance(ev.get("sha256"), str):
        return {"verdict": "invalid", "detail": "not a VoiceShield evidence file (no call record or digest)"}
    digest_ok = digest(call) == ev["sha256"]
    sig = ev.get("signature")
    out = {"call_id": call.get("id"), "digest_ok": digest_ok, "signature": "missing", "signed_by_this_server": None}
    if isinstance(sig, dict) and sig.get("alg") == "Ed25519":
        try:
            pub, val = bytes.fromhex(sig.get("public_key", "")), bytes.fromhex(sig.get("value", ""))
            ok = ed25519.verify(pub, message(call.get("id"), ev.get("generated_at", ""), ev["sha256"]), val)
        except ValueError:
            ok = False
        out["signature"] = "valid" if ok else "invalid"
        out["key_id"] = sig.get("key_id")
        if trusted_public_hex is not None:
            out["signed_by_this_server"] = ok and sig.get("public_key") == trusted_public_hex
    if not digest_ok or out["signature"] == "invalid":
        out["verdict"] = "tampered"
        out["detail"] = ("the signature does not match" if digest_ok else
                         "the record no longer matches its SHA-256 digest" +
                         (": it was edited after the server signed it" if out["signature"] == "valid" else ""))
    elif out["signature"] == "missing":
        out["verdict"] = "intact-unsigned"
        out["detail"] = "the record matches its digest; the file has no signature (made before signing was added)"
    else:
        out["verdict"] = "authentic"
        out["detail"] = "the record matches its digest and the Ed25519 signature is valid" + \
            ("" if trusted_public_hex is None else
             " (signed by this server)" if out["signed_by_this_server"] else " (signed by ANOTHER key, not this server)")
    return out
