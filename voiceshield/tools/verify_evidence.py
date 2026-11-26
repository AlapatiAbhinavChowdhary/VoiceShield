"""Check a VoiceShield evidence file offline: digest and Ed25519 signature, no server needed.

    python tools/verify_evidence.py voiceshield-session-52-evidence.json [--public-key HEX]

--public-key: the bank's copy of the server key (GET /api/evidence/public-key). Without it the signature is
checked against the key embedded in the file, which proves integrity but not WHO signed it.
Exit code 0 = authentic or intact-unsigned, 1 = tampered or invalid.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from server.evidence import check  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--public-key", help="trusted server public key (hex)")
    a = ap.parse_args()
    r = check(json.loads(Path(a.file).read_text(encoding="utf-8")), a.public_key)
    print(f"{r['verdict'].upper()}: {r['detail']}")
    for k in ("call_id", "digest_ok", "signature", "key_id", "signed_by_this_server"):
        if k in r:
            print(f"  {k}: {r[k]}")
    sys.exit(0 if r["verdict"] in ("authentic", "intact-unsigned") else 1)
