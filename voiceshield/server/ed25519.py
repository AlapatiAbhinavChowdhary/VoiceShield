"""Ed25519 signatures (RFC 8032) for incident evidence.

Uses the audited `cryptography` library (OpenSSL) when it is installed (requirements.txt), and otherwise the
standard-library reference implementation below, after the RFC's Python code. The reference code is not
constant-time: acceptable as a fallback for a local demo server, not for keys exposed to timing attacks.
"""
import hashlib

p = 2 ** 255 - 19
q = 2 ** 252 + 27742317777372353535851937790883648493


def _inv(x: int) -> int:
    return pow(x, p - 2, p)


d = -121665 * _inv(121666) % p
_SQRT_M1 = pow(2, (p - 1) // 4, p)


def _sha512_modq(b: bytes) -> int:
    return int.from_bytes(hashlib.sha512(b).digest(), "little") % q


def _add(P, Q):
    A, B = (P[1] - P[0]) * (Q[1] - Q[0]) % p, (P[1] + P[0]) * (Q[1] + Q[0]) % p
    C, D = 2 * P[3] * Q[3] * d % p, 2 * P[2] * Q[2] % p
    E, F, G, H = B - A, D - C, D + C, B + A
    return (E * F, G * H, F * G, E * H)


def _mul(s: int, P):
    Q = (0, 1, 1, 0)
    while s > 0:
        if s & 1:
            Q = _add(Q, P)
        P = _add(P, P)
        s >>= 1
    return Q


def _equal(P, Q) -> bool:
    return (P[0] * Q[2] - Q[0] * P[2]) % p == 0 and (P[1] * Q[2] - Q[1] * P[2]) % p == 0


def _recover_x(y: int, sign: int):
    if y >= p:
        return None
    x2 = (y * y - 1) * _inv(d * y * y + 1) % p
    if x2 == 0:
        return None if sign else 0
    x = pow(x2, (p + 3) // 8, p)
    if (x * x - x2) % p != 0:
        x = x * _SQRT_M1 % p
    if (x * x - x2) % p != 0:
        return None
    if (x & 1) != sign:
        x = p - x
    return x


_gy = 4 * _inv(5) % p
_gx = _recover_x(_gy, 0)
_G = (_gx, _gy, 1, _gx * _gy % p)


def _compress(P) -> bytes:
    zi = _inv(P[2])
    x, y = P[0] * zi % p, P[1] * zi % p
    return int.to_bytes(y | ((x & 1) << 255), 32, "little")


def _decompress(s: bytes):
    if len(s) != 32:
        return None
    y = int.from_bytes(s, "little")
    sign = y >> 255
    y &= (1 << 255) - 1
    x = _recover_x(y, sign)
    return None if x is None else (x, y, 1, x * y % p)


def _expand(secret: bytes):
    if len(secret) != 32:
        raise ValueError("Ed25519 secret key must be 32 bytes")
    h = hashlib.sha512(secret).digest()
    a = int.from_bytes(h[:32], "little")
    a &= (1 << 254) - 8
    a |= 1 << 254
    return a, h[32:]


def _public_key_ref(secret: bytes) -> bytes:
    a, _ = _expand(secret)
    return _compress(_mul(a, _G))


def _sign_ref(secret: bytes, msg: bytes) -> bytes:
    a, prefix = _expand(secret)
    A = _compress(_mul(a, _G))
    r = _sha512_modq(prefix + msg)
    Rs = _compress(_mul(r, _G))
    s = (r + _sha512_modq(Rs + A + msg) * a) % q
    return Rs + int.to_bytes(s, 32, "little")


# ---------------------------------------------------------------- backend: the audited library when installed
# Ed25519 is deterministic, so the library and the reference code produce byte-identical keys and signatures:
# evidence signed by either verifies with either (checked in the tests whenever the library is present).
try:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
    BACKEND = "cryptography (OpenSSL)"

    def public_key(secret: bytes) -> bytes:
        return Ed25519PrivateKey.from_private_bytes(secret).public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)

    def sign(secret: bytes, msg: bytes) -> bytes:
        return Ed25519PrivateKey.from_private_bytes(secret).sign(msg)

    def verify(public: bytes, msg: bytes, signature: bytes) -> bool:
        if len(public) != 32 or len(signature) != 64:
            return False
        try:
            Ed25519PublicKey.from_public_bytes(public).verify(signature, msg)
            return True
        except (InvalidSignature, ValueError):
            return False
except ImportError:                                        # stdlib fallback: same results, not constant-time
    BACKEND = "built-in RFC 8032 reference (stdlib)"
    public_key, sign = _public_key_ref, _sign_ref

    def verify(public: bytes, msg: bytes, signature: bytes) -> bool:
        return verify_ref(public, msg, signature)


def verify_ref(public: bytes, msg: bytes, signature: bytes) -> bool:
    if len(public) != 32 or len(signature) != 64:
        return False
    A, R = _decompress(public), _decompress(signature[:32])
    if A is None or R is None:
        return False
    s = int.from_bytes(signature[32:], "little")
    if s >= q:
        return False
    h = _sha512_modq(signature[:32] + public + msg)
    return _equal(_mul(s, _G), _add(R, _mul(h, A)))
