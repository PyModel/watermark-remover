# Ported from wiltodelta/remove-ai-watermarks src/remove_ai_watermarks/_internal/sm2.py, Apache-2.0; modified: flattened into this repo's stdlib script package (no package-internal imports)
"""SM3 hashing and SM2 signature verification (GB/T 32905, GB/T 32918).

Verification only: no key generation, signing, or secret material, so none of the
constant-time concerns of a signing implementation apply. The TC260 security
guide (TC260-PG-202511A) names SM3withSM2 for AI-label signatures, and no
dependency of this package provides SM2 (``cryptography`` has SM3 but not the
curve). Correctness is pinned by the standard's SM3 vectors and OpenSSL's SM2
verify vectors in ``tests/test_sm2.py``.
"""

from __future__ import annotations

import struct

Point = tuple[int, int]

# The SM2 recommended 256-bit curve (GB/T 32918.5).
P = 0xFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000FFFFFFFFFFFFFFFF
A = 0xFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000FFFFFFFFFFFFFFFC
B = 0x28E9FA9E9D9F5E344D5A9E4BCF6509A7F39789F515AB8F92DDBCBD414D940E93
N = 0xFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFF7203DF6B21C6052B53BBF40939D54123
G: Point = (
    0x32C4AE2C1F1981195F9904466A39C9948FE30BBFF2660BE1715A4589334C74C7,
    0xBC3736A2F4F6779C59BDCEE36B692153D0A9877CC62A474002DF32E52139F0A0,
)

# GB/T 32918.2's default signer identity, used when a signature names none.
DEFAULT_USER_ID = b"1234567812345678"

_SM3_IV = (
    0x7380166F,
    0x4914B2B9,
    0x172442D7,
    0xDA8A0600,
    0xA96F30BC,
    0x163138AA,
    0xE38DEE4D,
    0xB0FB0E4E,
)
_MASK = 0xFFFFFFFF


def _rotl(x: int, n: int) -> int:
    n %= 32
    return ((x << n) | (x >> (32 - n))) & _MASK


def _p0(x: int) -> int:
    return x ^ _rotl(x, 9) ^ _rotl(x, 17)


def _p1(x: int) -> int:
    return x ^ _rotl(x, 15) ^ _rotl(x, 23)


def sm3(data: bytes) -> bytes:
    """SM3 digest of ``data`` (32 bytes)."""
    bit_length = len(data) * 8
    padded = data + b"\x80" + b"\x00" * ((55 - len(data)) % 64) + bit_length.to_bytes(8, "big")
    v = list(_SM3_IV)
    for offset in range(0, len(padded), 64):
        w = list(struct.unpack(">16I", padded[offset : offset + 64]))
        for j in range(16, 68):
            w.append(
                _p1(w[j - 16] ^ w[j - 9] ^ _rotl(w[j - 3], 15)) ^ _rotl(w[j - 13], 7) ^ w[j - 6]
            )
        w1 = [w[j] ^ w[j + 4] for j in range(64)]
        a, b, c, d, e, f, g, h = v
        for j in range(64):
            t = 0x79CC4519 if j < 16 else 0x7A879D8A
            ss1 = _rotl((_rotl(a, 12) + e + _rotl(t, j)) & _MASK, 7)
            ss2 = ss1 ^ _rotl(a, 12)
            if j < 16:
                ff, gg = a ^ b ^ c, e ^ f ^ g
            else:
                ff, gg = (a & b) | (a & c) | (b & c), (e & f) | (~e & g)
            tt1 = (ff + d + ss2 + w1[j]) & _MASK
            tt2 = (gg + h + ss1 + w[j]) & _MASK
            a, b, c, d, e, f, g, h = tt1, a, _rotl(b, 9), c, _p0(tt2), e, _rotl(f, 19), g
        v = [x ^ y for x, y in zip(v, (a, b, c, d, e, f, g, h), strict=True)]
    return struct.pack(">8I", *v)


def _add(p: Point | None, q: Point | None) -> Point | None:
    if p is None:
        return q
    if q is None:
        return p
    if p[0] == q[0] and (p[1] + q[1]) % P == 0:
        return None
    numerator, denominator = (
        (3 * p[0] * p[0] + A, 2 * p[1]) if p == q else (q[1] - p[1], q[0] - p[0])
    )
    slope = numerator * pow(denominator, -1, P) % P
    x = (slope * slope - p[0] - q[0]) % P
    return x, (slope * (p[0] - x) - p[1]) % P


def _multiply(k: int, point: Point) -> Point | None:
    result: Point | None = None
    addend: Point | None = point
    while k:
        if k & 1:
            result = _add(result, addend)
        addend = _add(addend, addend)
        k >>= 1
    return result


def on_curve(point: Point) -> bool:
    x, y = point
    return 0 <= x < P and 0 <= y < P and (y * y - (x * x * x + A * x + B)) % P == 0


def lift_x(x: int, *, odd: bool) -> Point | None:
    """The curve point with this x coordinate and the requested y parity, if any."""
    if not 0 <= x < P:
        return None
    y_squared = (x * x * x + A * x + B) % P
    y = pow(y_squared, (P + 1) // 4, P)  # P % 4 == 3
    if y * y % P != y_squared:
        return None
    return (x, y) if (y & 1) == odd else (x, P - y)


def signer_digest(public_key: Point, message: bytes, user_id: bytes = DEFAULT_USER_ID) -> bytes:
    """``e = SM3(Z_A || M)``, the value an SM3withSM2 signature signs."""
    identity_bits = (len(user_id) * 8).to_bytes(2, "big")
    fields = (A, B, G[0], G[1], public_key[0], public_key[1])
    z = sm3(identity_bits + user_id + b"".join(value.to_bytes(32, "big") for value in fields))
    return sm3(z + message)


def verify_digest(public_key: Point, digest: bytes, r: int, s: int) -> bool:
    """Verify an SM2 signature ``(r, s)`` over an already computed digest ``e``."""
    if not (1 <= r < N and 1 <= s < N) or not on_curve(public_key):
        return False
    t = (r + s) % N
    if t == 0:
        return False
    point = _add(_multiply(s, G), _multiply(t, public_key))
    return point is not None and (int.from_bytes(digest, "big") + point[0]) % N == r


def verify(
    public_key: Point, message: bytes, signature: bytes, user_id: bytes = DEFAULT_USER_ID
) -> bool:
    """Verify a raw 64-byte ``r || s`` SM3withSM2 signature over ``message``.

    Signatures of any other length (DER blobs, truncated or padded buffers)
    verify False: a malformed signature is a stable rejection, never an error.
    """
    if len(signature) != 64:
        return False
    r, s = int.from_bytes(signature[:32], "big"), int.from_bytes(signature[32:], "big")
    return verify_digest(public_key, signer_digest(public_key, message, user_id), r, s)
