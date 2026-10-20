"""SM3 hashing and SM2 (verify-only) — upstream standard vectors, adapted.

Vectors from GB/T 32905-2016 (SM3) and OpenSSL's evppkey_sm2.txt (SM2 verify),
via the upstream test suite. Adapted to this repo's flat stdlib script package.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import sm2

# OpenSSL test/recipes/30-test_evp_data/evppkey_sm2.txt, key SM2_key1: the public
# point of its PKCS#8 key and two "Verify" rows (digest e, signature r, s).
OPENSSL_KEY = (
    0x393A2AD9E27E43ACAAAB85A19D3B85591E14406157175ADDD37BF77FF0CAF9ED,
    0x37FCF3A8B4F55C038523261FCA6636AFE6A7AFA2A1482A606F5CB3A18B2E2A1E,
)
OPENSSL_VERIFY = (
    (
        "D7AD397F6FFA5D4F7F11E7217F241607DC30618C236D2C09C1B9EA8FDADEE2E8",
        0xAB1DB64DE7C40EDBDE6651C9B8EBDB804673DB836E5D5C7FE15DCF9ED2725037,
        0xEBA714451FF69B0BB930B379E192E7CD5FA6E3C41C7FBD8303B799AB54A54621,
    ),
    (
        "B1139602C6ECC9E15E2F3F9C635A1AFE737058BC15387479C1EA0D0B3D90E9E5",
        0xE6E0414EBD3A656C35602AF14AB20287DBF30D57AF75C49A188ED4B42391F224,
        0x2F54F277C606F4605E1CE9514947FFDDF94C67A539804A4ED17F852288BDBE2E,
    ),
)


class TestSm3:
    def test_standard_vectors(self) -> None:
        # GB/T 32905-2016 appendix A examples 1 and 2.
        assert (
            sm2.sm3(b"abc").hex()
            == "66c7f0f462eeedd9d1f2d46bdc10e4e24167c4875cf2f7a2297da02b8f4ba8e0"
        )
        assert (
            sm2.sm3(b"abcd" * 16).hex()
            == "debe9ff92275b8a138604889c18e5a4d6fdb70e5387e5765293dcba39c0c5732"
        )

    def test_empty_input_is_stable(self) -> None:
        # Independent check: a second call must agree, and length is 32 bytes.
        first = sm2.sm3(b"")
        assert len(first) == 32
        assert first == sm2.sm3(b"")


class TestSm2Verify:
    def test_openssl_vectors_verify(self) -> None:
        for digest, r, s in OPENSSL_VERIFY:
            assert sm2.verify_digest(OPENSSL_KEY, bytes.fromhex(digest), r, s)

    def test_a_changed_digest_or_signature_fails(self) -> None:
        digest, r, s = OPENSSL_VERIFY[0]
        changed = bytes.fromhex(digest)[:-1] + b"\x00"
        assert not sm2.verify_digest(OPENSSL_KEY, changed, r, s)
        assert not sm2.verify_digest(OPENSSL_KEY, bytes.fromhex(digest), r, s ^ 1)

    def test_lift_x_returns_the_requested_parity(self) -> None:
        for odd in (False, True):
            point = sm2.lift_x(OPENSSL_KEY[0], odd=odd)
            assert point is not None
            assert sm2.on_curve(point)
            assert (point[1] & 1) == odd

    def test_an_off_curve_or_out_of_range_key_is_rejected(self) -> None:
        digest, r, s = OPENSSL_VERIFY[0]
        assert not sm2.verify_digest(
            (OPENSSL_KEY[0], OPENSSL_KEY[1] + 1), bytes.fromhex(digest), r, s
        )
        # Congruent to the real key mod P, so the point arithmetic alone would accept it.
        assert not sm2.verify_digest(
            (OPENSSL_KEY[0] + sm2.P, OPENSSL_KEY[1]), bytes.fromhex(digest), r, s
        )

    def test_out_of_range_r_and_s_are_stable_rejections(self) -> None:
        # Malformed signature integers (a truncated/DER-garbled r||s in int form)
        # must verify False, never raise.
        digest = bytes.fromhex(OPENSSL_VERIFY[0][0])
        assert not sm2.verify_digest(OPENSSL_KEY, digest, 0, 1)
        assert not sm2.verify_digest(OPENSSL_KEY, digest, 1, 0)
        assert not sm2.verify_digest(OPENSSL_KEY, digest, sm2.N, 1)
        assert not sm2.verify_digest(OPENSSL_KEY, digest, 1, sm2.N)
        assert not sm2.verify_digest(OPENSSL_KEY, digest, -1, 1)
        assert not sm2.verify_digest(OPENSSL_KEY, digest, 1, -1)

    def test_verify_rejects_malformed_signature_blobs(self) -> None:
        # Raw r||s (64 bytes) is the ported form; DER blobs and truncated or
        # padded buffers are stable rejections, not crashes.
        message = b"payload"
        assert not sm2.verify(OPENSSL_KEY, message, b"")
        assert not sm2.verify(OPENSSL_KEY, message, b"\x30\x44")  # DER prefix sniff
        assert not sm2.verify(OPENSSL_KEY, message, b"\x00" * 63)
        assert not sm2.verify(OPENSSL_KEY, message, b"\x00" * 65)

    def test_verify_only_surface(self) -> None:
        # Verification-only crypto: no key generation or signing capability.
        assert not hasattr(sm2, "sign")
        assert not hasattr(sm2, "sign_digest")
        assert not hasattr(sm2, "generate_key")
