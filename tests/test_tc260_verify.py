"""TC260 producer registry and SecurityData signature verification.

Registry vectors adapted from the upstream tc260_producers tests; signature
outcomes are exercised through crafted labels whose signatures are built with
a chosen-public-key construction (see `_craft_signature`) — no signing code
exists anywhere in the shipped modules.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import sm2
import tc260_verify
from tc260_verify import (
    TC260_PRODUCERS,
    TC260_SIGNING_KEYS,
    check_tc260_signature,
    label_message,
    producer_codes_for_mark,
    producer_for_code,
    producer_for_signer,
    uscc_of,
)

#: A scalar whose product with G becomes the label's embedded public key. The
#: test chooses the key, so a signature for it is solvable without any signing
#: implementation: for pub = k*G, s = (u - k*r) * (1+k)^-1 mod n satisfies
#: verify (s*G + (r+s)*pub == u*G). This is test-fixture math only.
CHOSEN_K = 0x2B42D00
EMBEDDED_KEY = sm2._multiply(CHOSEN_K, sm2.G)
assert EMBEDDED_KEY is not None

DOUBAO_USCC = "91110102MACQD9K640"
SM3_WITH_SM2 = "1.2.156.10197.1.501"


def _craft_signature(pub: sm2.Point, message: bytes, u: int = 0x51ED4B22) -> bytes:
    k = CHOSEN_K
    e = int.from_bytes(sm2.signer_digest(pub, message), "big")
    n = sm2.N
    u %= n
    # R = u*G gives the point verify will recompute: s*G + (r+s)*pub == u*G.
    point = sm2._multiply(u, sm2.G)
    assert point is not None
    r = (e + point[0]) % n
    s = (u - k * r) * pow(1 + k, -1, n) % n
    if not (1 <= r < n and 1 <= s < n) or (r + s) % n == 0:
        raise AssertionError("chosen-key construction produced a degenerate signature")
    return r.to_bytes(32, "big") + s.to_bytes(32, "big")


def _security_data(pub: sm2.Point, signature: bytes, *, key_id: int = 1) -> dict:
    return {
        "Type": "TC260PG",
        "PubSD": [
            {
                "Type": "PubKey",
                "AlgID": SM3_WITH_SM2,
                "KeyID": key_id,
                "KeyValue": ("04" + format(pub[0], "064x") + format(pub[1], "064x")),
            },
            {
                "Type": "DS",
                "AlgID": SM3_WITH_SM2,
                "KeyID": key_id,
                "TBSData": {"Type": "Md"},
                "Signature": signature.hex(),
            },
        ],
    }


def _label(**overrides: object) -> dict[str, str]:
    label = {
        "Label": "1",
        "ContentProducer": DOUBAO_USCC,
        "ProduceID": "c6a75f11-3f4e-4b0a-9d21-77c0aa12e55f",
    }
    label.update(overrides)  # type: ignore[arg-type]
    return label


def _security_json(security: dict) -> str:
    """ReservedCode1 wraps the SecurityData object, per TC260-PG-202511A."""
    return json.dumps({"SecurityData": security}, separators=(",", ":"))


def _signed_label(pub: sm2.Point, k: int, label: dict[str, str]) -> dict[str, str]:
    signature = _craft_signature(pub, label_message(label))
    return {**label, "ReservedCode1": _security_json(_security_data(pub, signature))}


class TestProducerRegistry:
    def test_every_identity_resolves_across_spellings(self) -> None:
        for row in TC260_PRODUCERS:
            for code in row.codes:
                for spelling in (code, code.swapcase(), f" {code} "):
                    assert producer_for_code(spelling) is row
                if len(code) == 18:
                    # "001" + type digit + USCC + product suffix normalizes to the USCC.
                    assert producer_for_code(f"0011{code}00001") is row
                assert producer_for_code(f"prefix-{code}-suffix") is None

    def test_registries_are_unambiguous(self) -> None:
        codes = [code.casefold() for row in TC260_PRODUCERS for code in row.codes]
        assert len(codes) == len(set(codes))
        assert len({row.key for row in TC260_PRODUCERS}) == len(TC260_PRODUCERS)
        assert all((row.signer is None) == (row.signing_key_x is None) for row in TC260_PRODUCERS)

    def test_mark_routing_resolves_registered_marks(self) -> None:
        doubao = producer_for_code(DOUBAO_USCC)
        assert doubao is not None and doubao.image_mark == "doubao"
        assert producer_codes_for_mark("doubao") == doubao.codes
        assert producer_codes_for_mark("not-a-mark") == ()

    def test_uscc_of_extracts_the_identity_segment(self) -> None:
        assert uscc_of(f"0011{DOUBAO_USCC}00001") == DOUBAO_USCC
        assert uscc_of(DOUBAO_USCC) == DOUBAO_USCC
        assert uscc_of("doubao") == "doubao"

    def test_pinned_signing_keys_derive_from_the_registry(self) -> None:
        minimax = producer_for_signer("MiniMax")
        assert minimax is not None and minimax.signing_key_x is not None
        assert TC260_SIGNING_KEYS["MiniMax"][0] == minimax.signing_key_x


class TestCheckTc260Signature:
    def test_a_valid_label_signature_verifies_with_its_own_key(self) -> None:
        label = _signed_label(EMBEDDED_KEY, CHOSEN_K, _label())
        result = check_tc260_signature(label)
        assert result is not None
        assert (result.label, result.signer) == ("verified", None)
        assert result.describe() == (
            "TC260 label signature verified with its own embedded key (no pinned signer)"
        )

    def test_a_changed_produce_id_fails_the_label_signature(self) -> None:
        label = _signed_label(EMBEDDED_KEY, CHOSEN_K, _label())
        produce_id = label["ProduceID"]
        altered = {**label, "ProduceID": produce_id[:-1] + ("0" if produce_id[-1] != "0" else "1")}
        result = check_tc260_signature(altered)
        assert result is not None
        assert (result.label, result.signer) == ("failed", None)
        assert result.describe() == "TC260 label signature does not verify"

    def test_a_repeated_check_verifies_once(self) -> None:
        label = _signed_label(EMBEDDED_KEY, CHOSEN_K, _label())
        first = check_tc260_signature(label)
        second = check_tc260_signature(dict(label))
        assert first == second

    def test_no_security_data_returns_none(self) -> None:
        assert check_tc260_signature(_label()) is None
        assert check_tc260_signature({_label()["Label"]: "1"}) is None

    def test_non_tc260_security_data_returns_none(self) -> None:
        label = _label(ReservedCode1=_security_json({"Type": "Other"}))
        assert check_tc260_signature(label) is None

    def test_security_data_without_signature_entries_is_absent(self) -> None:
        security = {
            "Type": "TC260PG",
            "PubSD": [_security_data(EMBEDDED_KEY, b"\x00" * 64)["PubSD"][0]],
        }
        label = _label(ReservedCode1=_security_json(security))
        result = check_tc260_signature(label)
        assert result is not None
        assert result.label == "absent"
        assert result.describe() == "TC260 SecurityData without a label signature"

    def test_unsupported_algorithm_is_reported_not_crashed(self) -> None:
        security = _security_data(EMBEDDED_KEY, b"\x00" * 64)
        security["PubSD"][1]["AlgID"] = "1.2.840.113549.1.1.1"  # RSA, per Honor YOYO labels
        label = _label(ReservedCode1=_security_json(security))
        result = check_tc260_signature(label)
        assert result is not None
        assert result.label == "unsupported"
        assert "not supported" in result.describe()

    def test_malformed_signature_hex_is_a_stable_rejection(self) -> None:
        security = _security_data(EMBEDDED_KEY, b"\x00" * 64)
        security["PubSD"][1]["Signature"] = "zz-not-hex"
        label = _label(ReservedCode1=_security_json(security))
        result = check_tc260_signature(label)
        assert result is not None
        assert result.label == "failed"

    def test_malformed_key_hex_is_a_stable_rejection(self) -> None:
        security = _security_data(EMBEDDED_KEY, b"\x00" * 64)
        security["PubSD"][0]["KeyValue"] = "not-hex-at-all"
        label = _label(ReservedCode1=_security_json(security))
        result = check_tc260_signature(label)
        assert result is not None
        assert result.label == "failed"

    def test_non_string_reserved_code_returns_none(self) -> None:
        label = _label(ReservedCode1="{}")  # no SecurityData key
        assert check_tc260_signature(label) is None

    def test_minimax_spelling_is_accepted_for_signed_content(self) -> None:
        pub = EMBEDDED_KEY
        signature = _craft_signature(pub, label_message(_label()))
        security = _security_data(pub, signature)
        security["PubSD"][1]["TBSData"] = {"Type": "LabelMataData"}
        label = _label(ReservedCode1=_security_json(security))
        result = check_tc260_signature(label)
        assert result is not None
        assert result.label == "verified"

    def test_huge_verify_inputs_bypass_the_signature_cache(self) -> None:
        """Oversized (message, signature) pairs skip the bounded lru_cache.

        The cache admitted every (key, message, signature) triple up to its
        256-entry count with no per-entry size bound, so one multi-megabyte
        label payload pinned megabytes of curve-math inputs in memory.
        Entries past a small threshold verify directly; the cache must not
        grow, and the verdict must not change.
        """
        tc260_verify._verified.cache_clear()
        pub = EMBEDDED_KEY
        huge_pad = "x" * (64 * 1024 + 1)
        label = _signed_label(pub, CHOSEN_K, _label(ProduceID=huge_pad))
        assert len(label_message(label)) > 64 * 1024
        before = tc260_verify._verified.cache_info()
        result = check_tc260_signature(label)
        after = tc260_verify._verified.cache_info()
        assert result is not None and result.label == "verified"
        assert (after.hits, after.misses) == (before.hits, before.misses)

    def test_a_huge_bad_signature_still_fails_without_caching(self) -> None:
        tc260_verify._verified.cache_clear()
        huge_pad = "x" * (64 * 1024 + 1)
        security = _security_data(EMBEDDED_KEY, b"\x00" * (64 * 1024 + 1))
        label = _label(ProduceID=huge_pad, ReservedCode1=_security_json(security))
        before = tc260_verify._verified.cache_info()
        result = check_tc260_signature(label)
        after = tc260_verify._verified.cache_info()
        assert result is not None and result.label == "failed"
        assert (after.hits, after.misses) == (before.hits, before.misses)


def test_verify_only_surface() -> None:
    assert not hasattr(tc260_verify, "sign")
    assert not hasattr(tc260_verify, "generate_key")
