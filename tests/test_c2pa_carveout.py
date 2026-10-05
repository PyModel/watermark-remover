"""C2PA carve-out: ported constants registry, bounded CBOR/caBX scanner, fallback wiring.

Carve-out contract (ticket 06):
- The CC BY 4.0 soft-binding algorithm snapshot is NOT ported; soft-binding
  lookups resolve to a labeled-unknown, never a guessed algorithm name.
- Without the c2pa Rust wheel, a bounded stdlib CBOR/caBX scan surfaces
  claim-generator/issuer hints validated against the ported registry.
- With the wheel present it stays the authoritative reader; the stdlib scan
  must not contradict it.
"""

from __future__ import annotations

import json
import struct
import sys
import zlib
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "skills" / "remove-ai-marks" / "scripts")
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "c2pa"

# Distinctive tokens from the upstream CC BY 4.0 soft-binding snapshot
# (c2pa-org/softbinding-algorithm-list). If any of these appear in the tree,
# the carve-out leaked licensed registry data into this repo.
CC_BY_SNAPSHOT_TOKENS = (
    b"com.digimarc",
    b"com.imatag",
    b"com.microsoft.invismark",
    b"com.microsoft.wavmark",
    b"com.verimatrix",
    b"com.nagra.nexguard",
    b"com.aiwatermark",
    b"ai.trufo",
    b"app.overlai",
    b"com.markany",
    b"com.mentaport",
    b"es.lumatrace",
    b"ai.verda.watermark",
    b"ai.contentlens",
    b"io.iscc",
    b"ai.steg",
    b"com.adobe.trustmark",
    b"com.adobe.icn",
    b"softbinding-algorithm-list",
)

_PRUNED_DIRS = {".git", ".venv", "node_modules", "__pycache__", ".pytest_cache", ".ruff_cache"}
_MAX_GREP_FILE_BYTES = 10 * 1024 * 1024


def _tree_files():
    for path in REPO_ROOT.rglob("*"):
        if path.is_dir():
            continue
        if _PRUNED_DIRS.intersection(path.relative_to(REPO_ROOT).parts):
            continue
        yield path


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def build_cabx_png(cabx_payload: bytes, path: Path) -> Path:
    """Write a minimal PNG with one caBX chunk before IDAT."""
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0)
    raw_idat = zlib.compress(b"\x00\x00\x00\x00")
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"caBX", cabx_payload)
        + _png_chunk(b"IDAT", raw_idat)
        + _png_chunk(b"IEND", b"")
    )
    return path


# ---------------------------------------------------------------------------
# Ported constants registry (carve-out)
# ---------------------------------------------------------------------------


def test_constants_registry_carveout_rows():
    import c2pa_constants as cc

    assert cc.C2PA_ISSUERS[b"Adobe"] == "Adobe"
    assert cc.C2PA_ISSUERS[b"OpenAI"] == "OpenAI"
    assert cc.C2PA_ISSUERS["北京火山引擎科技有限公司".encode()] == "ByteDance (Volcano Engine)"
    orgs = {vendor.org for vendor in cc.C2PA_AI_VENDORS}
    assert "Google LLC" in orgs and "fal.ai" in orgs and "Anthropic Claude" in orgs
    assert ("grok imagine", "xAI Grok Imagine") in cc.C2PA_CLAIM_GENERATOR_PLATFORMS
    assert cc.C2PA_ACTIONS[b"c2pa.created"] == "created"
    assert cc.C2PA_AI_TOOLS[b"Firefly"] == "Firefly"
    assert cc.C2PA_CHUNK_TYPE == b"caBX" and cc.PNG_SIGNATURE == b"\x89PNG\r\n\x1a\n"
    assert any(b"c2pa" in sig for sig in cc.C2PA_SIGNATURES)


def test_provenance_header_names_upstream_constants_module():
    header = (REPO_ROOT / "skills/remove-ai-marks/scripts/c2pa_constants.py").read_text(
        encoding="utf-8"
    )[:400]
    assert "Ported from wiltodelta/remove-ai-watermarks" in header
    assert "src/remove_ai_watermarks/_internal/constants.py" in header
    assert "Apache-2.0" in header
    assert "modified:" in header


def test_soft_binding_lookups_are_labeled_unknown():
    import c2pa_constants as cc

    label = cc.soft_binding_label("com.digimarc.validate.1")
    assert label == cc.soft_binding_label(b"com.digimarc.validate.1")
    assert "unknown" in label.lower()
    # Never a guessed algorithm name: the lookup must not echo registry data.
    assert "digimarc" not in label.lower()
    # No soft-binding registry symbols exist anywhere in the carve-out module.
    exported = dir(cc)
    for symbol in (
        "C2PA_SOFT_BINDING_ROWS",
        "C2PA_SOFT_BINDING_REGISTRY",
        "C2PA_SOFT_BINDINGS",
        "C2paSoftBindingAlgorithm",
    ):
        assert symbol not in exported


def test_no_cc_by_snapshot_data_anywhere_in_tree():
    self_path = Path(__file__).resolve()
    for path in _tree_files():
        if path == self_path:
            continue  # this file carries only the guard tokens themselves, not registry rows
        if path.stat().st_size > _MAX_GREP_FILE_BYTES:
            continue
        blob = path.read_bytes()
        for token in CC_BY_SNAPSHOT_TOKENS:
            assert token not in blob, f"CC BY 4.0 snapshot token {token!r} found in {path}"


# ---------------------------------------------------------------------------
# Bounded CBOR mini-parser
# ---------------------------------------------------------------------------


def _cbor_text(value: str) -> bytes:
    raw = value.encode("utf-8")
    if len(raw) < 24:
        return bytes([0x60 + len(raw)]) + raw
    if len(raw) < 256:
        return b"\x78" + bytes([len(raw)]) + raw
    return b"\x79" + len(raw).to_bytes(2, "big") + raw


def test_cbor_texts_roundtrip():
    from c2pa_constants import decode_cbor_texts

    payload = (
        b"\xa2"
        + _cbor_text("claim_generator")
        + _cbor_text("adobe firefly/1.0")
        + _cbor_text("k")
        + b"\x01"
    )
    result = decode_cbor_texts(payload)
    assert result.status == "ok" and result.reason is None
    assert "adobe firefly/1.0" in result.texts and "claim_generator" in result.texts


def test_cbor_nested_containers_decode():
    from c2pa_constants import decode_cbor_texts

    payload = b"\x82" + b"\x81" + _cbor_text("inner") + _cbor_text("outer")
    result = decode_cbor_texts(payload)
    assert result.status == "ok"
    assert list(result.texts) == ["inner", "outer"]


def test_cbor_tag_wraps_exactly_one_item():
    from c2pa_constants import decode_cbor_texts

    payload = b"\xc6" + _cbor_text("abc")  # tag(6) wrapping a single text string
    result = decode_cbor_texts(payload)
    assert result.status == "ok" and result.reason is None
    assert list(result.texts) == ["abc"]


def test_cbor_malformed_tag_payload_is_stable_rejection():
    from c2pa_constants import decode_cbor_texts

    result = decode_cbor_texts(b"\xc6\x63ab")  # tag wrapping a truncated text string
    assert result.status == "rejected"
    assert result.reason is not None and "truncat" in result.reason.lower()
    assert result.texts == ()


def test_cbor_truncated_payload_is_stable_rejection():
    from c2pa_constants import decode_cbor_texts

    result = decode_cbor_texts(b"\x63ab")  # text string claiming 3 bytes, carries 2
    assert result.status == "rejected"
    assert result.reason is not None and "truncat" in result.reason.lower()
    assert result.texts == ()


def test_cbor_deeply_nested_payload_is_stable_rejection():
    from c2pa_constants import MAX_CBOR_DEPTH, decode_cbor_texts

    depth = MAX_CBOR_DEPTH + 8
    payload = b"\x81" * depth + b"\x01"
    result = decode_cbor_texts(payload)
    assert result.status == "rejected"
    assert result.reason is not None and "depth" in result.reason.lower()
    assert result.texts == ()


def test_cbor_huge_payload_is_stable_rejection():
    from c2pa_constants import MAX_CBOR_PAYLOAD_BYTES, decode_cbor_texts

    result = decode_cbor_texts(b"\x00" * (MAX_CBOR_PAYLOAD_BYTES + 1))
    assert result.status == "rejected"
    assert result.reason is not None and "cap" in result.reason.lower()
    assert result.texts == ()


def test_cbor_indefinite_length_is_stable_rejection():
    from c2pa_constants import decode_cbor_texts

    result = decode_cbor_texts(b"\x7f" + b"\x61a" + b"\xff")
    assert result.status == "rejected"
    assert result.reason is not None and "indefinite" in result.reason.lower()


def test_cbor_item_budget_is_stable_rejection():
    from c2pa_constants import MAX_CBOR_ITEMS, decode_cbor_texts

    payload = b"\x9a" + (MAX_CBOR_ITEMS + 4).to_bytes(4, "big") + b"\x01" * (MAX_CBOR_ITEMS + 4)
    result = decode_cbor_texts(payload)
    assert result.status == "rejected"
    assert result.reason is not None and "budget" in result.reason.lower()


# ---------------------------------------------------------------------------
# caBX scan + fallback wiring in the soft-binding report
# ---------------------------------------------------------------------------


def _cbor_map_for_fixture() -> bytes:
    def text(value: str) -> bytes:
        raw = value.encode("utf-8")
        if len(raw) < 24:
            return bytes([0x60 + len(raw)]) + raw
        return b"\x78" + bytes([len(raw)]) + raw

    return (
        b"\xa2"
        + text("claim_generator")
        + text("Adobe Firefly")
        + text("specVersion")
        + text("1.4")
    )


FAKE_WHEEL_LABEL = "c2pa.soft-binding.synthetic-stub"


def _install_fake_wheel(monkeypatch):
    import types

    class FakeReader:
        def __init__(self, payload: str) -> None:
            self._payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def json(self):
            return self._payload

        def is_embedded(self):
            return True

        def get_remote_url(self):
            return None

        @classmethod
        def try_create(cls, path):
            return cls(
                json.dumps({"manifests": {"m": {"assertions": [{"label": FAKE_WHEEL_LABEL}]}}})
            )

    fake = types.ModuleType("c2pa")
    fake.Reader = FakeReader
    monkeypatch.setitem(sys.modules, "c2pa", fake)


def test_committed_fixture_yields_registry_findings():
    from c2pa_constants import SOFT_BINDING_UNKNOWN_LABEL, scan_c2pa_hints

    fixture = FIXTURES / "claim_generator_cabx.png"
    assert fixture.is_file(), "committed caBX fixture is missing"
    hints = scan_c2pa_hints(fixture)
    assert hints["has_cabx_chunk"] is True and hints["found"] is True
    assert hints["issuers"] == ["Adobe"]
    assert hints["claim_generator"] == "Adobe Firefly"
    assert hints["claim_generator_platforms"] == ["Adobe Firefly"]
    assert hints["ai_tools"] == ["Firefly"]
    assert hints["c2pa_spec"] == "1.4"
    assert hints["soft_binding_algorithms"] == [
        {"algorithm": "example.softbinding.algorithm", "label": SOFT_BINDING_UNKNOWN_LABEL}
    ]


def test_inspect_soft_binding_fallback_without_wheel(monkeypatch):
    import inspect_soft_binding

    monkeypatch.setitem(sys.modules, "c2pa", None)  # import of c2pa now fails: wheel absent
    report = inspect_soft_binding.inspect_soft_binding(FIXTURES / "claim_generator_cabx.png")
    assert report["c2pa_lib"] == {"available": False}
    hints = report["c2pa_hints"]
    assert hints["scanned"] is True and hints["source"] == "stdlib-fallback"
    assert hints["issuers"] == ["Adobe"] and hints["claim_generator"] == "Adobe Firefly"
    # Existing report shape untouched and still consistent with the caBX evidence.
    assert report["has_c2pa"] is True
    assert (
        report["soft_binding"]["found"] is False
    )  # no soft-binding/remote evidence in this fixture


def test_inspect_soft_binding_wheel_stays_authoritative(monkeypatch):
    import inspect_soft_binding

    _install_fake_wheel(monkeypatch)
    report = inspect_soft_binding.inspect_soft_binding(FIXTURES / "claim_generator_cabx.png")
    assert report["c2pa_lib"]["available"] is True
    assert FAKE_WHEEL_LABEL in report["soft_binding"]["labels"]
    hints = report["c2pa_hints"]
    assert hints["scanned"] is False
    assert hints["source"] == "c2pa-wheel-authoritative"
    assert hints["found"] is False  # the stdlib scan cannot contradict the wheel's answer


def test_oversize_cabx_chunk_is_bounded_rejection(monkeypatch, tmp_path):
    import c2pa_constants as cc
    import inspect_soft_binding

    monkeypatch.setattr(cc, "MAX_CABX_CHUNK_BYTES", 16)
    monkeypatch.setitem(sys.modules, "c2pa", None)
    payload = b"jumbf" + _cbor_map_for_fixture()  # 40+ bytes, over the patched cap
    png = build_cabx_png(payload, tmp_path / "oversize.png")
    report = inspect_soft_binding.inspect_soft_binding(png)
    hints = report["c2pa_hints"]
    assert hints["scanned"] is True and hints["has_cabx_chunk"] is False
    assert hints["rejected_reason"] is not None and "cap" in hints["rejected_reason"].lower()


def test_truncated_cbor_inside_cabx_is_stable_rejection(tmp_path):
    from c2pa_constants import scan_c2pa_hints

    payload = (
        b"\xa2"
        + _cbor_text("jumbf")
        + _cbor_text("Adobe Firefly")
        + _cbor_text("specVersion")
        + b"\x63ab"
    )
    png = build_cabx_png(payload, tmp_path / "trunc.png")
    hints = scan_c2pa_hints(png)
    assert hints["has_cabx_chunk"] is True
    assert hints["cbor"]["status"] == "rejected"
    assert hints["cbor"]["reason"] is not None and "truncat" in hints["cbor"]["reason"].lower()
    # Byte-level registry evidence still surfaces: the issuer match survives the CBOR rejection.
    assert hints["issuers"] == ["Adobe"] and hints["found"] is True
