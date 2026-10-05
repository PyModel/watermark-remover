# Ported from wiltodelta/remove-ai-watermarks src/remove_ai_watermarks/metadata_record.py (collect -> record -> judge pipeline), Apache-2.0; modified: collects this repo's evidence vocabulary — the pure per-format inspectors and the public typed probes — instead of upstream's raw metadata regions, and never runs optional tools or classifiers
"""Collect one file's provenance evidence into a portable, JSON-safe record.

WHY THIS EXISTS

Judgment (``identify``) must be reproducible without the file: collect here,
serialize, judge anywhere. ``collect_provenance_record`` reads the file once and
returns a versioned dict; ``identify.identify_from_record`` turns any such dict
(a JSON round-trip included) into the same verdict the file path produces.

WHAT GETS COLLECTED

The evidence is this repo's own inspection vocabulary, not raw bytes:

- the canonical pure format inspector for the container
  (``image_meta.inspect_png`` / ``inspect_jpeg`` / ... , or
  ``container_meta.inspect_container`` for document containers) — the same
  findings the inspect surface reports;
- the public typed probes over the file bytes: Samsung ``genAIType``, the xAI
  signature pair, the IPTC digital-source token, the Higgsfield job hint;
- the TC260 AIGC label parsed field-aware, with its ``SecurityData`` label
  signature verdict (``tc260_verify.check_tc260_signature``);
- for PNG, the bounded stdlib CBOR/caBX hint scan (``c2pa_constants``).

WHAT DELIBERATELY DOES NOT GET COLLECTED

``image_meta.inspect_image`` is not used: it runs the optional external tools
and the SynthID scorer, which are classifiers. Identification is a metadata
question; classifiers are opt-in elsewhere and never auto-run from here.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # script use: sys.path[0] is already the script dir
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tc260_label
import tc260_verify
from asset_kind import classify_asset
from common import MAX_INPUT_BYTES, read_bytes_bounded

#: Stable transport contract for records produced by this module. Separate from
#: the verdict's schema version: collection and interpretation evolve
#: independently as long as old records stay readable.
PROVENANCE_RECORD_SCHEMA_VERSION = 1
PROVENANCE_RECORD_TYPE = "provenance_record"


class UnsupportedRecordVersion(ValueError):
    """A record (or requested schema) carries a version this build cannot read."""


def require_record_version(
    version: object,
    *,
    contract: str = "provenance record",
    supported: tuple[int, ...] = (PROVENANCE_RECORD_SCHEMA_VERSION,),
) -> int:
    """Return ``version`` when it is an exactly-supported schema version.

    Stable rejection: anything else (wrong type, unknown version) raises
    ``UnsupportedRecordVersion`` naming the contract and the supported
    versions, so a consumer cannot mistake an unreadable record for a verdict.
    """
    if isinstance(version, bool) or not isinstance(version, int) or version not in supported:
        raise UnsupportedRecordVersion(
            f"unsupported {contract} schema version {version!r}; supported: "
            f"{', '.join(str(v) for v in supported)}"
        )
    return version


def _image_inspector(container: str):
    """The pure stdlib inspector for one image container, or None."""
    import image_meta

    return {
        "png": image_meta.inspect_png,
        "jpeg": image_meta.inspect_jpeg,
        "webp": image_meta.inspect_webp,
        "bmp": image_meta.inspect_bmp,
        "gif": image_meta.inspect_gif,
        "tiff": image_meta.inspect_tiff,
        "heif": image_meta.inspect_heic,
        "avif": image_meta.inspect_heic,
    }.get(container)


def _image_evidence(path: Path, data: bytes, container: str) -> dict[str, Any]:
    import image_meta

    inspector = _image_inspector(container)
    if inspector is not None:
        has_c2pa, has_ai, findings, notes = inspector(data)
    else:
        has_c2pa, has_ai, findings, notes = (
            False,
            False,
            [],
            [f"format '{container}' is not inspected"],
        )

    label = tc260_label.label_from_metadata(data) if data else None
    signature = None
    if label is not None:
        checked = tc260_verify.check_tc260_signature(label)
        if checked is not None:
            signature = {"status": checked.label, "signer": checked.signer}

    iptc = image_meta.iptc_digital_source_finding(data)
    iptc_typed = None
    if iptc is not None:
        kind = "enhanced" if "composite" in iptc else "generated"
        token = iptc.split("IPTC digital-source ", 1)[1].split(" ", 1)[0]
        iptc_typed = {"token": token, "kind": kind}

    return {
        "has_c2pa": bool(has_c2pa),
        "has_ai_metadata": bool(has_ai),
        "findings": list(findings),
        "notes": list(notes),
        "tc260_label": dict(label) if label else None,
        "tc260_signature": signature,
        "samsung_genai_type": image_meta.samsung_genai_in(data),
        "xai_pair": any("xAI/Grok provenance marker" in f for f in findings),
        "iptc_digital_source": iptc_typed,
        "hf_job_id": image_meta.hf_job_hint_in(data),
        "generator_products": _generator_products(findings),
        "ai_hints": _ai_hint_tokens(findings),
        "c2pa_hints": _c2pa_hints(path, container),
    }


def _generator_products(findings: list[str]) -> list[str]:
    """Generator product names from ``AI generator (...)`` findings.

    The finding format is built by ``image_meta._generator_product_hits`` and
    asserted across the test suite: ``AI generator (Software=DALL-E)``. Values
    are ``key=product`` pairs from generator-bearing metadata keys; the product
    is what survives the ``=``.
    """
    products: list[str] = []
    for finding in findings:
        if "AI generator (" not in finding:
            continue
        inside = finding.split("AI generator (", 1)[1].rsplit(")", 1)[0]
        for item in inside.split(", "):
            products.append(item.split("=", 1)[-1].strip())
    return list(dict.fromkeys(products))


_AI_TOKENS_NO_AI = frozenset(
    token.casefold()
    for token in (
        "c2pa",
        "C2PA",
        "jumb",
        "JUMB",
        "c2ma",
        "contentcredentials",
        "contentauth",
        "cai:",
    )
)
_C2PA_HINT_TOKENS = _AI_TOKENS_NO_AI | {"http://ns.adobe.com/xmp/instanceid/"}


def _generic_hit_tokens(finding: str) -> list[str] | None:
    """The flat hint tokens of a ``<carrier>: <token, token>`` finding, or None.

    Only findings whose entire payload tokenizes against the known
    hint-marker vocabulary count; anything else (TC260 lines, vendor deltas,
    notes) is not a flat hint line.
    """
    carrier, sep, rest = finding.partition(": ")
    if not sep or not carrier or " " in carrier or len(carrier) > 24:
        return None
    tokens = [t.strip() for t in rest.split(", ")]
    import image_meta

    known = {token.casefold() for token in image_meta.AI_META_HINTS + image_meta.C2PA_MARKERS}
    if tokens and all(token.casefold() in known for token in tokens if token):
        return tokens
    return None


def _ai_hint_tokens(findings: list[str]) -> list[str]:
    """AI hint tokens among the flat generic-hit findings (C2PA names excluded)."""
    found: list[str] = []
    for finding in findings:
        tokens = _generic_hit_tokens(finding)
        if not tokens:
            continue
        for token in tokens:
            if token.casefold() not in _C2PA_HINT_TOKENS and token not in found:
                found.append(token)
    return found


def _c2pa_hints(path: Path, container: str) -> dict[str, Any] | None:
    """The bounded stdlib CBOR/caBX scan for PNGs; None elsewhere.

    The optional ``c2pa`` wheel stays authoritative where it is wired in
    (``inspect_soft_binding``); the record's fallback evidence is the same
    bounded stdlib scan the soft-binding report uses.
    """
    if container != "png":
        return None
    from c2pa_constants import scan_c2pa_hints

    try:
        return scan_c2pa_hints(path)
    except (OSError, ValueError):
        return {"scanned": False, "reason": "scan failed", "cbor": {"status": "skipped"}}


def _container_evidence(path: Path) -> dict[str, Any]:
    """Document containers reuse the canonical container inspector."""
    from container_meta import inspect_container

    report = inspect_container(path).to_dict()
    return {
        "has_c2pa": bool(report.get("has_c2pa")),
        "has_ai_metadata": bool(report.get("has_ai_metadata")),
        "findings": list(report.get("findings") or []),
        "notes": list(report.get("notes") or []),
        "tc260_label": None,
        "tc260_signature": None,
        "samsung_genai_type": None,
        "xai_pair": any("xAI/Grok provenance marker" in f for f in report.get("findings") or []),
        "iptc_digital_source": None,
        "hf_job_id": None,
        "generator_products": _generator_products(report.get("findings") or []),
        "ai_hints": _ai_hint_tokens(report.get("findings") or []),
        "c2pa_hints": None,
    }


_EMPTY_EVIDENCE: dict[str, Any] = {
    "has_c2pa": False,
    "has_ai_metadata": False,
    "findings": [],
    "notes": [],
    "tc260_label": None,
    "tc260_signature": None,
    "samsung_genai_type": None,
    "xai_pair": False,
    "iptc_digital_source": None,
    "hf_job_id": None,
    "generator_products": [],
    "ai_hints": [],
    "c2pa_hints": None,
}


def collect_provenance_record(
    path: Path,
    *,
    schema_version: int = PROVENANCE_RECORD_SCHEMA_VERSION,
    force_kind: str | None = None,
) -> dict[str, Any]:
    """Collect everything the provenance verdict reads, as a JSON-safe record.

    One collection pass over ``path``. The result carries explicit
    ``status``/``issues`` so a consumer cannot mistake a failed read for an
    unknown provenance verdict. Classifiers and optional external tools are
    never invoked.

    Args:
        path: The file to collect from.
        schema_version: Output schema implemented by the consumer.
        force_kind: Override asset classification ("text", "image",
            "container"); ``text`` collects an empty evidence payload — the
            provenance verdict is a metadata question about binary assets.

    Returns:
        A versioned JSON-serializable dict with ``schema_version``,
        ``record_type``, ``status``, ``issues``, ``container`` and ``evidence``.
    """
    require_record_version(
        schema_version, contract="provenance record", supported=(PROVENANCE_RECORD_SCHEMA_VERSION,)
    )
    path = Path(path)
    issues: list[dict[str, str]] = []
    try:
        kind = force_kind or classify_asset(path)
    except OSError as error:
        return _error_record(
            path, schema_version, [{"stage": "source", "code": "unavailable"}], error
        )

    if kind in ("text", "unknown", "refused"):
        record = _base_record(path, schema_version, kind)
        record["evidence"] = dict(_EMPTY_EVIDENCE)
        if kind == "unknown":
            record["issues"].append({"stage": "classify", "code": "unknown-container"})
        return record

    if kind == "container":
        record = _base_record(path, schema_version, kind)
        try:
            record["evidence"] = _container_evidence(path)
        except Exception:
            record["issues"].append({"stage": "container", "code": "collection-failed"})
            record["evidence"] = dict(_EMPTY_EVIDENCE)
        return record

    data = b""
    try:
        data = read_bytes_bounded(path, MAX_INPUT_BYTES, label="provenance record source")
    except (OSError, ValueError) as error:
        issues.append({"stage": "read", "code": "read-failed"})
        record = _base_record(path, schema_version, kind)
        record["status"] = "error"
        record["issues"] = issues
        record["evidence"] = dict(_EMPTY_EVIDENCE)
        del error
        return record

    from image_meta import detect_format

    container = detect_format(data) if kind == "image" else kind
    record = _base_record(path, schema_version, container)
    try:
        record["evidence"] = _image_evidence(path, data, container)
    except Exception:
        record["issues"].append({"stage": "image", "code": "collection-failed"})
        record["evidence"] = dict(_EMPTY_EVIDENCE)
    return record


def _base_record(path: Path, schema_version: int, container: str) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "record_type": PROVENANCE_RECORD_TYPE,
        "status": "complete",
        "issues": [],
        "container": container,
        "name": path.name,
        "evidence": {},
    }


def _error_record(
    path: Path, schema_version: int, issues: list[dict[str, str]], error: Exception
) -> dict[str, Any]:
    del error  # the stable shape reports the stage/code, not the OS message
    record = _base_record(path, schema_version, "unknown")
    record["status"] = "error"
    record["issues"] = issues
    record["evidence"] = dict(_EMPTY_EVIDENCE)
    return record
