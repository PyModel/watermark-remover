#!/usr/bin/env python3
"""Detect C2PA soft bindings and remote-manifest references (detection only).

Hard-bound metadata stripping (what clean_image/clean_file do) does NOT defeat:
  - soft bindings: an in-content watermark whose assertion lets a verifier
    re-link a remote Content Credentials manifest after metadata removal
  - remote manifests referenced by URL inside the manifest store

This tool reports their presence so the user knows a strip may be re-linked.
Stdlib byte-scan always runs; if the optional `c2pa` package is installed, its
reader is tried first for authoritative manifest JSON.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import eprint, read_bytes_bounded
from image_meta import C2PA_MARKERS

MAX_SCAN_BYTES = 256 * 1024 * 1024

SOFT_BINDING_HINTS = (
    b"c2pa.soft-binding",
    b"soft-binding",
    b"softbinding",
    b"c2pa.remote-manifest",
    b"remote-manifest",
    b"remote_manifest",
)

_URL_RE = re.compile(rb"https?://[^\s\"'<>\\)\]]+")
_MANIFESTISH = re.compile(rb"manifest|credential|c2pa|contentauth|verify", re.I)
_STRUCTURED_LABEL_RE = re.compile(
    rb"(?:[\"']label[\"']\s*:\s*[\"']|\blabel\s*=\s*[\"'])"
    rb"(c2pa\.(?:soft-binding|remote-manifest))[\"']",
    re.I,
)


def _scan_bytes(data: bytes) -> dict[str, Any]:
    lower = data.lower()
    has_c2pa = any(n.lower() in lower for n in C2PA_MARKERS)
    labels = sorted(
        {match.group(1).decode("ascii").lower() for match in _STRUCTURED_LABEL_RE.finditer(data)}
    )

    # Ordinary embedded manifests contain certificate, OCSP, vocabulary, and
    # assertion URLs. Those are not remote-manifest evidence. Only retain URLs
    # near an explicit soft/remote-binding marker in the raw fallback scanner.
    urls: set[str] = set()
    url_hints = [
        *(b"c2pa.remote-manifest", b"remote-manifest", b"remote_manifest"),
        *(label.encode("ascii") for label in labels),
    ]
    for hint in url_hints:
        start = 0
        while (index := lower.find(hint.lower(), start)) >= 0:
            window = data[max(0, index - 256) : min(len(data), index + 4096)]
            urls.update(
                match.group(0).decode("ascii", errors="replace")
                for match in _URL_RE.finditer(window)
                if _MANIFESTISH.search(match.group(0))
            )
            start = index + len(hint)
    return {"has_c2pa": has_c2pa, "labels": labels, "urls": sorted(urls)[:20]}


def _try_c2pa_lib(path: Path) -> dict[str, Any] | None:
    try:
        import c2pa  # type: ignore
    except ImportError:
        return None
    try:
        reader = c2pa.Reader(str(path))
        manifest = json.loads(reader.json())
    except Exception as e:
        return {"available": True, "error": str(e)[:300]}
    labels: list[str] = []
    for man in (manifest.get("manifests") or {}).values():
        for assertion in man.get("assertions") or []:
            label = str(assertion.get("label", ""))
            if "soft" in label.lower() or "remote" in label.lower():
                labels.append(label)
    return {"available": True, "soft_binding_labels": labels}


def inspect_soft_binding(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    scan = _scan_bytes(data)
    lib = _try_c2pa_lib(path)
    if lib and lib.get("soft_binding_labels"):
        for lbl in lib["soft_binding_labels"]:
            if lbl not in scan["labels"]:
                scan["labels"].append(lbl)
        scan["has_c2pa"] = True

    found = bool(scan["has_c2pa"] and (scan["labels"] or scan["urls"]))
    return {
        "path": str(path),
        "has_c2pa": scan["has_c2pa"],
        "soft_binding": {
            "found": found,
            "labels": scan["labels"],
            "manifest_urls": scan["urls"],
        },
        "c2pa_lib": lib or {"available": False},
        "warning": (
            "Soft binding / remote manifest detected: stripping embedded metadata "
            "may NOT remove provenance — a verifier can re-link the remote manifest "
            "or recover via the in-content watermark."
            if found
            else None
        ),
        "note": (
            "Detection only. This project does not remove in-content (soft-bound) "
            "watermarks; see DESIGN.md out-of-scope."
        ),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", type=Path)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    if not args.path.is_file():
        eprint(f"not a file: {args.path}")
        return 2
    report = inspect_soft_binding(args.path)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        sb = report["soft_binding"]
        print(f"Path: {report['path']}")
        print(f"C2PA present: {report['has_c2pa']}")
        print(f"Soft binding found: {sb['found']}")
        for lbl in sb["labels"]:
            print(f"  label: {lbl}")
        for url in sb["manifest_urls"]:
            print(f"  url: {url}")
        if report["warning"]:
            eprint(f"warning: {report['warning']}")
    return 1 if report["soft_binding"]["found"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
