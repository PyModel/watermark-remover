"""Attribution infrastructure: NOTICE and the per-file provenance convention.

The porting effort brings in Apache-2.0 code from
`wiltodelta/remove-ai-watermarks` behind this repo's MIT license.  A
redistributor must be able to see exactly what was ported and from whom, so
these repo-hygiene checks keep the attribution artifacts present and honest.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_notice_names_upstream_and_license() -> None:
    """The root NOTICE must exist and name the upstream project + license."""
    notice = ROOT / "NOTICE"
    assert notice.is_file(), "NOTICE missing at repo root"
    text = notice.read_text(encoding="utf-8")
    assert "wiltodelta/remove-ai-watermarks" in text, "NOTICE must name the upstream project"
    assert "Apache" in text and "2.0" in text, "NOTICE must state Apache-2.0"


def test_porting_docs_define_the_provenance_header_convention() -> None:
    """The documented per-file header convention must state its exact shape.

    Ported files start with:
    ``Ported from wiltodelta/remove-ai-watermarks <path>, Apache-2.0; modified: <what>``
    so a redistributor can trace every ported line without reading history.
    """
    contributing = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    marker = "Ported from wiltodelta/remove-ai-watermarks <path>, Apache-2.0; modified: <what>"
    assert marker in contributing, (
        "CONTRIBUTING.md must document the per-file provenance header convention"
    )
