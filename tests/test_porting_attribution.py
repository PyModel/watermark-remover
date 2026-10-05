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
