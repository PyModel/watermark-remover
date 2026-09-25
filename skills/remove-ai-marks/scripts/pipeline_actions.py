#!/usr/bin/env python3
"""What a clean did to a file, as a stable machine code plus the human line.

Every stripper and cleaner reports its steps as :class:`Action` values.  A
step's ``code`` names *what kind* of step it was and never changes wording;
``params`` carry its counts and names (a part, a chunk, a field, a byte
count); ``text`` is the line the CLI has always printed.  ``effect`` says
whether the step changed the output, was only informational, or warns that a
tool failed or fell back.

A result payload carries the same steps twice, from one list, so the two can
never disagree:

* ``actions`` -- the human lines, unchanged from before codes existed;
* ``action_details`` -- ``{"code", "effect", "text", "params"}`` per step, in
  the same order, for any consumer that must not parse the lines.

Each code has exactly one constructor (a few codes have more than one, where
the historical wording differs), so a call site never pairs a code with a
text by hand.  ``ActionCode`` is the complete list of codes; the TUI keeps a
sentence for every one of them and a test holds it to that.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any


class Effect(str, Enum):
    """What a step did to the output."""

    #: The output differs from the input because of this step.
    CHANGE = "change"
    #: Nothing was removed, or the step was neutral bookkeeping.
    INFO = "info"
    #: A tool failed, fell back or was missing; the result may be weaker.
    WARNING = "warning"
    #: The file could not be cleaned at all.
    ERROR = "error"


class ActionCode(str, Enum):
    """Every step a clean can report.  Values are the wire codes."""

    NOTHING_REMOVED = "nothing_removed"
    # Raster images
    DROP_PNG_CHUNK = "drop_png_chunk"
    DROP_JPEG_SEGMENT = "drop_jpeg_segment"
    PRESERVE_JPEG_SCAN = "preserve_jpeg_scan"
    DROP_WEBP_CHUNK = "drop_webp_chunk"
    DROP_GIF_EXTENSION = "drop_gif_extension"
    DROP_TIFF_TAG = "drop_tiff_tag"
    DROP_BMP_TRAILER = "drop_bmp_trailer"
    KEEP_BMP_TRAILER = "keep_bmp_trailer"
    BMP_UNPARSED = "bmp_unparsed"
    NEUTRALIZE_BOX = "neutralize_box"
    ZERO_PAYLOAD = "zero_payload"
    NEUTRALIZE_TOKENS = "neutralize_tokens"
    EXIFTOOL_STRIP = "exiftool_strip"
    SYNTHID_BAND_REMOVAL = "synthid_band_removal"
    WMCT_MARKER_WRITTEN = "wmct_marker_written"
    WMCT_MARKER_SKIPPED = "wmct_marker_skipped"
    # Text-based containers
    DROP_FRONTMATTER_KEY = "drop_frontmatter_key"
    DROP_EMPTY_FRONTMATTER = "drop_empty_frontmatter"
    CLEAN_DATA_URI = "clean_data_uri"
    DROP_HTML_META = "drop_html_meta"
    DROP_JSON_LD = "drop_json_ld"
    DROP_DATA_AI_ATTRIBUTES = "drop_data_ai_attributes"
    DROP_SVG_METADATA = "drop_svg_metadata"
    DROP_SVG_XMP = "drop_svg_xmp"
    DROP_SVG_COMMENT = "drop_svg_comment"
    DROP_SVG_GENERATOR_ATTRIBUTES = "drop_svg_generator_attributes"
    # Zip containers
    CLEAN_EMBEDDED_MEDIA = "clean_embedded_media"
    CLEAN_PART = "clean_part"
    DROP_PART = "drop_part"
    SCRUB_FIELD = "scrub_field"
    DROP_CONTENT_TYPE_OVERRIDES = "drop_content_type_overrides"
    PRUNE_RELATIONSHIPS = "prune_relationships"
    PRUNE_MANIFEST = "prune_manifest"
    DROP_GENERATOR_META = "drop_generator_meta"
    SCRUB_CREATOR = "scrub_creator"
    DROP_OPF_META = "drop_opf_meta"
    LAYER_A_TEXT = "layer_a_text"
    # PDF
    EXIFTOOL_RUN = "exiftool_run"
    TOOL_FAILED = "tool_failed"
    TOOL_MISSING = "tool_missing"
    TRY_FALLBACK = "try_fallback"
    PDF_ENCRYPTED = "pdf_encrypted"
    PDF_DECRYPTED = "pdf_decrypted"
    DROP_PDF_METADATA = "drop_pdf_metadata"
    PDF_REWRITTEN = "pdf_rewritten"
    PDF_REWRITE_FAILED = "pdf_rewrite_failed"
    PDF_COPIED_UNCHANGED = "pdf_copied_unchanged"
    C2PATOOL_HINT = "c2patool_hint"
    # Visible-mark removal
    VISIBLE_NEEDS_SOURCE = "visible_needs_source"
    VISIBLE_PLAN = "visible_plan"
    MASK_SOURCE = "mask_source"
    REFINE_MASK = "refine_mask"
    EFFECTIVE_MASK = "effective_mask"
    INPAINT_SKIPPED = "inpaint_skipped"
    INPAINT = "inpaint"
    # Visible-mark dry run
    PLAN_LOCALIZE = "plan_localize"
    PLAN_REFINE_MASK = "plan_refine_mask"
    PLAN_INPAINT = "plan_inpaint"
    PLAN_STRIP_METADATA = "plan_strip_metadata"
    PLAN_DEGRADE = "plan_degrade"
    PLAN_PUBLISH = "plan_publish"
    # Whole-file failure
    FAILED = "failed"


@dataclass(frozen=True)
class Action:
    """One step of a clean.  Build it with a constructor below, never directly."""

    code: ActionCode
    effect: Effect
    text: str
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))

    @property
    def changed(self) -> bool:
        return self.effect is Effect.CHANGE

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "effect": self.effect.value,
            "text": self.text,
            "params": {key: _wire(value) for key, value in self.params.items()},
        }


def _wire(value: Any) -> Any:
    if isinstance(value, Action):
        return value.to_dict()
    if isinstance(value, (list, tuple)):
        return [_wire(item) for item in value]
    return value


def report_actions(actions: Sequence[Action]) -> dict[str, list[Any]]:
    """The two payload fields for *actions*: ``actions`` and ``action_details``."""
    return {
        "actions": [action.text for action in actions],
        "action_details": [action.to_dict() for action in actions],
    }


def retarget_report(report: dict[str, Any], old: str, new: str) -> None:
    """Rename *old* to *new* in a report's steps, in both fields.

    For a path a step named while it was staged in a temporary directory and
    that has since been published elsewhere.
    """
    report["actions"] = [text.replace(old, new) for text in report["actions"]]
    report["action_details"] = [_retarget(detail, old, new) for detail in report["action_details"]]


def _retarget(value: Any, old: str, new: str) -> Any:
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, list):
        return [_retarget(item, old, new) for item in value]
    if isinstance(value, dict):
        return {key: _retarget(item, old, new) for key, item in value.items()}
    return value


def any_change(actions: Iterable[Action]) -> bool:
    return any(action.changed for action in actions)


def _act(code: ActionCode, effect: Effect, text: str, **params: Any) -> Action:
    return Action(code, effect, text, {k: v for k, v in params.items() if v is not None})


def _summary(actions: Sequence[Action]) -> str:
    """The legacy parenthetical for a nested clean: its first two lines."""
    return ", ".join(action.text for action in actions[:2])


# --- shared ------------------------------------------------------------------


def nothing_removed(fmt: str, text: str) -> Action:
    """A cleaner found nothing to remove.  *text* is that cleaner's own wording."""
    return _act(ActionCode.NOTHING_REMOVED, Effect.INFO, text, format=fmt)


def layer_a_text(removed: int, replaced: int) -> Action:
    return _act(
        ActionCode.LAYER_A_TEXT,
        Effect.CHANGE,
        f"layer A text: removed={removed} replaced={replaced}",
        removed=removed,
        replaced=replaced,
    )


def failed(error: object) -> Action:
    return _act(ActionCode.FAILED, Effect.ERROR, f"error: {error}", error=str(error))


# --- raster images -----------------------------------------------------------


def drop_png_chunk(chunk: str, *, c2pa_in_payload: bool = False) -> Action:
    reason = "C2PA marker in payload" if c2pa_in_payload else None
    text = f"drop chunk {chunk}" + (f" ({reason})" if reason else "")
    return _act(ActionCode.DROP_PNG_CHUNK, Effect.CHANGE, text, chunk=chunk, reason=reason)


def drop_jpeg_segment(segment: str, *, reason: str | None = None) -> Action:
    """*segment* is ``APPn`` or ``COM``."""
    name = "COM comment" if segment == "COM" else segment
    text = f"drop {name}" + (f" ({reason})" if reason else "")
    return _act(ActionCode.DROP_JPEG_SEGMENT, Effect.CHANGE, text, segment=segment, reason=reason)


def preserve_jpeg_scan() -> Action:
    return _act(
        ActionCode.PRESERVE_JPEG_SCAN, Effect.INFO, "preserved entropy-coded scan through EOI"
    )


def drop_webp_chunk(chunk: str) -> Action:
    return _act(ActionCode.DROP_WEBP_CHUNK, Effect.CHANGE, f"drop WebP chunk {chunk}", chunk=chunk)


def drop_gif_extension(extension: str) -> Action:
    """*extension* is ``comment``, ``XMP application``, ``application`` or ``extension``."""
    return _act(
        ActionCode.DROP_GIF_EXTENSION,
        Effect.CHANGE,
        f"drop GIF {extension} extension",
        extension=extension,
    )


def drop_tiff_tag(tag: int, name: str | None) -> Action:
    """*name* is the tag's metadata name, or None when AI markers alone dropped it."""
    text = f"drop TIFF tag {tag} ({name})" if name else f"drop TIFF tag {tag} (AI markers)"
    return _act(ActionCode.DROP_TIFF_TAG, Effect.CHANGE, text, tag=tag, name=name)


def drop_bmp_trailer(size: int, markers: Sequence[str]) -> Action:
    reason = f" ({', '.join(markers)})" if markers else ""
    return _act(
        ActionCode.DROP_BMP_TRAILER,
        Effect.CHANGE,
        f"drop {size} BMP trailing byte(s){reason}",
        bytes=size,
        markers=list(markers),
    )


def keep_bmp_trailer() -> Action:
    return _act(
        ActionCode.KEEP_BMP_TRAILER, Effect.INFO, "BMP trailing bytes kept (keep-non-ai-metadata)"
    )


def bmp_unparsed() -> Action:
    return _act(
        ActionCode.BMP_UNPARSED, Effect.WARNING, "BMP header not fully parsed; left unchanged"
    )


def neutralize_box(box: str, size: int) -> Action:
    return _act(
        ActionCode.NEUTRALIZE_BOX,
        Effect.CHANGE,
        f"neutralized '{box}' box -> free (zeroed {size} payload bytes)",
        box=box,
        bytes=size,
    )


def zero_uuid_payload(size: int) -> Action:
    return _act(
        ActionCode.ZERO_PAYLOAD,
        Effect.CHANGE,
        f"zeroed XMP uuid box payload ({size} bytes, offsets preserved)",
        target="XMP uuid box",
        bytes=size,
    )


def zero_item_payload(item: str, size: int) -> Action:
    """*item* is ``Exif item`` or ``XMP item``."""
    return _act(
        ActionCode.ZERO_PAYLOAD,
        Effect.CHANGE,
        f"zeroed entire {item} payload ({size} bytes, offsets preserved)",
        target=item,
        bytes=size,
    )


def neutralize_tokens(target: str, tokens: Sequence[str]) -> Action:
    return _act(
        ActionCode.NEUTRALIZE_TOKENS,
        Effect.CHANGE,
        f"neutralized AI tokens in {target}: {', '.join(tokens)}",
        target=target,
        tokens=list(tokens),
    )


def exiftool_strip() -> Action:
    return _act(ActionCode.EXIFTOOL_STRIP, Effect.CHANGE, "exiftool -all= pass")


def synthid_band_removal(strength: float) -> Action:
    return _act(
        ActionCode.SYNTHID_BAND_REMOVAL,
        Effect.CHANGE,
        f"SynthID band removal: strength={strength} (seed-independent DCT suppression)",
        strength=strength,
    )


def wmct_marker_written() -> Action:
    return _act(
        ActionCode.WMCT_MARKER_WRITTEN,
        Effect.CHANGE,
        "wmCt replacement marker written (strip-without-replacement remains the default)",
    )


def wmct_marker_skipped(reason: str) -> Action:
    return _act(
        ActionCode.WMCT_MARKER_SKIPPED,
        Effect.WARNING,
        f"wmCt replacement marker skipped: {reason}",
        reason=reason,
    )


# --- text-based containers ---------------------------------------------------


def drop_frontmatter_key(key: str, *, value_hit: bool = False) -> Action:
    text = (
        f"drop frontmatter key (value hit): {key}" if value_hit else f"drop frontmatter key: {key}"
    )
    return _act(ActionCode.DROP_FRONTMATTER_KEY, Effect.CHANGE, text, key=key, value_hit=value_hit)


def drop_empty_frontmatter() -> Action:
    return _act(ActionCode.DROP_EMPTY_FRONTMATTER, Effect.CHANGE, "removed empty frontmatter block")


def clean_data_uri(mime: str, sub_actions: Sequence[Action]) -> Action:
    return _act(
        ActionCode.CLEAN_DATA_URI,
        Effect.CHANGE,
        f"cleaned embedded data:image/{mime} ({_summary(sub_actions)})",
        mime=f"image/{mime}",
        actions=list(sub_actions),
    )


def drop_html_meta(tag: str) -> Action:
    return _act(ActionCode.DROP_HTML_META, Effect.CHANGE, f"drop meta: {tag[:80]}", tag=tag[:80])


def drop_json_ld() -> Action:
    return _act(ActionCode.DROP_JSON_LD, Effect.CHANGE, "drop json-ld provenance-like script")


def drop_data_ai_attributes(count: int) -> Action:
    return _act(
        ActionCode.DROP_DATA_AI_ATTRIBUTES,
        Effect.CHANGE,
        f"drop data-ai* attributes x{count}",
        count=count,
    )


def drop_svg_metadata(count: int) -> Action:
    return _act(
        ActionCode.DROP_SVG_METADATA, Effect.CHANGE, f"drop <metadata> x{count}", count=count
    )


def drop_svg_xmp(count: int) -> Action:
    return _act(ActionCode.DROP_SVG_XMP, Effect.CHANGE, f"drop xmpmeta x{count}", count=count)


def drop_svg_comment() -> Action:
    return _act(ActionCode.DROP_SVG_COMMENT, Effect.CHANGE, "drop SVG comment with AI markers")


def drop_svg_generator_attributes(count: int) -> Action:
    return _act(
        ActionCode.DROP_SVG_GENERATOR_ATTRIBUTES,
        Effect.CHANGE,
        f"drop generator-like attrs x{count}",
        count=count,
    )


# --- zip containers ----------------------------------------------------------


def clean_embedded_media(part: str, sub_actions: Sequence[Action]) -> Action:
    return _act(
        ActionCode.CLEAN_EMBEDDED_MEDIA,
        Effect.CHANGE,
        f"clean embedded media in {part} ({_summary(sub_actions)})",
        part=part,
        actions=list(sub_actions),
    )


def clean_part(part: str, sub_actions: Sequence[Action]) -> Action:
    """Steps a nested cleaner took inside one part (an EPUB's XHTML or OPF)."""
    return _act(
        ActionCode.CLEAN_PART,
        Effect.CHANGE if any_change(sub_actions) else Effect.INFO,
        f"{part}: {_summary(sub_actions)}",
        part=part,
        actions=list(sub_actions),
    )


def drop_part(part: str, *, markers: bool = False) -> Action:
    """*markers*: the part was dropped because it carries AI/C2PA markers."""
    reason = "AI/C2PA markers" if markers else None
    text = f"drop part {part}" + (f" ({reason})" if reason else "")
    return _act(ActionCode.DROP_PART, Effect.CHANGE, text, part=part, reason=reason)


def scrub_field(part: str, field_name: str) -> Action:
    return _act(
        ActionCode.SCRUB_FIELD,
        Effect.CHANGE,
        f"scrub {part} field {field_name}",
        part=part,
        field=field_name,
    )


def scrub_opf_field(field_name: str) -> Action:
    return _act(
        ActionCode.SCRUB_FIELD,
        Effect.CHANGE,
        f"scrub {field_name} (AI vendor name)",
        field=field_name,
        reason="AI vendor name",
    )


def drop_content_type_overrides(target: str, count: int) -> Action:
    """*target* is ``customXml`` (every customXml part) or ``custom.xml``."""
    text = (
        f"drop Content_Types customXml overrides x{count}"
        if target == "customXml"
        else f"drop Content_Types custom.xml override x{count}"
    )
    return _act(
        ActionCode.DROP_CONTENT_TYPE_OVERRIDES, Effect.CHANGE, text, target=target, count=count
    )


def prune_relationships(part: str, count: int) -> Action:
    return _act(
        ActionCode.PRUNE_RELATIONSHIPS,
        Effect.CHANGE,
        f"prune dangling relationships x{count} in {part}",
        part=part,
        count=count,
    )


def prune_odf_manifest(count: int) -> Action:
    return _act(
        ActionCode.PRUNE_MANIFEST,
        Effect.CHANGE,
        f"drop manifest entries x{count}",
        manifest="ODF",
        count=count,
    )


def prune_opf_manifest(count: int) -> Action:
    return _act(
        ActionCode.PRUNE_MANIFEST,
        Effect.CHANGE,
        f"prune OPF manifest entries x{count}",
        manifest="OPF",
        count=count,
    )


def drop_generator_meta() -> Action:
    return _act(
        ActionCode.DROP_GENERATOR_META,
        Effect.CHANGE,
        "drop meta:generator",
        field="meta:generator",
    )


def scrub_creator() -> Action:
    return _act(
        ActionCode.SCRUB_CREATOR, Effect.CHANGE, "scrub creator-like meta", field="dc:creator"
    )


def drop_opf_meta() -> Action:
    return _act(ActionCode.DROP_OPF_META, Effect.CHANGE, "drop OPF meta tag")


# --- PDF ---------------------------------------------------------------------


def exiftool_run(returncode: int) -> Action:
    """exiftool ran; a nonzero exit means it did not strip cleanly."""
    return _act(
        ActionCode.EXIFTOOL_RUN,
        Effect.CHANGE if returncode == 0 else Effect.WARNING,
        f"exiftool -all= (rc={returncode})",
        returncode=returncode,
    )


def tool_failed(
    tool: str,
    text: str,
    *,
    returncode: int | None = None,
    detail: str | None = None,
    fallback: str | None = None,
) -> Action:
    """*tool* failed.  *text* is the historical line for that failure."""
    return _act(
        ActionCode.TOOL_FAILED,
        Effect.WARNING,
        text,
        tool=tool,
        returncode=returncode,
        detail=detail,
        fallback=fallback,
    )


def exiftool_failed(returncode: int | None, detail: str) -> Action:
    """The raster exiftool pass failed."""
    text = (
        f"exiftool failed (rc={returncode}): {detail}"
        if returncode is not None
        else f"exiftool failed: {detail}"
    )
    return tool_failed("exiftool", text, returncode=returncode, detail=detail)


def tool_missing(tool: str, text: str) -> Action:
    return _act(ActionCode.TOOL_MISSING, Effect.WARNING, text, tool=tool)


def try_fallback(tool: str) -> Action:
    return _act(ActionCode.TRY_FALLBACK, Effect.INFO, f"trying {tool} fallback", tool=tool)


def pdf_encrypted() -> Action:
    return _act(
        ActionCode.PDF_ENCRYPTED,
        Effect.WARNING,
        "encrypted PDF (password required); copied as-is",
    )


def pdf_decrypted() -> Action:
    return _act(ActionCode.PDF_DECRYPTED, Effect.INFO, "decrypted with empty password")


def drop_pdf_page_metadata(pages: int) -> Action:
    return _act(
        ActionCode.DROP_PDF_METADATA,
        Effect.CHANGE,
        f"pypdf: drop per-page /Metadata x{pages}",
        target="page metadata",
        pages=pages,
    )


def drop_pdf_docinfo() -> Action:
    return _act(
        ActionCode.DROP_PDF_METADATA,
        Effect.CHANGE,
        "pypdf: drop document info dictionary",
        target="document info dictionary",
    )


def drop_pdf_catalog_xmp() -> Action:
    return _act(
        ActionCode.DROP_PDF_METADATA,
        Effect.CHANGE,
        "pypdf: drop catalog XMP packet",
        target="catalog XMP packet",
    )


def pdf_rewritten_pypdf() -> Action:
    return _act(
        ActionCode.PDF_REWRITTEN,
        Effect.CHANGE,
        "pypdf: cloned full document graph; removed docinfo/XMP",
        tool="pypdf",
    )


def pdf_rewritten_qpdf(returncode: int) -> Action:
    return _act(
        ActionCode.PDF_REWRITTEN,
        Effect.CHANGE,
        f"qpdf --linearize structural rewrite (rc={returncode})",
        tool="qpdf",
        returncode=returncode,
    )


def pdf_rewrite_failed(
    text: str, *, returncode: int | None = None, detail: str | None = None
) -> Action:
    """No structural rewrite: the old metadata bytes may remain recoverable."""
    return _act(
        ActionCode.PDF_REWRITE_FAILED,
        Effect.WARNING,
        text,
        tool="qpdf",
        returncode=returncode,
        detail=detail,
    )


def pdf_copied_unchanged() -> Action:
    return _act(
        ActionCode.PDF_COPIED_UNCHANGED,
        Effect.WARNING,
        "no structural PDF cleaner succeeded; copied unchanged",
    )


def c2patool_hint() -> Action:
    return _act(
        ActionCode.C2PATOOL_HINT,
        Effect.INFO,
        "c2patool available for inspect; strip via exiftool/re-export",
    )


# --- visible-mark removal ----------------------------------------------------


def visible_needs_source() -> Action:
    return _act(
        ActionCode.VISIBLE_NEEDS_SOURCE, Effect.INFO, "supply --mask, --box, or --detect-command"
    )


def visible_plan() -> Action:
    return _act(
        ActionCode.VISIBLE_PLAN,
        Effect.INFO,
        "then refine/fill holes, dilate d=3, inpaint, restore original outside mask",
    )


def mask_source(source: str) -> Action:
    return _act(ActionCode.MASK_SOURCE, Effect.INFO, f"mask source: {source}", source=source)


def refine_mask(dilation_radius: int, before: int, after: int) -> Action:
    return _act(
        ActionCode.REFINE_MASK,
        Effect.INFO,
        f"fill holes + dilate radius={dilation_radius}: {before}->{after} pixels",
        dilation_radius=dilation_radius,
        pixels_before=before,
        pixels_after=after,
    )


def effective_mask(before: int, after: int, published: str | None) -> Action:
    """*published* is the mask file written, or None when it stayed in memory."""
    where = f" (published {published})" if published else " (not published)"
    return _act(
        ActionCode.EFFECTIVE_MASK,
        Effect.INFO,
        f"effective mask: {before}->{after} pixels{where}",
        pixels_before=before,
        pixels_after=after,
        published=published,
    )


def inpaint_skipped() -> Action:
    return _act(ActionCode.INPAINT_SKIPPED, Effect.INFO, "no inpainting run (print-plan backend)")


def inpaint_texture(x: int, y: int, width: int, height: int, edge_mse: float) -> Action:
    return _act(
        ActionCode.INPAINT,
        Effect.CHANGE,
        f"texture-patch inpaint source=({x},{y},{width},{height}) edge_mse={edge_mse:.2f}",
        backend="texture",
        source_patch=[x, y, width, height],
        edge_mse=round(edge_mse, 2),
    )


def inpaint_simple() -> Action:
    return _act(
        ActionCode.INPAINT,
        Effect.CHANGE,
        "nearest-boundary inpaint + restore (uniform-background fallback)",
        backend="simple",
    )


def inpaint_external() -> Action:
    return _act(
        ActionCode.INPAINT,
        Effect.CHANGE,
        "external inpaint + stdlib restore outside mask",
        backend="external",
    )


# --- visible-mark dry run ----------------------------------------------------


def plan_localize(source: str) -> Action:
    return _act(
        ActionCode.PLAN_LOCALIZE,
        Effect.INFO,
        f"localize visible mark via {source}",
        source=source,
    )


def plan_refine_mask(dilation_radius: int) -> Action:
    return _act(
        ActionCode.PLAN_REFINE_MASK,
        Effect.INFO,
        f"fill holes + dilate radius={dilation_radius}",
        dilation_radius=dilation_radius,
    )


def plan_inpaint(backend: str) -> Action:
    return _act(
        ActionCode.PLAN_INPAINT, Effect.INFO, f"inpaint with {backend} backend", backend=backend
    )


def plan_strip_metadata() -> Action:
    return _act(ActionCode.PLAN_STRIP_METADATA, Effect.INFO, "strip requested metadata")


def plan_degrade(strategy: str) -> Action:
    return _act(
        ActionCode.PLAN_DEGRADE,
        Effect.INFO,
        f"apply {strategy} degradation",
        strategy=strategy,
    )


def plan_publish(mask: str, image: str) -> Action:
    return _act(
        ActionCode.PLAN_PUBLISH,
        Effect.INFO,
        f"publish mask to {mask} and image to {image}",
        mask=mask,
        image=image,
    )
