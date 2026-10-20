# Metadata and provenance

How `wm` strips C2PA, EXIF, XMP, and document properties, and where it stops.

```bash
# PNG, JPEG, HEIC, HEIF, AVIF
python3 "$SCRIPTS/clean_image.py" photo.heic -o photo.cleaned.heic

# Any supported format
python3 "$SCRIPTS/clean_file.py" document.pdf -o document.cleaned.pdf

# Detect soft-binding or remote-manifest risk
python3 "$SCRIPTS/inspect_soft_binding.py" image.png --json
```

C2PA facts the parser relies on:

- Manifest Stores use JUMBF and BMFF structures, claims and assertions, and COSE signatures.
- JPEG embeds through APP11/JUMBF.
- PNG uses the private ancillary `caBX` chunk, not generic `tEXt`.
- HEIF and AVIF are ISO-BMFF. Cleaning neutralizes matching JUMBF/C2PA UUID boxes and direct-file Exif and XMP extents in place, preserving offsets. Unsupported external or idat metadata layouts fail closed.

Soft-binding removal remains out of scope. The inspector warns when an in-content binding or remote manifest may re-link provenance after hard-bound metadata is stripped.

### PDF quality

PDF cleaning uses this fallback order:

1. `exiftool`
2. `qpdf --linearize` structural rewrite (when qpdf is present)
3. optional `pypdf` full-document clone (outlines, forms, attachments, and catalog retained, docinfo and XMP removed)
4. byte-exact unchanged copy with an explicit residual warning

```bash
python3 -m pip install pypdf
```

Encrypted PDFs are never regex-edited.
