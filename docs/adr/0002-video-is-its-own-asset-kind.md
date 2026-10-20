# Video is its own asset kind and refuses closed when its runtime is absent

Video (mp4, mov, m4v, webm, mkv, avi, flv) joins the routing contract as a fourth
asset kind, decided by extension and magic bytes before any runtime check. When
ffmpeg, ffprobe, or the video extra is absent, clean refuses the file with a
shell-quoted install hint and writes nothing. It never falls through to
`unknown` and never to `text`.

The obvious alternative was to leave video files as `unknown`, which clean
already refuses. That was rejected because a known kind with a missing
dependency is a different fact from an unrecognized file, and the hint is the
only thing that tells the operator the difference. This extends "unknown
binaries are never mangled as text" rather than weakening it. It is hard to
reverse because the kind list and the refusal are part of the public routing
contract shared by the CLI, the TUI, and the HTTP service.
