#!/usr/bin/env python3
"""Optional MarkLLM text-watermark harness backed by an external THU-BPM/MarkLLM checkout.

This script does NOT vendor upstream code. It imports ``AutoWatermark`` from a
user-provided checkout (https://github.com/THU-BPM/MarkLLM) at runtime, using
that environment's optional dependencies (torch, transformers, datasets, ...).

MarkLLM is Apache-2.0. It is a research/verification harness: detection is only
valid against the SAME scheme config + keys used at generation. It cannot
certify that a vendor detector will fail on the given text.

Decision model:
  A detector may say what it observed; only the decision engine may say what
  that observation means. Each report therefore carries:
    - detector_verdict: the raw MarkLLM observation (DETECTED / NOT_DETECTED)
    - verdict: the document-level decision, one of
        DETECTED       score at/above threshold
        NOT_DETECTED   below threshold AND provenance/key match confirmed AND
                       enough scored tokens
        INCONCLUSIVE   below threshold but provenance unknown/mismatched, or
                       score inside the abstention band, or too short
        UNSUPPORTED    document provenance declares a scheme we cannot run
        ERROR          detector returned no score/threshold
  A negative with unknown provenance must never be read as "watermark-free".
  Provenance comes from a sidecar (<input>.wm.json) written by the `watermark`
  subcommand, or from an operator assertion via --key-id.

Subcommands:
  detect    run detection on a text file with a known scheme/config
  watermark generate watermarked (and optionally unwatermarked) sample text
            from a prompt, for controlled before/after experiments

Exit codes:
  0  success
  1  runtime error (model load, detection/generation failure)
  2  bad input (missing/unreadable file, binary input, bad args)
  3  unavailable (not configured / missing checkout / missing deps)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from common import (
    atomic_write_text,
    emit_json,
    eprint,
    read_text_input,
    which,
)

# Scheme name as the user types it -> MarkLLM algorithm name (config/{ALG}.json).
SCHEMES = {
    "kgw": "KGW",
    "synthid": "SynthID",
    "synthid-text": "SynthID",
}

DEFAULT_MODEL = "facebook/opt-1.3b"

# Document-level verdicts produced by the decision engine. A detector may say
# what it observed; only the decision engine may say what that observation
# means. The engine never lets a negative with unknown provenance, a missing
# score, an unsupported scheme, or a too-short sample collapse into "clean".
VERDICT_DETECTED = "DETECTED"
VERDICT_NOT_DETECTED = "NOT_DETECTED"
VERDICT_INCONCLUSIVE = "INCONCLUSIVE"
VERDICT_UNSUPPORTED = "UNSUPPORTED"
VERDICT_ERROR = "ERROR"

# Relative abstention band around the config threshold. Scores within
# |score - threshold| / threshold < band are treated as insufficient evidence
# (SynthID's published design abstains to preserve target error rates; KGW's
# z-test is likewise unreliable in the near-threshold region).
DEFAULT_ABSTENTION_BAND = 0.10

# Below this many tokens, a negative is not a strong negative even with
# matching provenance: KGW/SynthID detect via low-stakes word choices, which
# are too few in short text (measured: 677 chars -> miss, 1912 chars -> hit).
DEFAULT_MIN_TOKENS = 200

# Provenance sidecar suffix: <input>.wm.json records scheme/key/config_hash
# for text generated under a known key, making detection deterministic.
SIDECAR_SUFFIX = ".wm.json"

# Algorithm configs are ~200 B (KGW/SynthID). Cap well above that so a crafted
# or accidental huge file is refused before either this script or upstream
# reads it into memory.
MAX_CONFIG_BYTES = 1 << 20


class _Unavailable(RuntimeError):
    """Backend present but unusable (unconfigured checkout, missing deps)."""


def resolve_upstream(raw: str | None) -> Path | None:
    if not raw:
        return None
    upstream = Path(raw).expanduser().resolve()
    if not upstream.is_dir():
        return None
    return upstream


def resolve_device(raw: str | None) -> str:
    """Resolve the ``auto`` device hint to a concrete torch device."""
    if raw and raw != "auto":
        return raw
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        # Never auto-select mps: SynthID/KGW build torch.Generator(device=...),
        # which supports only cpu/cuda and raises RuntimeError on 'mps' (Apple
        # Silicon). Fall through to cpu. Pass --device mps explicitly to override.
    except Exception:  # noqa: S110
        pass
    return "cpu"


def _load_algorithm(
    upstream: Path, alg: str, config: Path, model: str, device: str, offline: bool = False
):
    """Import the checkout and build an ``AutoWatermark`` instance.

    Returns ``(watermark, tokenizer)``; the tokenizer is needed for the
    decision engine's token-length calibration.
    """
    sys.path.insert(0, str(upstream))
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from utils.transformers_config import TransformersConfig
        from watermark.auto_watermark import AutoWatermark
    except ImportError as e:
        raise _Unavailable(f"MarkLLM dependencies missing: {e}") from e

    # --offline: never contact the HF hub. local_files_only makes transformers
    # fail fast instead of hanging, and HF_HUB_OFFLINE covers the lower-level
    # hub calls. Note: the operator-supplied MarkLLM checkout imported through
    # watermark.auto_watermark is trusted and executes locally; offline mode
    # only prevents Hugging Face Hub downloads and transformers remote-code
    # loading (trust_remote_code is never enabled here).
    if offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    load_kwargs = {"local_files_only": True} if offline else {}

    tokenizer = AutoTokenizer.from_pretrained(model, **load_kwargs)
    lm = AutoModelForCausalLM.from_pretrained(model, **load_kwargs).to(device)
    transformers_config = TransformersConfig(
        model=lm,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=200,
        min_length=0,
        do_sample=True,
        no_repeat_ngram_size=4,
    )
    wm = AutoWatermark.load(
        alg,
        algorithm_config=str(config),
        transformers_config=transformers_config,
    )
    return wm, tokenizer


def _threshold_from_config(config: Path) -> float | None:
    try:
        data = json.loads(config.read_text("utf-8"))
    except (OSError, ValueError):
        return None
    for key in ("threshold", "z_threshold"):
        value = data.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _sha256_file(path: Path) -> str:
    """Stable content hash of a config file, for provenance matching."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit(upstream: Path) -> str | None:
    """Best-effort pinned-commit id of the MarkLLM checkout (contract version)."""
    git = which("git")
    if git is None:
        return None
    try:
        import subprocess

        out = subprocess.run(
            [git, "-C", str(upstream), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    commit = out.stdout.strip()
    return commit or None


def _sidecar_path_for(input_path: str) -> Path | None:
    """Return <input>.wm.json if a provenance sidecar exists next to the input."""
    if input_path == "-":
        return None
    sidecar = Path(input_path + SIDECAR_SUFFIX)
    if not sidecar.is_file():
        return None
    try:
        if sidecar.stat().st_size > MAX_CONFIG_BYTES:
            return None
    except OSError:
        return None
    return sidecar


def _read_sidecar(path: Path) -> dict:
    """Read a provenance sidecar; malformed sidecars degrade to {} (unknown)."""
    try:
        data = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _count_tokens(tokenizer: object, text: str) -> int | None:
    """Count input tokens via the loaded tokenizer; None if unavailable."""
    encode = getattr(tokenizer, "encode", None)
    if encode is None:
        return None
    try:
        ids = encode(text)
        if isinstance(ids, int):
            return ids
        return len(ids)
    except Exception:
        return None


def _decide(
    *,
    detector_verdict: str,
    score: float | None,
    threshold: float | None,
    provenance_match: bool | None,
    input_tokens: int | None,
    min_tokens: int,
    abstention_band: float,
) -> tuple[str, str]:
    """Map detector observations + provenance to a document-level verdict.

    The detector says what it observed (``detector_verdict``); this is the
    decision engine that decides what the observation means. The invariant:
    a negative must never become "clean" when the key/provenance is unknown,
    the score sits in the abstention band, or the sample is too short.
    """
    if detector_verdict == VERDICT_DETECTED:
        # Aggressive posture: a positive detector observation is always
        # reported as DETECTED, even when its score sits inside the
        # abstention band or the config exposes no threshold. The band
        # and min-length checks only qualify negatives.
        if score is not None and threshold is not None:
            return (
                VERDICT_DETECTED,
                f"score {score:.4f} at/above threshold {threshold:.4f}",
            )
        return (VERDICT_DETECTED, "detector reported watermarked")
    if score is None or threshold is None:
        return VERDICT_ERROR, "detector returned no score or threshold"
    if abstention_band > 0 and threshold != 0:
        rel = abs(score - threshold) / abs(threshold)
        if rel < abstention_band:
            return (
                VERDICT_INCONCLUSIVE,
                f"score {score:.4f} within {abstention_band:.0%} abstention band "
                f"of threshold {threshold:.4f}",
            )
    if provenance_match is not True:
        return (
            VERDICT_INCONCLUSIVE,
            "detector below threshold but provenance/key match not confirmed",
        )
    if input_tokens is not None and input_tokens < min_tokens:
        return (
            VERDICT_INCONCLUSIVE,
            f"below minimum calibrated length ({input_tokens} < {min_tokens} tokens)",
        )
    return (
        VERDICT_NOT_DETECTED,
        f"score {score:.4f} below threshold {threshold:.4f} with confirmed provenance/key match",
    )


def _resolve_config(upstream: Path, alg: str, config: str | None) -> Path:
    path = Path(config).expanduser().resolve() if config else upstream / "config" / f"{alg}.json"
    if not path.is_file():
        raise _Unavailable(f"MarkLLM config not found: {path}")
    try:
        size = path.stat().st_size
    except OSError as e:
        raise _Unavailable(f"cannot stat MarkLLM config {path}: {e}") from e
    if size > MAX_CONFIG_BYTES:
        raise _Unavailable(f"MarkLLM config too large ({size} bytes > {MAX_CONFIG_BYTES}): {path}")
    return path


def _cmd_detect(args: argparse.Namespace, upstream: Path, alg: str) -> int:
    if args.path != "-" and not Path(args.path).is_file():
        eprint(f"not a file: {args.path}")
        return 2
    text = read_text_input(args.path, allow_binary=args.force_text)
    sidecar_path = _sidecar_path_for(args.path)

    device = resolve_device(args.device)

    try:
        config = _resolve_config(upstream, alg, args.config)
        threshold = _threshold_from_config(config)
    except _Unavailable as e:
        eprint(str(e))
        return 3

    # --- Provenance resolution (decision engine input) ---
    # A sidecar (<input>.wm.json) records how text generated under a known key
    # was marked. Without it, provenance is UNKNOWN unless the operator asserts
    # a key identity with --key-id. Resolved before the (expensive) model load
    # so UNSUPPORTED provenance fails fast.
    sidecar = _read_sidecar(sidecar_path) if sidecar_path is not None else {}
    sidecar_scheme = sidecar.get("scheme")
    sidecar_scheme_l = str(sidecar_scheme).lower() if sidecar_scheme else None
    sidecar_config_hash = sidecar.get("config_hash")
    config_hash = _sha256_file(config)

    # UNSUPPORTED: the document's provenance declares a scheme we cannot run.
    if sidecar_scheme is not None and sidecar_scheme_l not in SCHEMES:
        payload = {
            "available": True,
            "upstream_dir": str(upstream),
            "scheme": alg,
            "config": str(config),
            "model": args.model,
            "device": device,
            "verdict": VERDICT_UNSUPPORTED,
            "verdict_reason": f"document provenance declares unsupported scheme '{sidecar_scheme}'",
            "detector_verdict": None,
            "is_watermarked": None,
            "score": None,
            "threshold": threshold,
            "provenance_match": False,
            "provenance": sidecar,
            "config_hash": config_hash,
            "implementation_commit": args._implementation_commit,
            "input_chars": len(text),
            "input_tokens": None,
            "effective_scored_tokens": None,
        }
        if args.json:
            emit_json(payload)
        else:
            print(f"{alg}: {VERDICT_UNSUPPORTED} (unsupported scheme '{sidecar_scheme}')")
        return 0

    try:
        wm, tokenizer = _load_algorithm(
            upstream, alg, config, args.model, device, offline=args.offline
        )
        result = wm.detect_watermark(text, return_dict=True)
    except _Unavailable as e:
        eprint(str(e))
        return 3
    except Exception as e:
        eprint(f"detection error: {e}")
        return 1

    is_watermarked = bool(result.get("is_watermarked", False))
    score = result.get("score")
    try:
        score = float(score)
    except (TypeError, ValueError):
        score = None

    key_id = args.key_id or sidecar.get("key_id")
    # provenance_match: True only when the sidecar's config matches the config
    # we detected with (content hash), or the operator asserted a key id.
    if sidecar_config_hash is not None:
        provenance_match = sidecar_config_hash == config_hash
    elif key_id is not None:
        provenance_match = True
    else:
        provenance_match = None
    # A sidecar declaring a different *supported* scheme is still a mismatch
    # for the requested detector. Compare normalized algorithm names so the
    # synthid/synthid-text aliases match each other.
    if (
        sidecar_scheme is not None
        and SCHEMES.get(sidecar_scheme_l) is not None
        and SCHEMES.get(sidecar_scheme_l) != alg
    ):
        provenance_match = False

    detector_verdict = VERDICT_DETECTED if is_watermarked else VERDICT_NOT_DETECTED
    input_tokens = _count_tokens(tokenizer, text)
    effective = result.get("num_tokens")
    try:
        effective = int(effective) if effective is not None else None
    except (TypeError, ValueError):
        effective = None
    if effective is None:
        effective = input_tokens

    verdict, reason = _decide(
        detector_verdict=detector_verdict,
        score=score,
        threshold=threshold,
        provenance_match=provenance_match,
        input_tokens=input_tokens,
        min_tokens=args.min_tokens,
        abstention_band=args.abstention_band,
    )

    payload = {
        "available": True,
        "upstream_dir": str(upstream),
        "scheme": alg,
        "config": str(config),
        "model": args.model,
        "device": device,
        "verdict": verdict,
        "verdict_reason": reason,
        "detector_verdict": detector_verdict,
        "is_watermarked": is_watermarked,
        "score": score,
        "threshold": threshold,
        "provenance_match": provenance_match,
        "key_id": key_id,
        "provenance": sidecar or None,
        "config_hash": config_hash,
        "implementation_commit": args._implementation_commit,
        "input_chars": len(text),
        "input_tokens": input_tokens,
        "effective_scored_tokens": effective,
        "min_tokens": args.min_tokens,
        "abstention_band": args.abstention_band,
    }

    if args.json:
        emit_json(payload)
    else:
        label = {
            VERDICT_DETECTED: "detected",
            VERDICT_NOT_DETECTED: "not detected",
            VERDICT_INCONCLUSIVE: "inconclusive",
            VERDICT_ERROR: "error",
            VERDICT_UNSUPPORTED: "unsupported",
        }.get(verdict, verdict)
        score_txt = f"{score:.4f}" if score is not None else "n/a"
        thresh_txt = f"{threshold:.4f}" if threshold is not None else "n/a"
        pm = (
            "matched"
            if provenance_match is True
            else "mismatched"
            if provenance_match is False
            else "unknown"
        )
        print(
            f"{alg}: {label} (score {score_txt}, threshold {thresh_txt}, "
            f"provenance/key match: {pm})"
        )
        if verdict == VERDICT_INCONCLUSIVE:
            print(f"  reason: {reason}")
        if verdict != VERDICT_DETECTED:
            print("  This result does NOT establish that the document is watermark-free.")

    return 0


def _cmd_watermark(args: argparse.Namespace, upstream: Path, alg: str) -> int:
    prompt = read_text_input(args.prompt, allow_binary=args.force_text)

    device = resolve_device(args.device)

    try:
        config = _resolve_config(upstream, alg, args.config)
        wm, _tokenizer = _load_algorithm(
            upstream, alg, config, args.model, device, offline=args.offline
        )
        if args.seed is not None:
            import torch

            torch.manual_seed(args.seed)
        wm.config.gen_kwargs["max_new_tokens"] = args.max_new_tokens
        wm.config.gen_kwargs["min_length"] = args.min_length
        watermarked = wm.generate_watermarked_text(prompt)
        unwatermarked = None
        if args.unwatermarked_output:
            unwatermarked = wm.generate_unwatermarked_text(prompt)
    except _Unavailable as e:
        eprint(str(e))
        return 3
    except Exception as e:
        eprint(f"generation error: {e}")
        return 1

    wm_out = "-" if args.watermarked_output is None else args.watermarked_output
    if wm_out == "-":
        # Never mix generated text with the --json payload on stdout: a batch
        # consumer json.loads()ing stdout would choke on the sample. In JSON
        # mode the sample goes to stderr; non-JSON mode keeps the CLI contract
        # of writing the sample to stdout.
        (sys.stderr if args.json else sys.stdout).write(watermarked)
    else:
        atomic_write_text(Path(wm_out), watermarked)
        # Provenance sidecar: record how this sample was marked so detection
        # of it later is deterministic instead of guesswork. key_id is an
        # identifier, never the raw secret. The full config is embedded so a
        # later detection can verify config_hash without re-reading the file.
        config = _resolve_config(upstream, alg, args.config)
        try:
            params = json.loads(config.read_text("utf-8"))
        except (OSError, ValueError):
            params = None
        sidecar = {
            "scheme": alg,
            "key_id": args.key_id,
            "tokenizer": args.model,
            "watermark_parameters": params,
            "config_hash": _sha256_file(config),
            "generator": args.model,
            "generation_settings": {
                "max_new_tokens": args.max_new_tokens,
                "min_length": args.min_length,
                "seed": args.seed,
            },
            "implementation_commit": args._implementation_commit,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        atomic_write_text(Path(wm_out + SIDECAR_SUFFIX), json.dumps(sidecar, indent=2))
    if unwatermarked is not None:
        atomic_write_text(Path(args.unwatermarked_output), unwatermarked)

    payload = {
        "available": True,
        "upstream_dir": str(upstream),
        "scheme": alg,
        "config": str(config),
        "model": args.model,
        "device": device,
        "watermarked_output": wm_out,
        "unwatermarked_output": args.unwatermarked_output,
        "watermarked_chars": len(watermarked),
        "unwatermarked_chars": len(unwatermarked) if unwatermarked is not None else None,
    }

    if args.json:
        emit_json(payload)
    else:
        print(f"{alg}: watermarked sample ({payload['watermarked_chars']} chars) -> {wm_out}")
        if unwatermarked is not None:
            print(
                f"      unwatermarked sample ({payload['unwatermarked_chars']} chars) -> {args.unwatermarked_output}"
            )

    return 0


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--upstream-dir",
        type=Path,
        default=None,
        help="MarkLLM checkout root (default: $MARKLLM_DIR)",
    )
    p.add_argument(
        "--rlimit-as",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--scheme",
        required=True,
        choices=sorted(SCHEMES),
        help="Watermark scheme to use (kgw, synthid)",
    )
    p.add_argument(
        "--config",
        default=None,
        help="Algorithm config JSON (default: <checkout>/config/<ALG>.json)",
    )
    p.add_argument(
        "--model",
        default=os.environ.get("MARKLLM_MODEL", DEFAULT_MODEL),
        help=f"HF causal LM for scoring (default: $MARKLLM_MODEL or {DEFAULT_MODEL})",
    )
    p.add_argument(
        "--device",
        default="auto",
        help="auto|cpu|cuda|mps (default: auto)",
    )
    p.add_argument(
        "--offline",
        action="store_true",
        help="Never contact the HF hub: load the scoring model from the local "
        "cache only (fails fast if not cached)",
    )
    p.add_argument(
        "--force-text",
        action="store_true",
        help="Process input even when it looks like a binary container",
    )
    p.add_argument(
        "--key-id",
        default=None,
        help="Identifier of the watermark key this text was generated under "
        "(never the raw secret). Establishes provenance_match for detection; "
        "recorded in the sidecar by `watermark`.",
    )


def _apply_rlimit_as(bytes_: int | None) -> None:
    """Cap this process's address space before any heavy import (POSIX only).

    Runs as the first thing the child does, so the limit is in force for the
    whole MarkLLM harness (torch, transformers, ...) — equivalent to a
    preexec_fn-set rlimit, applied at the subprocess boundary. The adapter
    (text_detectors.MarkLLMTextDetector) passes --rlimit-as when
    WATERMARKS_MARKLLM_RLIMIT_AS is configured. Failures degrade silently,
    matching common.subprocess_rlimits().
    """
    if bytes_ is None:
        return
    try:
        import resource
    except ImportError:
        return
    try:
        resource.setrlimit(resource.RLIMIT_AS, (bytes_, bytes_))
    except ValueError:
        # macOS rejects lowering the hard limit while the current soft limit
        # is still RLIM_INFINITY; lower the soft limit first, then the hard.
        try:
            resource.setrlimit(resource.RLIMIT_AS, (bytes_, resource.RLIM_INFINITY))
            resource.setrlimit(resource.RLIMIT_AS, (bytes_, bytes_))
        except (OSError, ValueError):
            pass
    except OSError:
        pass


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    detect = sub.add_parser("detect", help="Detect a scheme watermark in text")
    detect.add_argument("path", help="Text file to detect on, or - for stdin")
    _add_common(detect)
    detect.add_argument(
        "--min-tokens",
        type=int,
        default=DEFAULT_MIN_TOKENS,
        help=f"Minimum scored tokens for a strong negative (default: {DEFAULT_MIN_TOKENS})",
    )
    detect.add_argument(
        "--abstention-band",
        type=float,
        default=DEFAULT_ABSTENTION_BAND,
        help=f"Relative near-threshold abstention band (default: {DEFAULT_ABSTENTION_BAND})",
    )
    detect.add_argument("--json", action="store_true", help="Emit JSON on stdout")
    detect.set_defaults(handler=_cmd_detect)

    wm = sub.add_parser("watermark", help="Generate watermarked sample text")
    wm.add_argument("prompt", help="Prompt file, or - for stdin")
    wm.add_argument(
        "-o",
        "--watermarked-output",
        default=None,
        help="Output path for the watermarked sample (default: stdout)",
    )
    wm.add_argument(
        "-o2",
        "--unwatermarked-output",
        default=None,
        help="Also write an unwatermarked sample to this path",
    )
    wm.add_argument("--max-new-tokens", type=int, default=200)
    wm.add_argument("--min-length", type=int, default=0)
    wm.add_argument("--seed", type=int, default=None, help="Optional RNG seed")
    _add_common(wm)
    wm.add_argument("--json", action="store_true", help="Emit JSON on stdout")
    wm.set_defaults(handler=_cmd_watermark)

    args = p.parse_args()
    _apply_rlimit_as(args.rlimit_as)

    raw_upstream = args.upstream_dir or os.environ.get("MARKLLM_DIR")
    upstream = resolve_upstream(str(raw_upstream) if raw_upstream else None)
    if upstream is None:
        eprint(
            "MarkLLM not configured: set MARKLLM_DIR or pass --upstream-dir",
        )
        return 3

    if not (upstream / "watermark").is_dir():
        eprint(f"MarkLLM checkout incomplete (no watermark/ dir): {upstream}")
        return 3

    alg = SCHEMES[args.scheme]
    # Contract version: the checkout commit this run detects against. Used in
    # reports so results stay comparable across upgrades.
    args._implementation_commit = _git_commit(upstream)
    return args.handler(args, upstream, alg)


if __name__ == "__main__":
    raise SystemExit(main())
