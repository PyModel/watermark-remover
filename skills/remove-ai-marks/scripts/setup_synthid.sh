#!/usr/bin/env bash
set -euo pipefail

# Bootstrap an external reverse-SynthID checkout for the optional pixel scorer.
#
# The upstream project (https://github.com/aloshdenny/reverse-SynthID) is
# licensed under a non-commercial Research License and is NOT bundled in this
# repository. This script clones it locally and installs only the dependencies
# needed by score_synthid.py.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_DIR="${REVERSE_SYNTHID_DIR:-$HOME/reverse-SynthID}"
DIR=""
REF="main"
PYTHON="${PYTHON:-python3}"
FULL=0

usage() {
  cat <<'EOF'
Usage: setup_synthid.sh [--dir PATH] [--ref REF] [--full] [--python PYTHON]

Clones (if needed) aloshdenny/reverse-SynthID, creates a venv, and installs
the Python dependencies required by score_synthid.py.

Options:
  --dir PATH     checkout directory (default: $REVERSE_SYNTHID_DIR or ~/reverse-SynthID)
  --ref REF      git ref to clone (default: main)
  --full         install upstream requirements.txt (adds torch/diffusers for VAE bypass)
  --python PY    Python interpreter used to create the venv (default: python3)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dir)
      DIR="${2:?--dir requires a value}"
      shift 2
      ;;
    --ref)
      REF="${2:?--ref requires a value}"
      shift 2
      ;;
    --full)
      FULL=1
      shift
      ;;
    --python)
      PYTHON="${2:?--python requires a value}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

DIR="${DIR:-$DEFAULT_DIR}"
mkdir -p "$(dirname "$DIR")"
if command -v realpath >/dev/null 2>&1; then
  DIR="$(realpath -m "$DIR")"
else
  DIR="$(cd "$(dirname "$DIR")" && pwd)/$(basename "$DIR")"
fi

if [[ ! -d "$DIR/.git" ]]; then
  echo "Cloning reverse-SynthID into $DIR"
  git clone --depth 1 --filter=blob:none --sparse --branch "$REF" \
    https://github.com/aloshdenny/reverse-SynthID.git "$DIR"
  git -C "$DIR" sparse-checkout set --no-cone \
    '/src/' \
    '/artifacts/spectral_codebook_v4.npz' \
    '/requirements.txt' \
    '/LICENSE' \
    '/README.md'
else
  echo "Using existing checkout: $DIR"
fi

if [[ ! -x "$DIR/.venv/bin/python" ]]; then
  echo "Creating venv at $DIR/.venv"
  "$PYTHON" -m venv "$DIR/.venv"
fi

echo "Installing Python dependencies"
"$DIR/.venv/bin/python" -m pip install --upgrade pip
if [[ "$FULL" -eq 1 ]]; then
  echo "Installing full upstream requirements.txt (includes torch/diffusers)"
  "$DIR/.venv/bin/python" -m pip install -r "$DIR/requirements.txt"
else
  echo "Installing scorer-only dependencies"
  "$DIR/.venv/bin/python" -m pip install -r "$SCRIPT_DIR/requirements-synthid-scorer.txt"
fi

codebook="$DIR/artifacts/spectral_codebook_v4.npz"
if [[ ! -f "$codebook" ]]; then
  echo "warning: codebook not found at $codebook" >&2
  echo "run: git -C '$DIR' sparse-checkout add '/artifacts/spectral_codebook_v4.npz'" >&2
fi

cat <<EOF

Done. Score an image with:

  export REVERSE_SYNTHID_DIR="$DIR"
  "$DIR/.venv/bin/python" "\$REPO/skills/remove-ai-marks/scripts/score_synthid.py" IMAGE
EOF
