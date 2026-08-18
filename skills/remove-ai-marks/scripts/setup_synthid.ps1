<#
.SYNOPSIS
  Windows port of setup_synthid.sh.

.DESCRIPTION
  Bootstraps the external reverse-SynthID checkout for the optional pixel
  scorer. This is SCORING only, not removal.

  The upstream project (https://github.com/aloshdenny/reverse-SynthID) is
  under a non-commercial Research License and is NOT bundled in this
  repository. This script clones it locally and installs only the
  dependencies that score_synthid.py needs.

  Difference from the .sh version: the Windows venv lives in .venv\Scripts\
  instead of .venv/bin/.

.PARAMETER Dir
  Checkout directory (default: $env:REVERSE_SYNTHID_DIR or ~\reverse-SynthID)

.PARAMETER Ref
  Commit to use (default: the pinned SHA; do not point at a moving branch)

.PARAMETER Full
  Install the full upstream requirements.txt (adds torch/diffusers for the
  VAE bypass) instead of only the scorer dependencies.

.PARAMETER Python
  Interpreter used to create the venv (default: python)
#>
[CmdletBinding()]
param(
    [string]$Dir,
    [string]$Ref = 'b11083676fd3ee3ff97ce9d03c0e409e46905902',
    [switch]$Full,
    [string]$Python = 'python'
)

$ErrorActionPreference = 'Stop'
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

function Invoke-Checked {
    param([string]$What, [scriptblock]$Block)
    & $Block
    if ($LASTEXITCODE -ne 0) { throw "$What failed (exit code $LASTEXITCODE)" }
}

if (-not $Dir) {
    if ($env:REVERSE_SYNTHID_DIR) { $Dir = $env:REVERSE_SYNTHID_DIR }
    else { $Dir = Join-Path $HOME 'reverse-SynthID' }
}
$parent = Split-Path -Parent $Dir
if ($parent -and -not (Test-Path $parent)) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
if (-not (Test-Path $Dir)) { New-Item -ItemType Directory -Force -Path $Dir | Out-Null }
$Dir = (Resolve-Path $Dir).Path

if (-not (Test-Path (Join-Path $Dir '.git'))) {
    Write-Host "Cloning reverse-SynthID into $Dir (pinned ref: $Ref)"
    Invoke-Checked 'git clone' { git clone --depth 1 --filter=blob:none --sparse https://github.com/aloshdenny/reverse-SynthID.git $Dir }
    Invoke-Checked 'git fetch' { git -C $Dir fetch --depth 1 origin $Ref }
    Invoke-Checked 'git checkout' { git -C $Dir checkout --detach $Ref }
    Invoke-Checked 'sparse-checkout' {
        git -C $Dir sparse-checkout set --no-cone '/src/' '/artifacts/spectral_codebook_v4.npz' '/requirements.txt' '/LICENSE' '/README.md'
    }
    $head = (git -C $Dir rev-parse HEAD).Trim()
    if ($head -ne $Ref) { throw "error: expected pinned ref $Ref, got $head" }
} else {
    Write-Host "Using existing checkout: $Dir"
    $head = ''
    try { $head = "$(git -C $Dir rev-parse HEAD)".Trim() } catch { $head = '' }
    if ($head -ne $Ref) {
        Write-Host "existing checkout not at pinned ref $Ref (HEAD: $head); re-pinning"
        Invoke-Checked 'git fetch' { git -C $Dir fetch --depth 1 origin $Ref }
        Invoke-Checked 'git checkout' { git -C $Dir checkout --detach $Ref }
        Invoke-Checked 'sparse-checkout' {
            git -C $Dir sparse-checkout set --no-cone '/src/' '/artifacts/spectral_codebook_v4.npz' '/requirements.txt' '/LICENSE' '/README.md'
        }
        $head = (git -C $Dir rev-parse HEAD).Trim()
        if ($head -ne $Ref) { throw "error: expected pinned ref $Ref, got $head" }
    }
}

$venvPython = Join-Path $Dir '.venv\Scripts\python.exe'
if (-not (Test-Path $venvPython)) {
    Write-Host "Creating venv at $Dir\.venv"
    Invoke-Checked 'create venv' { & $Python -m venv (Join-Path $Dir '.venv') }
}

Write-Host 'Installing Python dependencies'
# Pinned pip (an unpinned --upgrade pip was a supply-chain drift point).
Invoke-Checked 'pip install pip' { & $venvPython -m pip install --upgrade 'pip==26.2.1' }

if ($Full) {
    Write-Host 'Installing the full upstream requirements.txt (includes torch/diffusers)'
    Invoke-Checked 'pip install upstream' { & $venvPython -m pip install -r (Join-Path $Dir 'requirements.txt') }
} else {
    Write-Host 'Installing scorer-only dependencies'
    Invoke-Checked 'pip install scorer' { & $venvPython -m pip install -r (Join-Path $ScriptDir 'requirements-synthid-scorer.txt') }
}

$codebook = Join-Path $Dir 'artifacts\spectral_codebook_v4.npz'
if (-not (Test-Path $codebook)) {
    Write-Warning "codebook not found at $codebook"
    Write-Warning "run: git -C '$Dir' sparse-checkout add '/artifacts/spectral_codebook_v4.npz'"
}

Write-Host ''
Write-Host 'Done. Score an image with:'
Write-Host ''
Write-Host "  `$env:REVERSE_SYNTHID_DIR = '$Dir'"
Write-Host "  python '$ScriptDir\score_synthid.py' IMAGE"
