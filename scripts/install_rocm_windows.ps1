<#
.SYNOPSIS
    Install HY-World 2.0 (ROCm port) on Windows with an AMD GPU.

.DESCRIPTION
    Windows-native ROCm is a public preview, so this script is deliberately
    conservative: if a working ROCm PyTorch is already importable it reuses it
    and never reinstalls torch over the top. The wheel that works for a given
    GPU is architecture-specific -- pytorch.org ROCm wheels are built against
    ROCm 6.x and segfault on a 7.x runtime, and gfx1151 (Radeon 8060S / Ryzen
    AI Max+) needs the TheRock nightly index, which ships native kernels for it.

    Steps: venv -> (torch check) -> requirements-rocm.txt -> weights ->
    optional bundled examples -> verification. gsplat (needed only for the
    fly-through video) is a separate, optional build: tools\build_gsplat_rocm.bat.

.PARAMETER InstallTorch
    Install torch from the ROCm nightly index for gfx1151 even if one is present.

.PARAMETER SkipWeights
    Skip the Hugging Face downloads.

.PARAMETER ReconOnly
    Download only WorldMirror 2.0 (~4.8 GB), not the panorama models (~55 GB).

.PARAMETER NoExamples
    Do not restore upstream's bundled worldrecon examples (~110 MB, needs git).

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\install_rocm_windows.ps1
#>
param(
    [switch]$InstallTorch,
    [switch]$SkipWeights,
    [switch]$ReconOnly,
    [switch]$NoExamples
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$RepoDir = Join-Path $ProjectDir "HY-World-2.0"
$VenvDir = Join-Path $ProjectDir "venv"

function Write-Info { param($m) Write-Host "[INFO]  $m" -ForegroundColor Cyan }
function Write-Ok   { param($m) Write-Host "[ OK ]  $m" -ForegroundColor Green }
function Write-Warn2{ param($m) Write-Host "[WARN]  $m" -ForegroundColor Yellow }
function Write-Err  { param($m) Write-Host "[ERR ]  $m" -ForegroundColor Red }

Write-Host "============================================================"
Write-Host "  HY-World 2.0 - ROCm install (Windows)"
Write-Host "  Project: $ProjectDir"
Write-Host "============================================================"

# ---- Python ---------------------------------------------------------------
$SysPython = "python"
& $SysPython --version *> $null
if (-not $?) { Write-Err "python not found in PATH (Python 3.11 or 3.12 is required)"; exit 1 }
$ver = (& $SysPython -c "import sys; print('%d.%d' % sys.version_info[:2])")
if ([version]$ver -lt [version]"3.11" -or [version]$ver -ge [version]"3.13") {
    Write-Warn2 "Python $ver found; the port is verified on 3.12 and the ROCm wheels target 3.11-3.12."
}
if (-not (Test-Path $VenvDir)) {
    Write-Info "Creating venv at $VenvDir"
    & $SysPython -m venv $VenvDir
}
$Python = Join-Path $VenvDir "Scripts\python.exe"
Write-Ok "Using $Python ($(& $Python --version 2>&1))"
& $Python -m pip install --upgrade pip setuptools wheel | Out-Null

# ---- torch / ROCm ---------------------------------------------------------
$probe = @'
import json
try:
    import torch
    print(json.dumps({"ok": True, "version": torch.__version__, "hip": torch.version.hip,
                      "cuda": torch.cuda.is_available(),
                      "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                      "arch": getattr(torch.cuda.get_device_properties(0), "gcnArchName", None) if torch.cuda.is_available() else None}))
except Exception as e:
    print(json.dumps({"ok": False, "error": str(e)}))
'@
$info = (& $Python -c $probe) | ConvertFrom-Json
$needTorch = $InstallTorch -or (-not $info.ok) -or (-not $info.hip) -or (-not $info.cuda)
if ($needTorch) {
    Write-Info "Installing the ROCm PyTorch build from AMD's nightly index (gfx1151 / gfx1200 / gfx1201)"
    Write-Host "  Other Radeon parts: see https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/ for the"
    Write-Host "  matching wheel index, install torch by hand, then re-run this script."
    & $Python -m pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ "rocm[libraries,devel]" torch torchvision torchaudio
    if (-not $?) { Write-Err "torch install failed"; exit 1 }
    $info = (& $Python -c $probe) | ConvertFrom-Json
}
if (-not ($info.ok -and $info.cuda)) {
    Write-Err "torch does not see a HIP device: $($info | ConvertTo-Json -Compress)"
    Write-Host "  Check the Adrenalin driver version and that the wheel matches your GPU architecture."
    exit 1
}
Write-Ok "torch $($info.version), HIP $($info.hip), GPU $($info.name) ($($info.arch))"

# ---- Python dependencies --------------------------------------------------
Write-Info "Installing the port's dependencies (requirements-rocm.txt; torch is left alone)"
& $Python -m pip install -r (Join-Path $RepoDir "requirements-rocm.txt")
if (-not $?) { Write-Err "pip install failed"; exit 1 }
Write-Ok "dependencies installed"

# ---- Weights and examples -------------------------------------------------
if (-not $SkipWeights) {
    $dl = @("download_weights.py")
    if ($ReconOnly) { $dl += "--recon-only" }
    if (-not $NoExamples) { $dl += "--examples" }
    Write-Info "Downloading weights: python $($dl -join ' ')"
    Push-Location $ProjectDir
    try { & $Python @dl } finally { Pop-Location }
    if (-not $?) { Write-Err "download failed (re-run: it resumes)"; exit 1 }
} elseif (-not $NoExamples) {
    Push-Location $ProjectDir
    try { & $Python download_weights.py --recon-only --examples } finally { Pop-Location }
}

# ---- Verify ---------------------------------------------------------------
Push-Location $ProjectDir
try { & $Python scripts\doctor.py } finally { Pop-Location }

Write-Host ""
Write-Host "============================================================"
Write-Host "  Done. Start the UI with:  .\launch.ps1   (or double-click 'HY-World 2.0.cmd')"
Write-Host "  Then open http://localhost:7860"
Write-Host "  Optional, for the fly-through video: tools\build_gsplat_rocm.bat (needs MSVC Build Tools)"
Write-Host "============================================================"
