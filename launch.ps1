<#
.SYNOPSIS
    Launch the HY-World 2.0 web UI on Windows (ROCm, CUDA or CPU).

.DESCRIPTION
    Sets the same environment as HY-World-2.0/scripts/rocm_env.sh and starts
    gradio_app.py from the project venv. The two settings that matter on ROCm:

      TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
          Unlocks flash / mem-efficient SDPA on RDNA3(.5) parts such as
          gfx1151. Without it attention silently runs on the MATH backend --
          ~13x slower and quadratic in memory. hyworld2/compat sets it too, so
          this is belt and braces for anything that imports torch first.

      MIOPEN_USER_DB_PATH / MIOPEN_CUSTOM_CACHE_DIR
          MIOpen compiles convolution kernels on first use; keeping the cache in
          a stable place means only the first run of a new resolution pays for
          it.

    HSA_OVERRIDE_GFX_VERSION is deliberately NOT set: it is only for torch
    builds that lack kernels for your GPU, and forcing it when native kernels
    exist makes things slower or wrong.

.PARAMETER Port
    Port for the UI (default 7860).

.PARAMETER ListenHost
    Interface to bind (default 127.0.0.1; use 0.0.0.0 to reach it from the LAN).

.PARAMETER Device
    Force a torch device, e.g. "cpu" -- maps to HYWORLD_DEVICE. Debugging aid.

.PARAMETER NoBrowser
    Do not open the browser automatically.
#>
param(
    [int]$Port = 7860,
    [string]$ListenHost = "127.0.0.1",
    [string]$Device,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path

$Python = "python"
$VenvPython = Join-Path $ProjectDir "venv\Scripts\python.exe"
if (Test-Path $VenvPython) { $Python = $VenvPython }

# ROCm knobs (harmless on CUDA / CPU). setdefault semantics: the user's shell wins.
if (-not $env:TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL) { $env:TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL = "1" }
if (-not $env:PYTORCH_ALLOC_CONF) { $env:PYTORCH_ALLOC_CONF = "expandable_segments:True" }   # no-op on Windows ROCm, kept for Linux
$miopenCache = Join-Path $HOME ".cache\miopen"
New-Item -ItemType Directory -Force $miopenCache | Out-Null
if (-not $env:MIOPEN_USER_DB_PATH)     { $env:MIOPEN_USER_DB_PATH = $miopenCache }
if (-not $env:MIOPEN_CUSTOM_CACHE_DIR) { $env:MIOPEN_CUSTOM_CACHE_DIR = $miopenCache }
$env:TOKENIZERS_PARALLELISM = "false"

if ($Device) { $env:HYWORLD_DEVICE = $Device }

$args_ = @("-u", "gradio_app.py", "--host", $ListenHost, "--port", "$Port")
if ($NoBrowser) { $args_ += "--no-browser" }

Write-Host "=============================================="
Write-Host "  HY-World 2.0 - Web UI"
Write-Host "  Project: $ProjectDir"
Write-Host "  URL:     http://${ListenHost}:$Port"
if ($Device) { Write-Host "  Device:  $Device (forced)" }
Write-Host "=============================================="

Push-Location $ProjectDir
try {
    & $Python @args_
} finally {
    Pop-Location
}
