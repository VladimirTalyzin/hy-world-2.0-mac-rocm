@echo off
rem ============================================================================
rem  HY-World 2.0 - one-click launcher for the web UI.
rem
rem  Double-click this file. It starts the local server and opens the browser;
rem  closing this window stops the server.
rem
rem  The environment set below is the same as launch.ps1 / scripts/rocm_env.sh:
rem
rem    TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL  unlocks flash / mem-efficient
rem        SDPA on RDNA3(.5) parts such as gfx1151. Without it attention falls
rem        back to the MATH backend - roughly 13x slower and quadratic in memory.
rem
rem    MIOPEN_USER_DB_PATH / MIOPEN_CUSTOM_CACHE_DIR  keep MIOpen's compiled
rem        convolution kernels between runs, so only the first run at a new
rem        resolution pays the compile.
rem
rem  Optional arguments are passed straight through to gradio_app.py, e.g.
rem      "HY-World 2.0.cmd" --port 7861
rem      "HY-World 2.0.cmd" --host 0.0.0.0        (reachable from the LAN)
rem ============================================================================
setlocal
cd /d "%~dp0"
title HY-World 2.0 - Web UI

set "PYTHON=%~dp0venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

if not defined TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL set "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1"
if not defined PYTORCH_ALLOC_CONF set "PYTORCH_ALLOC_CONF=expandable_segments:True"
if not defined MIOPEN_USER_DB_PATH set "MIOPEN_USER_DB_PATH=%USERPROFILE%\.cache\miopen"
if not defined MIOPEN_CUSTOM_CACHE_DIR set "MIOPEN_CUSTOM_CACHE_DIR=%MIOPEN_USER_DB_PATH%"
if not exist "%MIOPEN_USER_DB_PATH%" mkdir "%MIOPEN_USER_DB_PATH%" 2>nul
set "TOKENIZERS_PARALLELISM=false"

echo ==============================================
echo   HY-World 2.0 - Web UI
echo   Project: %~dp0
echo   The browser opens by itself in a few seconds.
echo   Close this window to stop the server.
echo ==============================================
echo.

rem -u keeps stdout unbuffered so the log below appears as it happens.
"%PYTHON%" -u gradio_app.py %*

rem Only reached if the server exited; without the pause a crash would close
rem the window before the traceback could be read.
echo.
echo Server stopped (exit code %ERRORLEVEL%).
pause
