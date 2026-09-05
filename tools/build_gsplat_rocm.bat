@echo off
setlocal enableextensions
REM Build the bundled gsplat (mask-gaussian variant) as a HIP extension for ROCm.
REM
REM torch.utils.cpp_extension derives ROCM_HOME from `where hipcc`, which with
REM the `rocm-sdk` wheels resolves to <python-prefix>\Scripts\hipcc.exe. Its
REM parent-of-parent is the Python prefix, so torch then looks for a
REM non-existent <prefix>\bin\hipcc.exe. The real ROCm tree (bin + include +
REM lib) is the _rocm_sdk_core package. Note `rocm-sdk path --root` reports
REM _rocm_sdk_devel, which has an empty include/ and is NOT usable here.
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1

set "PY=%~dp0..\venv\Scripts\python.exe"
if not defined ROCM_HOME (
  "%PY%" -c "import _rocm_sdk_core,os;print(os.path.dirname(_rocm_sdk_core.__file__))" > "%TEMP%\_rocm_home.txt"
  set /p ROCM_HOME=<"%TEMP%\_rocm_home.txt"
)
if not defined ROCM_HOME (
  echo ERROR: could not locate the ROCm SDK core package.
  exit /b 1
)
echo ROCM_HOME=%ROCM_HOME%
set "ROCM_PATH=%ROCM_HOME%"
set "HIP_PATH=%ROCM_HOME%"
set "PATH=%ROCM_HOME%\bin;%PATH%"

set PYTORCH_ROCM_ARCH=gfx1151
set MAX_JOBS=16
set DISTUTILS_USE_SDK=1

cd /d "%~dp0..\HY-World-2.0\hyworld2\worldgen\third_party\gsplat_maskgaussian"
"%PY%" -m pip install -v -e . --no-build-isolation --no-deps
exit /b %errorlevel%
