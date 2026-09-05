#!/usr/bin/env bash
# =============================================================================
# Launch the HY-World 2.0 web UI (macOS/MPS, Linux/ROCm, or CUDA/CPU).
#
# On Linux this sources HY-World-2.0/scripts/rocm_env.sh -- the AOTriton SDPA
# switch and the MIOpen cache location, both explained there. On macOS it sets
# the MPS fallback so ops without a Metal kernel drop to CPU instead of raising,
# and lifts the allocator's soft high-watermark cap that aborts long runs.
#
#   ./launch.sh [--port N] [--host ADDR] [--device cpu|cuda|mps] [--no-browser]
# =============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${PROJECT_DIR}/HY-World-2.0"

PORT="${GRADIO_PORT:-7860}"
HOST="127.0.0.1"
EXTRA=()

usage() {
    cat <<EOF
Usage: $0 [--port N] [--host ADDR] [--device DEV] [--no-browser]

  --port         port for the UI (default 7860)
  --host         interface to bind (default 127.0.0.1; 0.0.0.0 for the LAN)
  --device       force a torch device (HYWORLD_DEVICE), e.g. cpu -- for debugging
  --no-browser   do not open the browser automatically
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)       PORT="$2"; shift 2 ;;
        --host)       HOST="$2"; shift 2 ;;
        --device)     export HYWORLD_DEVICE="$2"; shift 2 ;;
        --no-browser) EXTRA+=("--no-browser"); shift ;;
        -h|--help)    usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

# Prefer the project venv; fall back to whatever python is on PATH.
if [[ -x "${PROJECT_DIR}/venv/bin/python" ]]; then
    PYTHON="${PROJECT_DIR}/venv/bin/python"
elif [[ -x "${PROJECT_DIR}/venv/Scripts/python.exe" ]]; then
    PYTHON="${PROJECT_DIR}/venv/Scripts/python.exe"
else
    echo "[WARN] venv not found under ${PROJECT_DIR}/venv -- using the current interpreter" >&2
    PYTHON="python"
fi

export TOKENIZERS_PARALLELISM=false

if [[ "$(uname)" == "Darwin" ]]; then
    export PYTORCH_ENABLE_MPS_FALLBACK=1
    export PYTORCH_MPS_HIGH_WATERMARK_RATIO="${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-0.0}"
else
    # shellcheck disable=SC1091
    . "${REPO_DIR}/scripts/rocm_env.sh"
fi

echo "=============================================="
echo "  HY-World 2.0 — Web UI"
echo "  Project: ${PROJECT_DIR}"
echo "  URL:     http://${HOST}:${PORT}"
[[ -n "${HYWORLD_DEVICE:-}" ]] && echo "  Device:  ${HYWORLD_DEVICE} (forced)"
echo "=============================================="

cd "${PROJECT_DIR}"
exec "${PYTHON}" -u gradio_app.py --host "${HOST}" --port "${PORT}" "${EXTRA[@]}"
