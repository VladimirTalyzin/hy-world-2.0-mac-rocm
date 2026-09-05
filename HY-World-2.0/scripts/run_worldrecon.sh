#!/usr/bin/env bash
# Feed-forward 3D reconstruction (WorldMirror 2.0) on ROCm.
#
#   ./scripts/run_worldrecon.sh <image-dir-or-video> [extra pipeline args...]
#
# Example:
#   ./scripts/run_worldrecon.sh examples/worldrecon/realistic/Desk
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "$HERE/scripts/rocm_env.sh"

INPUT="${1:?usage: run_worldrecon.sh <image-dir-or-video> [args...]}"
shift || true

PYTHON="${PYTHON:-$HERE/../venv/Scripts/python.exe}"
[ -x "$PYTHON" ] || PYTHON="${PYTHON:-python}"
WEIGHTS="${HYWORLD_WEIGHTS:-$HERE/../weights}"

exec "$PYTHON" -u -m hyworld2.worldrecon.pipeline \
    --input_path "$INPUT" \
    --pretrained_model_name_or_path "$WEIGHTS" \
    --output_path "${HYWORLD_OUTPUT:-$HERE/../outputs}" \
    --enable_bf16 \
    "$@"
