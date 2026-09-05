#!/usr/bin/env bash
# 360-degree panorama generation (HY-Pano 2.0, Qwen-Image-Edit backend) on ROCm.
#
#   ./scripts/run_panogen.sh <input-image> [extra args...]
#
# The base model (Qwen/Qwen-Image-Edit-2509, ~54 GB in bf16) and the HY-Pano
# LoRA are expected under $HYWORLD_WEIGHTS; both are downloaded from the Hub on
# first use otherwise.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "$HERE/scripts/rocm_env.sh"

IMAGE="${1:?usage: run_panogen.sh <input-image> [args...]}"
shift || true

PYTHON="${PYTHON:-$HERE/../venv/Scripts/python.exe}"
[ -x "$PYTHON" ] || PYTHON="python"
WEIGHTS="${HYWORLD_WEIGHTS:-$HERE/../weights}"

cd "$HERE/hyworld2/panogen"
exec "$PYTHON" -u pipeline_with_qwen_image.py \
    --image "$IMAGE" \
    --pretrained-model-name-or-path "$WEIGHTS/Qwen-Image-Edit-2509" \
    --lora-path "$WEIGHTS" \
    --lora-subfolder HY-Pano-2.0 \
    "$@"
