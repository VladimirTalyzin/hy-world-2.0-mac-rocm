#!/usr/bin/env bash
# Environment defaults for running HY-World 2.0 on ROCm.
# Source this before launching any of the pipelines:  . scripts/rocm_env.sh
#
# Everything here is also applied programmatically by hyworld2/compat/backend.py
# (via os.environ.setdefault), so sourcing this file is optional — it exists to
# make the settings visible and overridable from the shell.

# Unlock AOTriton flash / mem-efficient SDPA on architectures ROCm still marks
# experimental (RDNA3 / RDNA3.5: gfx1100, gfx1101, gfx1151, ...). Without it
# scaled_dot_product_attention silently drops to the MATH backend: ~13x slower
# and O(N^2) in memory.
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=${TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL:-1}

# Reduce caching-allocator fragmentation. Note this is a no-op on ROCm for
# Windows ("expandable_segments not supported on this platform"); it is kept
# for Linux ROCm.
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}

# MIOpen compiles and caches convolution kernels on first use. The DPT heads in
# WorldMirror use many distinct conv shapes, so the *first* run of a new
# resolution pays several minutes of kernel compilation; later runs hit the
# cache. Keep the cache on fast local storage and let it persist.
export MIOPEN_USER_DB_PATH=${MIOPEN_USER_DB_PATH:-"$HOME/.cache/miopen"}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_CUSTOM_CACHE_DIR:-"$MIOPEN_USER_DB_PATH"}
mkdir -p "$MIOPEN_USER_DB_PATH" 2>/dev/null || true

# Optional: skip MIOpen's exhaustive kernel search. Cuts first-run latency a
# lot at a small steady-state throughput cost. Uncomment for interactive use.
# export MIOPEN_FIND_MODE=FAST

# Some APUs (Strix Halo / gfx1151) report an unsupported arch string to
# libraries that gate on a fixed list. Uncomment if a dependency refuses to
# initialise; it makes them treat the iGPU as the closest supported RDNA3 part.
# export HSA_OVERRIDE_GFX_VERSION=11.0.0
