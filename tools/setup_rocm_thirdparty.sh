#!/usr/bin/env bash
# Fetch the header-only libraries the gsplat HIP build needs and that are
# neither shipped by the `rocm-sdk` wheels nor vendored in the HY-World-2.0 tree.
#
#   rocThrust  torch/headeronly/util/complex.h does `#include <thrust/complex.h>`
#              whenever __HIPCC__ is defined, so every HIP extension built
#              against PyTorch needs it. rocm-sdk-devel installs an empty
#              include/ tree (even after `rocm-sdk init`).
#   rocPRIM    rocThrust / hipCUB dependency.
#   hipCUB     gsplat's packed-projection and tile-intersection kernels include
#              <hipcub/hipcub.hpp> (hipified from <cub/cub.cuh>).
#   glm        gsplat vendors glm as a git submodule upstream; HY-World-2.0
#              ships only the empty directory, so the build fails with
#              "glm/gtc/type_ptr.hpp file not found" on CUDA as well.
#              It is placed OUTSIDE the gsplat tree on purpose: torch's hipify
#              mirrors gsplat/cuda -> gsplat/hip and copies only source
#              extensions it knows, dropping glm's .inl files and breaking it.
#
# rocThrust/rocPRIM/hipCUB normally get a version header from CMake's
# configure_file(); we generate the same file here.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-$HERE/third_party_rocm}"
TAG="${ROCM_LIB_TAG:-rocm-7.0.0}"

mkdir -p "$DEST"
clone() {  # clone <url> <ref> <dir>
    [ -d "$3/.git" ] || git clone --depth 1 --branch "$2" "$1" "$3"
}
clone https://github.com/ROCm/rocThrust.git "$TAG" "$DEST/rocThrust"
clone https://github.com/ROCm/rocPRIM.git   "$TAG" "$DEST/rocPRIM"
clone https://github.com/ROCm/hipCUB.git    "$TAG" "$DEST/hipCUB"
clone https://github.com/g-truc/glm.git     "${GLM_TAG:-1.0.1}" "$DEST/glm"

# configure_file() equivalents -------------------------------------------------
gen_version() {  # gen_version <template> <output> <cmake-var-prefix> <maj> <min> <patch>
    [ -f "$1" ] || { echo "skip (no template): $1"; return; }
    local number=$(( $4 * 100000 + $5 * 100 + $6 ))
    sed -e "s/@$3_VERSION_NUMBER@/${number}/g" \
        -e "s/@$3_VERSION_MAJOR@/$4/g" \
        -e "s/@$3_VERSION_MINOR@/$5/g" \
        -e "s/@$3_VERSION_PATCH@/$6/g" \
        "$1" > "$2"
    echo "generated $2"
}

# Version numbers come from each project's CMakeLists.txt (rocm_setup_version).
gen_version "$DEST/rocThrust/thrust/rocthrust_version.hpp.in" \
            "$DEST/rocThrust/thrust/rocthrust_version.hpp" rocthrust 4 0 0
gen_version "$DEST/rocPRIM/rocprim/include/rocprim/rocprim_version.hpp.in" \
            "$DEST/rocPRIM/rocprim/include/rocprim/rocprim_version.hpp" rocprim 4 0 0
gen_version "$DEST/hipCUB/hipcub/include/hipcub/hipcub_version.hpp.in" \
            "$DEST/hipCUB/hipcub/include/hipcub/hipcub_version.hpp" hipcub 4 0 0

echo "third-party headers ready in $DEST"
