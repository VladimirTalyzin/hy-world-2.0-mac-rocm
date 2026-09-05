#!/usr/bin/env bash
# =============================================================================
# HY-World 2.0 (ROCm / MPS port) — installer for Linux with AMD ROCm and for
# macOS with Apple Silicon.
#
#   ./scripts/install.sh [--install-torch] [--skip-weights] [--recon-only] [--no-examples]
#
# Linux/ROCm:  a working ROCm PyTorch is reused if one is importable. If not,
#              --install-torch picks the wheel index from the GPU architecture
#              (rocminfo): gfx1151 / gfx12xx go to AMD's TheRock nightlies,
#              which ship native kernels for those parts; everything else to
#              pytorch.org's ROCm index. pytorch.org wheels are built against
#              ROCm 6.x and segfault on a 7.x runtime, so the choice matters.
# macOS:       the default PyPI torch wheel includes MPS. Note the port is not
#              yet verified on a Mac (see MPS_PORT.md).
# =============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="${PROJECT_DIR}/HY-World-2.0"
VENV_DIR="${PROJECT_DIR}/venv"

INSTALL_TORCH=0; SKIP_WEIGHTS=0; RECON_ONLY=0; NO_EXAMPLES=0
for a in "$@"; do
    case "$a" in
        --install-torch) INSTALL_TORCH=1 ;;
        --skip-weights)  SKIP_WEIGHTS=1 ;;
        --recon-only)    RECON_ONLY=1 ;;
        --no-examples)   NO_EXAMPLES=1 ;;
        -h|--help) sed -n 2,16p "$0"; exit 0 ;;
        *) echo "unknown option: $a" >&2; exit 1 ;;
    esac
done

info() { echo -e "\033[36m[INFO]\033[0m  $*"; }
ok()   { echo -e "\033[32m[ OK ]\033[0m  $*"; }
warn() { echo -e "\033[33m[WARN]\033[0m  $*"; }
err()  { echo -e "\033[31m[ERR ]\033[0m  $*" >&2; }

OS="$(uname)"
echo "============================================================"
echo "  HY-World 2.0 - install ($OS)"
echo "  Project: $PROJECT_DIR"
echo "============================================================"

# ---- Python ---------------------------------------------------------------
SYS_PY="${PYTHON:-python3}"
command -v "$SYS_PY" >/dev/null || { err "python3 not found (3.11 or 3.12 required)"; exit 1; }
if [[ ! -d "$VENV_DIR" ]]; then
    info "Creating venv at $VENV_DIR"
    "$SYS_PY" -m venv "$VENV_DIR"
fi
PY="$VENV_DIR/bin/python"
ok "Using $PY ($("$PY" --version))"
"$PY" -m pip install --upgrade pip setuptools wheel >/dev/null

# ---- torch ----------------------------------------------------------------
probe() {
    "$PY" - <<'EOF'
import json
try:
    import torch
    d = {"ok": True, "version": torch.__version__, "hip": torch.version.hip,
         "cuda": torch.cuda.is_available(), "mps": hasattr(torch.backends, "mps") and torch.backends.mps.is_available()}
    if d["cuda"]:
        p = torch.cuda.get_device_properties(0); d["name"] = p.name; d["arch"] = getattr(p, "gcnArchName", None)
    print(json.dumps(d))
except Exception as e:
    print(json.dumps({"ok": False, "error": str(e)}))
EOF
}
INFO="$(probe)"
have_gpu() { echo "$INFO" | "$PY" -c 'import json,sys; d=json.load(sys.stdin); sys.exit(0 if d.get("ok") and (d.get("cuda") or d.get("mps")) else 1)'; }

if [[ $INSTALL_TORCH -eq 1 ]] || ! have_gpu; then
    if [[ "$OS" == "Darwin" ]]; then
        info "Installing torch (the default wheel includes MPS)"
        "$PY" -m pip install torch torchvision torchaudio
    else
        ARCH="$(rocminfo 2>/dev/null | grep -o 'gfx[0-9a-f]*' | head -1 || true)"
        info "GPU architecture: ${ARCH:-unknown}"
        case "$ARCH" in
            gfx1151|gfx1200|gfx1201)
                INDEX="https://rocm.nightlies.amd.com/v2/${ARCH}/"
                info "Installing torch from AMD's nightly index for $ARCH"
                "$PY" -m pip install --index-url "$INDEX" "rocm[libraries,devel]" torch torchvision torchaudio ;;
            *)
                info "Installing torch from pytorch.org (ROCm 6.4 index)"
                "$PY" -m pip install --index-url https://download.pytorch.org/whl/rocm6.4 torch torchvision torchaudio ;;
        esac
    fi
    INFO="$(probe)"
fi
if ! have_gpu; then
    err "torch sees no GPU: $INFO"
    [[ "$OS" == "Darwin" ]] || echo "  Check the ROCm driver, or export HSA_OVERRIDE_GFX_VERSION for unlisted parts."
    exit 1
fi
ok "torch: $INFO"

# ---- dependencies ---------------------------------------------------------
info "Installing the port's dependencies (requirements-rocm.txt; torch is left alone)"
"$PY" -m pip install -r "$REPO_DIR/requirements-rocm.txt"
ok "dependencies installed"

# ---- weights and examples -------------------------------------------------
cd "$PROJECT_DIR"
if [[ $SKIP_WEIGHTS -eq 0 ]]; then
    ARGS=()
    [[ $RECON_ONLY -eq 1 ]] && ARGS+=(--recon-only)
    [[ $NO_EXAMPLES -eq 0 ]] && ARGS+=(--examples)
    info "Downloading weights: download_weights.py ${ARGS[*]:-}"
    "$PY" download_weights.py "${ARGS[@]}"
elif [[ $NO_EXAMPLES -eq 0 ]]; then
    "$PY" download_weights.py --recon-only --examples
fi

# ---- verify ---------------------------------------------------------------
"$PY" scripts/doctor.py || true

echo
echo "============================================================"
echo "  Done. Start the UI with:  ./launch.sh"
echo "  Then open http://localhost:7860"
[[ "$OS" == "Darwin" ]] || echo "  Optional, fly-through video on ROCm: see ROCM_PORT.md (gsplat HIP build)"
echo "============================================================"
