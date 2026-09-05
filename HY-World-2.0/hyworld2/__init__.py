# Importing the compat layer first configures the ROCm/MPS runtime knobs
# (notably TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL) before any model code
# dispatches attention. See hyworld2/compat/backend.py.
from . import compat  # noqa: F401
