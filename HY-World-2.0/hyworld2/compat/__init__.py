"""Cross-backend compatibility layer for the ROCm / MPS ports of HY-World 2.0."""

from .backend import (  # noqa: F401
    autocast,
    can_compile,
    configure_mps_env,
    configure_rocm_env,
    maybe_compile,
    describe,
    device_type,
    empty_cache,
    get_device,
    is_mps,
    is_rocm,
    max_memory_allocated,
    preferred_dtype,
    reset_peak_memory_stats,
    supports_bf16,
    synchronize,
)
from .attention import attention, attention_backend_name  # noqa: F401
from .distributed import (  # noqa: F401
    distributed_available,
    install_single_process_shims,
    rank,
    single_process,
    world_size,
)
from .conv3d_fold import fold_causal_conv3d, unfold_causal_conv3d  # noqa: F401
from .conv3d_unroll import (  # noqa: F401
    conv3d_as_conv2d,
    restore_wan_conv3d,
    unroll_wan_conv3d,
)
from .streaming_load import (  # noqa: F401
    checkpoint_keys,
    load_safetensors_into_model,
    remaining_meta_tensors,
)

# The worldgen stages call torch.distributed collectives unconditionally. On a
# build without a c10d backend those attributes do not exist at all, so fill in
# one-rank stand-ins before any of that code runs. No-op where a real backend
# is present. See distributed.py for what is and is not shimmed.
install_single_process_shims()
