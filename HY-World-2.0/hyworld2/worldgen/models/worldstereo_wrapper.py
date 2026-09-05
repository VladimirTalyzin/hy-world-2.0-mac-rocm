"""
WorldStereo unified inference class.

Bundles all sub-models (transformer, text/image encoders, VAE) and the
matching inference pipeline under a single diffusers-style interface::

    worldstereo = WorldStereo.from_pretrained(
        "/path/to/checkpoint_root",
        device=device,
    )
    output = worldstereo(**pipeline_inputs)

Hugging Face format expects ``config.json`` plus ``model.safetensors``
in the same directory.

The config must include a ``model_type`` field with one of the
supported values:

* ``worldstereo-camera``      – keyframe + camera control
* ``worldstereo-memory``      – keyframe + camera control + GGM + SSM
* ``worldstereo-memory-dmd``  – DMD (distribution matching distillation) mode
"""

from __future__ import annotations

import gc
import json
import os
import types
from typing import Any

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["DIFFUSERS_VERBOSITY"] = "error"

import torch
import torch.distributed as dist
from diffusers.models import AutoencoderKLWan
from diffusers.schedulers import UniPCMultistepScheduler
from omegaconf import OmegaConf
from safetensors.torch import load_file as load_safetensors
# FSDP is a hard import upstream, but torch.distributed.fsdp pulls in
# torch._C._distributed_c10d, which builds without a distributed backend (ROCm
# on Windows) do not have. Import it softly: single-GPU inference never calls
# either symbol, and asking for --fsdp without a backend now fails with a
# sentence instead of a ModuleNotFoundError three frames deep.
try:
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    _FSDP_IMPORT_ERROR: Exception | None = None
except Exception as _exc:  # pragma: no cover - platform dependent
    MixedPrecisionPolicy = fully_shard = None
    _FSDP_IMPORT_ERROR = _exc


def _require_fsdp() -> None:
    if _FSDP_IMPORT_ERROR is not None:
        raise RuntimeError(
            "FSDP was requested, but torch.distributed.fsdp is unavailable in "
            f"this PyTorch build ({_FSDP_IMPORT_ERROR}). Model sharding needs a "
            "distributed backend; run single-GPU without --fsdp."
        )

from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, UMT5EncoderModel

from .attention import WanAttnProcessorSP
from .dmd_scheduler import FlowGeneratorScheduler
from .pipelines.pipeline_dmd_keyframe import RefKFDMDGeneratorPipeline
from .pipelines.pipeline_pcd_keyframe import KFPCDControllerPipeline
from .pipelines.pipeline_ref_keyframe import KFPCDControllerRefPipeline
from .worldstereo import WorldStereoModel, WorldStereoRefSModel
try:
    from ...compat import (
        checkpoint_keys as _checkpoint_keys,
        is_mps as _is_mps,
        is_rocm as _is_rocm,
        load_safetensors_into_model as _stream_load,
        maybe_compile as _maybe_compile,
        remaining_meta_tensors as _remaining_meta,
        unroll_wan_conv3d as _unroll_wan_conv3d,
    )
except ImportError:  # run as a script from inside hyworld2/worldgen
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))))
    from hyworld2.compat import (
        checkpoint_keys as _checkpoint_keys,
        is_mps as _is_mps,
        is_rocm as _is_rocm,
        load_safetensors_into_model as _stream_load,
        maybe_compile as _maybe_compile,
        remaining_meta_tensors as _remaining_meta,
        unroll_wan_conv3d as _unroll_wan_conv3d,
    )

try:
    from ..src.general_utils import rank0_log
except ImportError:
    from src.general_utils import rank0_log

# ── suppress noisy third-party logs ───────────────────────────────────
import logging
import warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["DIFFUSERS_VERBOSITY"] = "error"

# transformers / diffusers print a wall of "Some weights were not
# initialized / unexpected keys" on every load.  We already inspect
# load_state_dict results ourselves in worldstereo_wrapper.py, so
# silence their own reporting.
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("diffusers").setLevel(logging.ERROR)
logging.getLogger("diffusers.modeling_utils").setLevel(logging.ERROR)

# huggingface_hub HTTP request logs (newer versions use httpx as the HTTP client)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub.file_download").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("filelock").setLevel(logging.ERROR)

from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

from diffusers.utils import logging as diffusers_logging
diffusers_logging.set_verbosity_error()

# torch.compile / inductor verbose output
logging.getLogger("torch._dynamo").setLevel(logging.WARNING)
logging.getLogger("torch._inductor").setLevel(logging.WARNING)

# misc deprecation / user warnings from HF internals
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=UserWarning, module="diffusers")
# ──────────────────────────────────────────────────────────────────────

SUPPORTED_MODEL_TYPES = ("worldstereo-camera", "worldstereo-memory", "worldstereo-memory-dmd")


def _get_half_dtype() -> torch.dtype:
    """Select the best half-precision dtype based on current GPU capability: bf16 > fp16 > fp32."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    elif torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7:
        return torch.float16
    else:
        return torch.float32


_TEXT_ENCODER_DTYPES = {
    "fp32": torch.float32, "float32": torch.float32,
    "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
    "fp16": torch.float16, "float16": torch.float16,
}


def _text_encoder_dtype() -> torch.dtype:
    """Precision for UMT5-XXL, which is 11 B parameters of text encoder.

    Upstream loads it in fp32 -- 21.2 GiB -- which is affordable when the model
    is sharded over eight cards and impossible next to a 32.5 GiB transformer on
    one. bf16 halves it and carries fp32's exponent range, so the overflow that
    makes fp16 unusable for T5-family encoders does not apply; diffusers' own
    Wan pipelines run this encoder in bf16 as a matter of course.

    Set ``HYWORLD_TEXT_ENCODER_DTYPE=fp32`` to get upstream's behaviour back.
    """
    override = os.environ.get("HYWORLD_TEXT_ENCODER_DTYPE", "").strip().lower()
    if override:
        if override not in _TEXT_ENCODER_DTYPES:
            raise ValueError(
                f"HYWORLD_TEXT_ENCODER_DTYPE={override!r} is not one of "
                f"{sorted(_TEXT_ENCODER_DTYPES)}"
            )
        return _TEXT_ENCODER_DTYPES[override]
    if _is_rocm() or _is_mps():
        return _get_half_dtype()
    return torch.float32


class WorldStereo:
    """Diffusers-style wrapper that owns every sub-model and its pipeline."""

    def __init__(self, pipeline: Any, cfg: Any) -> None:
        self.pipeline = pipeline
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        *,
        subfolder: str = "",
        local_files_only: bool = False,
        sp_world_size: int = 1,
        fsdp: bool = False,
        device_mesh=None,
        device: torch.device | None = None,
    ) -> "WorldStereo":
        """
        Build a WorldStereo instance from Hugging Face format
        (``config.json`` + ``model.safetensors``).

        Args:
            repo_id: Model directory or HF repo ID.
            subfolder: Subfolder within the HF repo or local directory. This is equivalent to the `model_type` (e.g., 'worldstereo-camera').
            local_files_only: If True, avoid downloading the file and return the path to the local cached file if it exists.
            sp_world_size: Sequence-Parallel degree (1 = disabled).
            fsdp: Wrap models with PyTorch FSDP.  Requires ``device_mesh``.
            device_mesh: ``DeviceMesh`` with dims ``("rep", "shard")``.
            device: Target CUDA device.
        """
        if os.path.isdir(repo_id):
            json_cfg_path = os.path.join(repo_id, subfolder, "config.json")
            safetensors_path = os.path.join(repo_id, subfolder, "model.safetensors")

            if not os.path.exists(json_cfg_path):
                raise FileNotFoundError(f"config.json not found under {json_cfg_path!r}")
            if not os.path.exists(safetensors_path):
                raise FileNotFoundError(f"model.safetensors not found at {safetensors_path!r}")
        else:
            from huggingface_hub import hf_hub_download
            json_cfg_path = hf_hub_download(
                repo_id=repo_id,
                filename="config.json",
                subfolder=subfolder if subfolder else None,
                local_files_only=local_files_only,
            )
            safetensors_path = hf_hub_download(
                repo_id=repo_id,
                filename="model.safetensors",
                subfolder=subfolder if subfolder else None,
                local_files_only=local_files_only,
            )

        cfg = OmegaConf.create(cls._load_hf_config(json_cfg_path))
        model_weights_path = safetensors_path

        model_type = subfolder
        if model_type not in SUPPORTED_MODEL_TYPES:
            raise ValueError(
                f"Unsupported model_type {model_type!r}. "
                f"Expected one of {SUPPORTED_MODEL_TYPES}."
            )

        transformer = cls._load_transformer(
            cfg,
            model_type,
            model_weights_path,
            sp_world_size=sp_world_size,
            fsdp=fsdp,
            device_mesh=device_mesh,
            device=device,
        )

        text_encoder, image_clip, vae = cls._load_aux(
            cfg, device=device, device_mesh=device_mesh, fsdp=fsdp, local_files_only=local_files_only
        )
        image_processor = CLIPImageProcessor.from_pretrained(
            cfg.base_model, do_rescale=False, subfolder="image_processor", local_files_only=local_files_only
        )
        tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, subfolder="tokenizer", local_files_only=local_files_only)

        pipeline = cls._build_pipeline(
            model_type,
            cfg,
            transformer=transformer,
            text_encoder=text_encoder,
            image_clip=image_clip,
            image_processor=image_processor,
            tokenizer=tokenizer,
            vae=vae,
            device=device,
            local_files_only=local_files_only,
        )

        rank0_log(f"WorldStereo ({model_type}) ready.")
        return cls(pipeline=pipeline, cfg=cfg)

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Forward all arguments to the underlying inference pipeline."""
        return self.pipeline(*args, **kwargs)

    def to(self, device: torch.device) -> "WorldStereo":
        self.pipeline = self.pipeline.to(device)
        return self

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_hf_config(config_json_path: str) -> dict[str, Any]:
        with open(config_json_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        required_keys = ["base_model", "controlnet_cfg"]
        missing = [k for k in required_keys if k not in cfg]
        if missing:
            raise ValueError(
                f"config.json missing required keys: {missing}. "
                "Please use the conversion script to export a valid HF package."
            )

        return cfg

    @staticmethod
    def _build_transformer_on_meta(cfg, model_cls, freeze_backbone):
        """Build the module tree with no weights at all, or return None.

        The WorldStereo checkpoint carries the complete transformer -- backbone
        included -- so the base model's 65.9 GB of shards only ever serve to
        create tensors that are immediately overwritten. Building on ``meta``
        from the architecture config alone skips both that download and the
        ~33 GB of host RAM materialising them costs.

        Returns None when that is not possible (accelerate missing, or the
        config cannot be read), so the caller falls back to upstream's path.
        """
        try:
            from accelerate import init_empty_weights
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            rank0_log(f"Lean load unavailable ({exc}); using the base checkpoint.")
            return None

        try:
            arch_path = hf_hub_download(cfg.base_model, "transformer/config.json")
            with open(arch_path, "r", encoding="utf-8") as f:
                arch = json.load(f)
        except Exception as exc:  # network, auth, or a repo layout change
            rank0_log(f"Could not read the base architecture config ({exc}); "
                      "using the base checkpoint.")
            return None

        kwargs = {k: v for k, v in arch.items() if not k.startswith("_")}
        kwargs["controlnet_cfg"] = cfg.controlnet_cfg
        kwargs["base_model"] = cfg.base_model

        # init_empty_weights leaves buffers alone by default, which matters
        # here: the RoPE frequency tables are non-persistent, appear in no
        # checkpoint, and would be stranded on meta if they were emptied too.
        with init_empty_weights():
            model = model_cls(**kwargs)
            model.build_controlnet(load_uni3c=False, freeze_backbone=freeze_backbone)
        return model

    @staticmethod
    def _load_transformer(
        cfg,
        model_type: str,
        weights_path: str,
        *,
        sp_world_size: int,
        fsdp: bool,
        device_mesh,
        device,
    ):

        half_dtype = _get_half_dtype()
        rank0_log(f"Loading transformer ({model_type})… dtype={half_dtype}")

        model_cls = (WorldStereoModel if model_type == "worldstereo-camera"
                     else WorldStereoRefSModel)

        transformer = WorldStereo._build_transformer_on_meta(
            cfg, model_cls, cfg.freeze_backbone)
        lean = transformer is not None
        if lean:
            # Only commit to the lean path if the checkpoint really does cover
            # every parameter. Its header answers that without reading weights.
            wanted = set(transformer.state_dict())
            uncovered = wanted - _checkpoint_keys(weights_path)
            if uncovered:
                rank0_log(f"Checkpoint is missing {len(uncovered)} tensor(s) "
                          f"(e.g. {sorted(uncovered)[:3]}); falling back to "
                          "initialising from the base model.")
                del transformer
                lean = False
            else:
                rank0_log(f"Built {len(wanted)} tensors on meta; streaming the "
                          "checkpoint in without a second copy.")

        if not lean:
            transformer = model_cls.from_pretrained(
                cfg.base_model,
                subfolder="transformer",
                controlnet_cfg=cfg.controlnet_cfg,
                torch_dtype=half_dtype,
            )
            rank0_log("Building ControlNet…")
            transformer.build_controlnet(load_uni3c=False, freeze_backbone=cfg.freeze_backbone)

        if sp_world_size > 1:
            transformer.sp_size = sp_world_size
            for layer in transformer.controlnet.controlnet_blocks:
                layer.self_attn.processor.sp_size = sp_world_size
            for block in transformer.blocks:
                if model_type == "worldstereo-camera":
                    block.attn1.set_processor(WanAttnProcessorSP(sp_size=sp_world_size))
                else:
                    block.attn1.processor.sp_size = sp_world_size

        rank0_log(f"Loading HF safetensors weights from {weights_path}…")
        if lean:
            # Straight to the target device: with the parameters on meta there
            # is nothing to copy from, so this never stages a second copy in
            # host RAM. FSDP shards from CPU, so it wants them there instead.
            missing_keys, unexpected_keys = _stream_load(
                transformer, weights_path,
                device="cpu" if fsdp else device,
                dtype=half_dtype,
                requires_grad=False,
            )
            stranded = _remaining_meta(transformer)
            if stranded:
                raise RuntimeError(
                    f"{len(stranded)} tensor(s) are still on the meta device "
                    f"after loading, e.g. {stranded[:5]}. They would fail at "
                    "the first forward pass."
                )
        else:
            weights = load_safetensors(weights_path, device="cpu")
            result = transformer.load_state_dict(weights, strict=False)
            missing_keys, unexpected_keys = result.missing_keys, result.unexpected_keys
            del weights

        def _summarize_keys(keys: list[str], label: str) -> None:
            if not keys:
                return
            from collections import Counter
            # Count unloaded parameters
            total_params = sum(
                transformer.state_dict()[k].numel()
                for k in keys
                if k in transformer.state_dict()
            )
            # Count occurrence frequency of each field (split by ".") across all keys, take top-2
            field_counter: Counter[str] = Counter()
            for k in keys:
                parts = k.split(".")
                # Skip pure numeric indices (e.g. blocks.0) and common prefixes/suffixes
                field_counter.update(p for p in parts if not p.isdigit())
            top_fields = [f for f, _ in field_counter.most_common(2)]
            # Filter representative keys using top-2 fields (prefer keys that contain both fields)
            repr_keys = sorted([k for k in keys if all(f in k.split(".") for f in top_fields)])
            if not repr_keys:
                repr_keys = sorted(keys)
            sample_keys = repr_keys[:3]
            rank0_log(
                f"{label}: {len(keys)} keys ({total_params / 1e6:.1f}M params), "
                f"top fields: {top_fields}. "
                f"Representative: {sample_keys}"
                + (f" … and {len(keys) - len(sample_keys)} more" if len(keys) > len(sample_keys) else "")
            )
            rank0_log(f"These are frozen backbone weights initialized by the base video model ({cfg.base_model}).")

        _summarize_keys(unexpected_keys, "Unexpected keys")
        _summarize_keys(missing_keys, "Missing keys")

        if fsdp:
            _require_fsdp()
            fsdp_kwargs = dict(
                mp_policy=MixedPrecisionPolicy(
                    param_dtype=half_dtype,
                    reduce_dtype=torch.float32,
                ),
                mesh=device_mesh["rep", "shard"],
                reshard_after_forward=True,
            )
            transformer = transformer.to(half_dtype)
            for layer in transformer.blocks:
                fully_shard(layer, **fsdp_kwargs)
            for layer in transformer.controlnet.controlnet_blocks:
                fully_shard(layer, **fsdp_kwargs)
            fully_shard(transformer, **fsdp_kwargs)
            rank0_log("FSDP wrapping done for transformer.")
        else:
            transformer = transformer.to(device=device)

        gc.collect()
        torch.cuda.empty_cache()
        return transformer.eval()

    @staticmethod
    def _load_aux(cfg, *, device, device_mesh, fsdp: bool, local_files_only: bool = False):
        import transformers as _tr
        from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling

        # ---- text encoder ----
        te_dtype = _text_encoder_dtype()
        rank0_log(f"Loading TextEncoder (UMT5)… dtype={te_dtype}")
        text_encoder = UMT5EncoderModel.from_pretrained(
            cfg.base_model, subfolder="text_encoder", torch_dtype=te_dtype, local_files_only=local_files_only
        ).eval()
        if _tr.__version__ >= "5.0.0":
            rank0_log("Patching text_encoder.encoder.embed_tokens for transformers>=5.0.0", "WARNING")
            text_encoder.encoder.embed_tokens = text_encoder.shared
        # Eager unless Inductor can really generate kernels here; see
        # hyworld2/compat/backend.py::can_compile.
        text_encoder = _maybe_compile(text_encoder)

        # ---- image encoder ----
        rank0_log("Loading ImageEncoder (CLIP)…")
        image_clip = CLIPVisionModel.from_pretrained(
            cfg.base_model, subfolder="image_encoder", torch_dtype=torch.float32, local_files_only=local_files_only
        ).eval()
        if _tr.__version__ >= "5.0.0":
            rank0_log("Patching CLIP vision forward for transformers>=5.0.0", "WARNING")

            def _clip_vision_forward(self, pixel_values=None, interpolate_pos_encoding=False, **kwargs):
                if pixel_values is None:
                    raise ValueError("pixel_values is required")
                hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
                hidden_states = self.pre_layrnorm(hidden_states)
                encoder_outputs = self.encoder(inputs_embeds=hidden_states, **kwargs)
                pooled_output = self.post_layernorm(encoder_outputs.last_hidden_state[:, 0, :])
                return BaseModelOutputWithPooling(
                    last_hidden_state=encoder_outputs.last_hidden_state,
                    pooler_output=pooled_output,
                    hidden_states=encoder_outputs.hidden_states,
                )

            def _clip_encoder_forward(self, inputs_embeds, attention_mask=None, **kwargs):
                hidden_states = inputs_embeds
                encoder_states = ()
                for layer in self.layers:
                    encoder_states = encoder_states + (hidden_states,)
                    hidden_states = layer(hidden_states, attention_mask, **kwargs)
                encoder_states = encoder_states + (hidden_states,)
                return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_states)

            image_clip.vision_model.forward = types.MethodType(_clip_vision_forward, image_clip.vision_model)
            image_clip.vision_model.encoder.forward = types.MethodType(_clip_encoder_forward, image_clip.vision_model.encoder)

        # ---- VAE ----
        vae_dtype = _get_half_dtype()
        rank0_log(f"Loading 3D-VAE… dtype={vae_dtype}")
        vae = AutoencoderKLWan.from_pretrained(
            cfg.base_model, subfolder="vae", torch_dtype=vae_dtype, local_files_only=local_files_only
        ).eval()

        # MIOpen's Conv3d kernels for this VAE's shapes are pathologically slow
        # on gfx1151: 177.6 s to encode one 21-frame clip at 480x832, measured
        # after its exhaustive solver search had already finished, against 5.5 s
        # with each causal Conv3d rewritten as a sum of Conv2d passes -- 32x, and
        # 0.36 vs 11.6 TFLOP/s of the 10.63 TFLOP a pass actually needs. The
        # rewrite is the same arithmetic, checked to fp32 rounding by
        # tools/t_conv3d_unroll.py. Opt out with HYWORLD_VAE_CONV_UNROLL=0.
        if _is_rocm() and os.environ.get("HYWORLD_VAE_CONV_UNROLL", "1") != "0":
            unrolled = _unroll_wan_conv3d(vae)
            if unrolled:
                rank0_log(f"Unrolled {unrolled} causal Conv3d layers to Conv2d "
                          "in the VAE (ROCm Conv3d workaround).")

        vae = _maybe_compile(vae)

        if fsdp:
            _require_fsdp()
            fsdp_kwargs = dict(
                mp_policy=MixedPrecisionPolicy(
                    param_dtype=torch.float32, reduce_dtype=torch.float32,
                ),
                mesh=device_mesh["rep", "shard"],
                reshard_after_forward=True,
            )
            for layer in text_encoder.encoder.block:
                fully_shard(layer, **fsdp_kwargs)
            fully_shard(text_encoder, **fsdp_kwargs)
            rank0_log("FSDP wrapping done for T5.")

            for layer in image_clip.vision_model.encoder.layers:
                fully_shard(layer, **fsdp_kwargs)
            fully_shard(image_clip, **fsdp_kwargs)
            rank0_log("FSDP wrapping done for CLIP.")

            gc.collect()
            torch.cuda.empty_cache()
        else:
            text_encoder = text_encoder.to(device=device)
            image_clip = image_clip.to(device=device)

        vae = vae.to(device=device)
        return text_encoder, image_clip, vae

    @staticmethod
    def _build_pipeline(
        model_type: str,
        cfg,
        *,
        transformer,
        text_encoder,
        image_clip,
        image_processor,
        tokenizer,
        vae,
        device,
        local_files_only: bool = False,
    ):
        common = dict(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            image_encoder=image_clip,
            image_processor=image_processor,
            transformer=transformer,
            vae=vae,
        )
        if model_type == "worldstereo-camera":
            scheduler = UniPCMultistepScheduler.from_pretrained(
                cfg.base_model, subfolder="scheduler", local_files_only=local_files_only
            )
            return KFPCDControllerPipeline(**common, scheduler=scheduler)

        if model_type == "worldstereo-memory":
            scheduler = UniPCMultistepScheduler.from_pretrained(
                cfg.base_model, subfolder="scheduler", local_files_only=local_files_only
            )
            return KFPCDControllerRefPipeline(**common, scheduler=scheduler)

        if model_type == "worldstereo-memory-dmd":
            scheduler = FlowGeneratorScheduler(
                start_timesteps=cfg.dmd_start_steps,
                num_train_timesteps=cfg.dmd_end_steps,
                shift=cfg.gen_shift,
                use_timestep_transform=True,
                dmd_steps=cfg.dmd_steps,
                rank=dist.get_rank(),
            )
            return RefKFDMDGeneratorPipeline(
                **common,
                scheduler=scheduler,
                device=device,
                vae_compile=False,
                vae_compile_mode="max-autotune",
            )

        raise ValueError(f"Unknown model_type: {model_type!r}")