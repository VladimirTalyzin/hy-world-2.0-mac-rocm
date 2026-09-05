"""Decode-only VAE tiling for the ROCm port.

Qwen-Image's VAE is a *video* VAE built from ``nn.Conv3d``. On gfx1151 MIOpen
handles 3D convolutions reasonably up to a point and then falls off a cliff
(128 channels, 3x3x3, bf16):

    240x488     15.5 ms   (6.69 TFLOP/s)
    480x976     57.5 ms   (7.21 TFLOP/s)
    960x1952  6971.9 ms   (0.24 TFLOP/s)   <-- 30x collapse

The two directions of the VAE sit on opposite sides of that cliff for a
1952x960 panorama, so a single tiling switch is wrong either way. Measured
standalone:

    encode 768x1360   no tiling    3.41 s   |  tiling (256)  135.78 s
    decode ->960x1952 no tiling    minutes  |  tiling (256)   96.83 s

i.e. tiling is a 37x *pessimisation* for the encode and a large win for the
decode. ``diffusers`` gates both paths on one ``use_tiling`` flag, so this
module enables tiling and wraps ``encode`` to opt out of it.

Two corrections to the above, both found by running it rather than reasoning
about it, and both worth keeping in view:

* The cliff is selected by *shape*, not size. A 16:9 photo (VAE input
  1376x768) runs those convolutions at 0.03 TFLOP/s where a square one
  (1024x1024) manages 1.87, so the unconditional "never tile the encode" rule
  above left a real run grinding for 15 minutes on an ordinary photograph.
  ``hyworld2.compat.conv3d_fold`` removes the cliff altogether and is the
  right fix for the *speed* half of the problem.
* Tiling is still needed for the decode, and not for speed. With the 54 GB
  model resident, 61.4 GB of this machine's 64 GB carve-out is in use, and an
  untiled decode to 1952x960 has nowhere to put its activations: it stops dead
  with the CPU idle and the GPU idle, blocked in the first synchronising copy
  after the denoising loop. So folding and tiling are complementary --
  folding makes each tile cheap, tiling keeps the peak bounded.
"""

from __future__ import annotations

__all__ = ["enable_decode_only_tiling"]


def enable_decode_only_tiling(vae, **tile_kwargs) -> bool:
    """Tile the decode, never the encode. Idempotent; returns True if applied."""
    if not hasattr(vae, "use_tiling") or not hasattr(vae, "enable_tiling"):
        return False
    vae.enable_tiling(**tile_kwargs)
    if getattr(vae, "_hyworld_encode_untiled", False):
        return True

    original_encode = vae.encode

    def encode_without_tiling(*args, **kwargs):
        was_tiling, vae.use_tiling = vae.use_tiling, False
        try:
            return original_encode(*args, **kwargs)
        finally:
            vae.use_tiling = was_tiling

    vae.encode = encode_without_tiling
    vae._hyworld_encode_untiled = True
    return True
