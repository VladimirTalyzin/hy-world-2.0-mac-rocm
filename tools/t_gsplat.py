"""Smoke-test the HIP build of gsplat: rasterize a few gaussians and check the
forward output, then a backward pass to exercise the cooperative-groups shims."""
import os, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "HY-World-2.0"))
from hyworld2 import compat
from gsplat.rendering import rasterization

d = compat.get_device()
torch.manual_seed(0)
N, W, H = 200, 128, 128
means   = torch.randn(N, 3, device=d) * 0.3 + torch.tensor([0., 0., 3.], device=d)
quats   = torch.nn.functional.normalize(torch.randn(N, 4, device=d), dim=-1)
scales  = torch.rand(N, 3, device=d) * 0.05 + 0.01
opac    = torch.rand(N, device=d) * 0.5 + 0.5
colors  = torch.rand(N, 3, device=d)
viewmat = torch.eye(4, device=d)[None]
K = torch.tensor([[[100., 0., W/2], [0., 100., H/2], [0., 0., 1.]]], device=d)

for t in (means, quats, scales, opac, colors):
    t.requires_grad_(True)

img, alpha, meta = rasterization(means=means, quats=quats, scales=scales,
                                 opacities=opac, colors=colors,
                                 viewmats=viewmat, Ks=K, width=W, height=H)
print(f"forward : {tuple(img.shape)} finite={bool(torch.isfinite(img).all())} "
      f"range=[{img.min().item():.4f}, {img.max().item():.4f}] "
      f"alpha_mean={alpha.mean().item():.4f}", flush=True)

img.sum().backward()
gnames = ["means", "quats", "scales", "opacities", "colors"]
grads = [means.grad, quats.grad, scales.grad, opac.grad, colors.grad]
ok = all(g is not None and torch.isfinite(g).all() for g in grads)
print("backward: " + ", ".join(
    f"{n}={'ok' if (g is not None and torch.isfinite(g).all()) else 'BAD'}"
    for n, g in zip(gnames, grads)), flush=True)
print("RESULT:", "PASS" if (ok and torch.isfinite(img).all() and img.abs().sum() > 0) else "FAIL")
