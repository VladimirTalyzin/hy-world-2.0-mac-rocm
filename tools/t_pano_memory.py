"""Find out why the panorama stalls after denoising, at 1952x960.

Symptoms, measured while it was stuck: the CPU spinning at ~1.1 cores, the GPU
at 4%, no page faults, no TDR reset (TdrDelay is already 60), and 60.9 GB of
the 64 GB carve-out in use. The stack sat on the first synchronising copy after
the denoising loop -- before `vae.decode` is even called -- which is where the
runtime would block if it could not get memory.

If that reading is right, the same pipeline should complete at a smaller output
and stall at the larger one, with the difference visible in the allocator. So
this loads the model once and walks up the sizes, printing what is reserved at
each step, and offers `--free-transformer` to test the obvious remedy: the 20B
transformer is dead weight by the time the VAE decodes, and moving it aside
should buy the decode all the room it needs.
"""
import argparse
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "HY-World-2.0"))
sys.path.append(str(ROOT / "HY-World-2.0" / "hyworld2" / "panogen"))
from hyworld2 import compat  # noqa: E402

import torch  # noqa: E402
from pipeline_with_qwen_image import HunyuanPanoPipeline  # noqa: E402

SRC = ROOT / "outputs" / "ui" / "pano_20260903_234258" / "input.png"
OUT = ROOT / "outputs" / "pano_memory_probe"


def mem(tag: str) -> None:
    free, total = torch.cuda.mem_get_info()
    print(f"    [{tag:<26}] allocated {torch.cuda.memory_allocated() / 2**30:6.2f} GiB | "
          f"reserved {torch.cuda.memory_reserved() / 2**30:6.2f} GiB | "
          f"free on device {free / 2**30:6.2f} / {total / 2**30:.1f} GiB", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--free-transformer", action="store_true",
                    help="move the transformer to CPU before the VAE decode")
    ap.add_argument("--steps", type=int, default=8, help="denoising steps (few: memory is the subject)")
    ap.add_argument("--sizes", default="1024x512,1536x768,1952x960")
    ap.add_argument("--in-thread", action="store_true",
                    help="run generation in a worker thread, as the web UI does")
    ap.add_argument("--like-ui", action="store_true",
                    help="also apply the UI's stage wrappers and stdout capture")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    print(compat.describe())
    t0 = time.perf_counter()
    pipeline = HunyuanPanoPipeline.from_pretrained(
        str(ROOT / "weights" / "Qwen-Image-Edit-2509"),
        lora_path=str(ROOT / "weights"), lora_subfolder="HY-Pano-2.0")
    print(f"model ready in {time.perf_counter() - t0:.1f} s")
    mem("after load")

    inner = pipeline.pipe
    device = compat.get_device()

    if args.like_ui:
        # Everything the web UI does to the pipeline that the plain CLI does
        # not: wrap the text encoder and VAE to print stage markers, and route
        # stdout through a buffer so the page can stream it. Neither should
        # matter to a GPU kernel; the point is to find out whether they do.
        def wrap(owner, name, tag):
            original = getattr(owner, name)

            def announced(*a, **kw):
                print(f"[Stage] {tag}", flush=True)
                t = time.perf_counter()
                try:
                    return original(*a, **kw)
                finally:
                    print(f"[Stage] {tag} done in {time.perf_counter() - t:.1f}s", flush=True)

            setattr(owner, name, announced)

        wrap(inner.vae, "encode", "vae-encode")
        wrap(inner.vae, "decode", "vae-decode")
        wrap(inner, "encode_prompt", "text")

        class Tee:
            def __init__(self, orig):
                self._orig, self.lock = orig, threading.Lock()
                self.lines = [""]

            def write(self, s):
                try:
                    self._orig.write(s)
                except Exception:
                    pass
                with self.lock:
                    self.lines[-1] += s
                    if len(self.lines[-1]) > 4000:
                        self.lines.append("")
                return len(s)

            def flush(self):
                try:
                    self._orig.flush()
                except Exception:
                    pass

            def isatty(self):
                return False

        sys.stdout = Tee(sys.stdout)
        sys.stderr = Tee(sys.stderr)

        # The last ingredient: the UI's status generator polls twice a second
        # from another thread while the job runs, taking the log lock and
        # asking psutil for CPU. Harmless in principle, but it is the only
        # remaining difference from the path that completes.
        import psutil

        stop = threading.Event()

        def poller():
            proc = psutil.Process()
            proc.cpu_percent()
            while not stop.is_set():
                proc.cpu_percent()
                with sys.stdout.lock:
                    _ = "\n".join(sys.stdout.lines[-300:])
                stop.wait(0.5)

        threading.Thread(target=poller, daemon=True).start()
        print("running with the UI's stage wrappers, stdout capture and status polling")

    if args.free_transformer:
        original_decode = inner.vae.decode

        def decode_with_room(*a, **kw):
            # By the time the VAE decodes, the 20B transformer has nothing left
            # to do this run; on an APU "moving it to CPU" is a copy within the
            # same physical RAM, but it releases the carve-out the decode needs.
            t = time.perf_counter()
            inner.transformer.to("cpu")
            compat.empty_cache()
            print(f"    transformer parked on CPU in {time.perf_counter() - t:.1f} s", flush=True)
            mem("transformer parked")
            try:
                return original_decode(*a, **kw)
            finally:
                t = time.perf_counter()
                inner.transformer.to(device)
                print(f"    transformer restored in {time.perf_counter() - t:.1f} s", flush=True)

        inner.vae.decode = decode_with_room
        print("transformer will be parked on CPU for each decode")

    for spec in args.sizes.split(","):
        w, h = (int(v) for v in spec.split("x"))
        print(f"\n=== {w}x{h} ===", flush=True)
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()

        def generate():
            return pipeline(str(SRC), prompt="", seed=42, height=h, width=w,
                            num_inference_steps=args.steps)

        if args.in_thread:
            # The web UI runs the pipeline off the main thread so it can stream
            # progress. That is the one remaining difference from the CLI path,
            # which completes every size, so it is worth testing on its own.
            box: dict = {}

            def worker():
                try:
                    box["out"] = generate()
                except BaseException as e:  # noqa: BLE001
                    box["err"] = e

            th = threading.Thread(target=worker, daemon=True)
            th.start()
            th.join(timeout=900)
            if th.is_alive():
                print("    STUCK: still running after 900 s in a worker thread", flush=True)
                mem("while stuck")
                return
            if "err" in box:
                raise box["err"]
            out = box["out"]
        else:
            out = generate()
        elapsed = time.perf_counter() - t
        out.save(OUT / f"pano_{w}x{h}.png")
        print(f"    DONE in {elapsed:.1f} s -> {OUT / f'pano_{w}x{h}.png'}", flush=True)
        print(f"    peak allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
        mem("after generation")


if __name__ == "__main__":
    main()
