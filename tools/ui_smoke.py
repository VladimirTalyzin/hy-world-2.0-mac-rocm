#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Drive the web UI end to end in a real browser and take the README screenshots.

    python tools/ui_smoke.py --url http://127.0.0.1:7860 [--pano PHOTO] [--skip-pano]

Runs, through the actual page (Playwright + Chromium): a reconstruction of the
bundled "realistic/Desk" example, a panorama from PHOTO at 1024x512 / 20 steps,
the panorama -> 3D path on that result, and one export on each. Every step
waits for the status line to say "Finished" and screenshots the tab into
assets/. It is a smoke test as much as a screenshot tool: if any flow breaks in
the browser, this is where it shows.

Needs:  pip install playwright && python -m playwright install chromium
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
ASSETS = PROJECT / "assets"


def wait_status(page, tab_root, timeout_s: float, previous: str = "") -> str:
    """Poll the tab's status line until it reports finished or failed.
    ``previous`` is the line left by an earlier job on the same tab, which
    must not count as this job's result."""
    t0 = time.time()
    last = ""
    while time.time() - t0 < timeout_s:
        try:
            txt = tab_root.locator(".hy-status").first.inner_text(timeout=60_000).strip()
        except Exception:  # noqa: BLE001 - the renderer can starve the page while splats sort in software
            continue
        if txt != last:
            print(f"    [{time.time() - t0:6.0f}s] {txt.splitlines()[0][:100]}", flush=True)
            last = txt
        if (txt.startswith("✅") or txt.startswith("❌")) and txt != previous:
            return txt
        time.sleep(3)
    raise TimeoutError(f"status still '{last}' after {timeout_s:.0f} s")


def wait_viewer(page, tab_root, needle: str, timeout_s: float = 240) -> None:
    """Wait until the viewer iframe inside the tab reports its layers loaded."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            src = tab_root.locator("iframe").first.get_attribute("src", timeout=60_000) or ""
            for frame in page.frames:
                if "/viewer/" in frame.url and src and src in frame.url:
                    txt = frame.locator("#hud").inner_text(timeout=30_000)
                    if needle in txt:
                        time.sleep(3)  # let the splats sort once more before the shot
                        return
        except Exception as e:  # noqa: BLE001 - see wait_status
            print(f"    (viewer probe: {type(e).__name__})", flush=True)
        time.sleep(3)
    print("    (viewer did not report ready; taking the screenshot anyway)")


def pick(page, root, dropdown_label: str, option_text: str) -> None:
    """Choose an option in a Gradio dropdown found by its label inside ``root``."""
    root.get_by_label(dropdown_label, exact=True).first.click()
    page.get_by_role("option", name=option_text, exact=True).first.click()


def shot(page, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(path), full_page=False, animations="disabled", timeout=300_000)
    print(f"    screenshot -> {path} ({path.stat().st_size / 1024:.0f} KB)", flush=True)
    # Software WebGL keeps the renderer so busy that later clicks time out;
    # once the shot is taken the viewer has done its job, so unload it.
    page.evaluate("document.querySelectorAll('iframe').forEach(f => { f.src = 'about:blank'; })")
    time.sleep(2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:7860")
    ap.add_argument("--pano", default=str(PROJECT / "HY-World-2.0/examples/worldrecon/realistic/Landmark/frame_0000.png"),
                    help="input photo for the panorama")
    ap.add_argument("--skip-recon", action="store_true")
    ap.add_argument("--skip-pano", action="store_true")
    ap.add_argument("--skip-p3d", action="store_true")
    ap.add_argument("--width", type=int, default=1500)
    ap.add_argument("--height", type=int, default=1000)
    ap.add_argument("--headed", action="store_true", help="open a visible browser window (uses the real GPU)")
    args = ap.parse_args()

    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        # Headless Chromium renders WebGL through SwiftShader, so the splat
        # viewer is slow and a screenshot can take a minute (hence the long
        # timeout in shot()). --headed uses the real GPU where a display exists.
        browser = p.chromium.launch(headless=not args.headed, args=["--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"])
        page = browser.new_page(viewport={"width": args.width, "height": args.height}, device_scale_factor=1)
        page.set_default_timeout(180_000)
        page.goto(args.url, wait_until="load")
        page.wait_for_selector("text=HY-World 2.0")
        print("page loaded", flush=True)

        # ------------------------------------------------------------ 3D scene
        if not args.skip_recon:
            print("== 3D scene: bundled example realistic/Desk", flush=True)
            page.get_by_role("tab", name=re.compile("3D scene")).click()
            tab = page.locator("div[role=tabpanel]:visible").first
            pick(page, tab, "…or pick a bundled example", "realistic/Desk")
            time.sleep(2)
            page.get_by_role("button", name=re.compile("Reconstruct 3D scene")).click()
            status = wait_status(page, tab, 20 * 60)
            if not status.startswith("✅"):
                print("reconstruction failed:", status); return 1
            wait_viewer(page, tab, "cameras")
            page.mouse.wheel(0, 0)
            shot(page, ASSETS / "screenshot.png")
            # one export from this tab, to prove the panel works after a real run
            pick(page, tab, "Format", "mesh.glb — textured mesh from the depth maps, glTF 2.0")
            tab.get_by_role("button", name=re.compile("Export")).first.click()
            status = wait_status(page, tab, 5 * 60, previous=status)
            print("   ", status.splitlines()[0])
            tab.locator(".hy-status").first.scroll_into_view_if_needed()
            shot(page, ASSETS / "screenshot_export.png")

        # ------------------------------------------------------------ panorama
        if not args.skip_pano:
            print("== Panorama: 1024x512, 20 steps", flush=True)
            page.get_by_role("tab", name=re.compile("Panorama$")).click()
            tab = page.locator("div[role=tabpanel]:visible").first
            tab.locator("input[type=file]").first.set_input_files(args.pano)
            time.sleep(2)
            tab.get_by_text("Generation parameters").first.click()
            time.sleep(0.5)
            steps = tab.get_by_label("number input for Diffusion steps").first
            steps.fill("20"); steps.press("Enter")
            pick(page, tab, "Panorama size", "1024 × 512 (quick test)")
            tab.get_by_label("Prompt (what the surroundings look like)").first.fill(
                "an old European city square, arcades, afternoon light")
            page.get_by_role("button", name=re.compile("Generate panorama")).click()
            status = wait_status(page, tab, 40 * 60)
            if not status.startswith("✅"):
                print("panorama failed:", status); return 1
            time.sleep(4)
            shot(page, ASSETS / "screenshot_panorama.png")
            pick(page, tab, "Format", "panorama.html — 360° HTML5 + three.js viewer, single file")
            tab.get_by_role("button", name=re.compile("Export")).first.click()
            print("   ", wait_status(page, tab, 5 * 60, previous=status).splitlines()[0])

        # ------------------------------------------------------------ panorama -> 3D
        if not args.skip_p3d:
            print("== Panorama -> 3D", flush=True)
            page.get_by_role("tab", name=re.compile("Panorama → 3D")).click()
            tab = page.locator("div[role=tabpanel]:visible").first
            if not args.skip_pano:
                page.get_by_role("button", name=re.compile("Use the last generated panorama")).click()
            else:
                panos = sorted((PROJECT / "outputs" / "ui").glob("pano_*/panorama.png"), key=lambda q: q.stat().st_mtime)
                if not panos:
                    print("no panorama under outputs/ui to build from"); return 1
                print("    using", panos[-1])
                tab.locator("input[type=file]").first.set_input_files(str(panos[-1]))
            time.sleep(2)
            page.get_by_role("button", name=re.compile("Build 3D scene from the panorama")).click()
            status = wait_status(page, tab, 40 * 60)
            if not status.startswith("✅"):
                print("panorama -> 3D failed:", status); return 1
            wait_viewer(page, tab, "points")
            shot(page, ASSETS / "screenshot_pano3d.png")
            pick(page, tab, "Format", "scene_web.zip — HTML5 + three.js scene, folder with a one-click local server")
            tab.get_by_role("button", name=re.compile("Export")).first.click()
            print("   ", wait_status(page, tab, 5 * 60, previous=status).splitlines()[0])

        browser.close()
    print("all flows finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
