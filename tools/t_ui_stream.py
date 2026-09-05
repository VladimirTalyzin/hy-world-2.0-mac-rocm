"""Check that the running UI streams its status line, without a browser.

Driving Gradio through a headless browser turned out to be unreliable here
(synthetic clicks and file-input assignment do not always reach the app), which
made it impossible to tell a UI bug from an automation artefact. gradio_client
speaks the same queue protocol the page does, so this exercises the real
handler -- generator, stage tracker and status renderer -- over HTTP.
"""
import sys
import time
from pathlib import Path

from gradio_client import Client, handle_file

# The UI's labels are Russian and Chinese; gradio_client prints them while
# introspecting, which a cp1251 console cannot encode.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
VIEWS = sorted((ROOT / "outputs" / "ui" / "pano3d_20260902_040854" / "images").glob("view_0[0-2].png"))
URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:7860/"


def main() -> None:
    if not VIEWS:
        raise SystemExit("no example views to send; run the panorama -> 3D tab once first")
    client = Client(URL, verbose=False)

    api = client.view_api(return_format="dict")
    named = api["named_endpoints"]
    print(f"named endpoints: {sorted(named)}\n")
    endpoint = "/run_reconstruction"
    if endpoint not in named:
        raise SystemExit(f"{endpoint} is missing; the UI exposes {sorted(named)}")
    print(f"calling {endpoint} with {len(VIEWS)} views of identical shape\n")

    job = client.submit(
        [handle_file(str(p)) for p in VIEWS],  # files
        None,        # example
        952,         # target size (must be one of the dropdown's choices)
        True,        # bf16
        True, True, False,        # sky, edge, confidence
        500_000,     # max gaussians
        False, 15,   # render video, interpolation
        "ru",        # language
        api_name=endpoint,
    )

    seen, t0 = [], time.time()
    while not job.done() and time.time() - t0 < 420:
        for out in job.outputs()[len(seen):]:
            status = out[0] if isinstance(out, (list, tuple)) else out
            line = str(status).splitlines()[0] if status else ""
            if line and (not seen or seen[-1] != line):
                print(f"  [{time.time() - t0:6.1f}s] {line}")
            seen.append(out)
        time.sleep(1.0)

    print(f"\nstatus frames received: {len(seen)}")
    if seen:
        print(f"final status: {str(seen[-1][0]).splitlines()[0]!r}")
    print("PASS" if len(seen) > 5 else "FAIL: the UI did not stream intermediate updates")


if __name__ == "__main__":
    main()
