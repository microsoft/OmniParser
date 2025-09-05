import argparse
import base64
import io
import os
import sys
import time
import subprocess
from pathlib import Path

import requests
from PIL import Image


def wait_for_health(url: str, timeout: float = 180.0) -> None:
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{url.rstrip('/')}/health", timeout=5)
            if r.status_code == 200 and r.json().get("status") == "ok":
                return
        except Exception:
            pass
        time.sleep(1.0)
    raise RuntimeError(f"Server at {url} did not become healthy within {timeout} seconds")


def parse_image(url: str, image_path: str):
    with open(image_path, 'rb') as f:
        files = {"image": (os.path.basename(image_path), f, "image/png")}
        r = requests.post(f"{url.rstrip('/')}/parse", files=files, timeout=300)
    r.raise_for_status()
    data = r.json()
    annotated_b64 = data["annotated_image_base64"]
    ann_list = data["annotation_list"]
    img_bytes = base64.b64decode(annotated_b64)
    annotated = Image.open(io.BytesIO(img_bytes))
    return annotated, ann_list


def main():
    parser = argparse.ArgumentParser(description="Test OmniParser FastAPI server end-to-end")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8510)
    parser.add_argument("--start-server", action="store_true", help="Start server as subprocess for the test")
    parser.add_argument("--initial-wait", type=float, default=10.0, help="Seconds to wait before first health check when starting the server")
    #parser.add_argument("--som_model_path", default="./OmniParser/weights/icon_detect/model.pt")
    #parser.add_argument("--caption_model_name", default="florence2")
    #parser.add_argument("--caption_model_path", default="./OmniParser/weights/icon_caption_florence")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--box_threshold", type=float, default=0.05)
    parser.add_argument("--image", default="./OmniParser/imgs/omni3.jpg")
    parser.add_argument("--output", default="./OmniParser/imgs/annotated_test.png")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    proc = None
    try:
        if args.start_server:
            cmd = [
                sys.executable,
                "OmniParser/fastapi_server.py",
                #"--som_model_path", args.som_model_path,
                #"--caption_model_name", args.caption_model_name,
                #"--caption_model_path", args.caption_model_path,
                "--device", args.device,
                "--BOX_TRESHOLD", str(args.box_threshold),
                "--host", args.host,
                "--port", str(args.port),
            ]
            print("Launching server:", " ".join(cmd))
            proc = subprocess.Popen(cmd)
            # Give the server process a short head start before health polling
            if args.initial_wait > 0:
                print(f"Waiting {args.initial_wait:.1f}s before health checks to allow initialization...")
                time.sleep(args.initial_wait)

        print(f"Waiting for server health at {base_url}/health ...")
        wait_for_health(base_url, timeout=600)
        print("Server is healthy.")

        print("GET /config:")
        cfg = requests.get(f"{base_url}/config", timeout=30).json()
        print(cfg)

        image_path = args.image
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Test image not found: {image_path}")
        print(f"Parsing image: {image_path}")
        annotated, ann_list = parse_image(base_url, image_path)

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        annotated.save(out_path)
        print(f"Annotated image saved to: {out_path}")
        print(f"Parsed {len(ann_list)} elements. Sample: {ann_list[:3]}")
        print("SUCCESS")
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    main()
