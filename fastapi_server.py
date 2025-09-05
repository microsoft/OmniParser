from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import uvicorn
import argparse
import sys
import os
import io
import base64

from PIL import Image
from shutil import copy2
from pathlib import Path

# Align import path with omnitool/omniparserserver
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_dir)
from util.omniparser import Omniparser as RootOmniparser


def parse_arguments():
    parser = argparse.ArgumentParser(description='OmniParser API')
    parser.add_argument('--som_model_path', type=str, required=False, default=None,
                        help='Local path to YOLO model weights (.pt). If not provided, will attempt to auto-install to weights/icon_detect/model.pt')
    parser.add_argument('--caption_model_name', type=str, default='florence2', help='Caption model name (florence2|blip2)')
    parser.add_argument('--caption_model_path', type=str, required=False, default=None,
                        help='Local path to caption model directory. If not provided, will attempt to auto-install to weights/icon_caption_florence')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda|cpu)')
    parser.add_argument('--BOX_TRESHOLD', type=float, default=0.05, help='YOLO confidence threshold')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Bind host')
    parser.add_argument('--port', type=int, default=8510, help='Bind port')
    parser.add_argument('--no_auto_install', action='store_true', help='Disable auto-install of model weights when missing')
    return parser.parse_args()


args = parse_arguments()


def _maybe_auto_install_models(cfg: dict, allow_network: bool = True) -> dict:
    """Ensure local model files exist. If paths are not provided or missing,
    download the V2 weights into ./weights using huggingface_hub, mirroring README steps.
    """
    weights_dir = Path(root_dir) / 'weights'
    icon_detect_dir = weights_dir / 'icon_detect'
    florence_dir = weights_dir / 'icon_caption_florence'
    default_yolo_path = icon_detect_dir / 'model.pt'

    # Fill defaults when not provided
    som_path = cfg.get('som_model_path') or str(default_yolo_path)
    cap_dir = cfg.get('caption_model_path') or str(florence_dir)

    # If both present and exist, nothing to do
    som_exists = os.path.isfile(som_path)
    cap_exists = os.path.isdir(cap_dir)

    if (som_exists and cap_exists) or not allow_network:
        # When network disabled or already present, just return the effective paths
        cfg['som_model_path'] = som_path
        cfg['caption_model_path'] = cap_dir
        return cfg

    # Attempt auto-install using huggingface_hub
    try:
        from huggingface_hub import snapshot_download
        weights_dir.mkdir(parents=True, exist_ok=True)
        # Download both subfolders from microsoft/OmniParser-v2.0
        # Place under ./weights (no symlinks so server can run from local files)
        snapshot_download(
            repo_id='microsoft/OmniParser-v2.0',
            allow_patterns=['icon_detect/*', 'icon_caption/*'],
            local_dir=str(weights_dir),
            local_dir_use_symlinks=False,
        )
        # Rename caption dir to icon_caption_florence per README
        src_caption = weights_dir / 'icon_caption'
        if src_caption.exists() and not florence_dir.exists():
            src_caption.rename(florence_dir)
    except Exception as e:
        print('[OmniParser] Auto-install failed via snapshot_download:', e)
        # Best-effort fallback: try downloading exact files
        try:
            from huggingface_hub import hf_hub_download
            icon_detect_dir.mkdir(parents=True, exist_ok=True)
            for f in ['train_args.yaml', 'model.pt', 'model.yaml']:
                try:
                    local_fp = hf_hub_download('microsoft/OmniParser-v2.0', filename=f'icon_detect/{f}')
                    dest = icon_detect_dir / f
                    if not dest.exists():
                        copy2(local_fp, dest)
                except Exception as ee:
                    print(f'[OmniParser] Failed to fetch icon_detect/{f}:', ee)
            # caption weights
            tmp_caption = weights_dir / 'icon_caption'
            tmp_caption.mkdir(parents=True, exist_ok=True)
            for f in ['config.json', 'generation_config.json', 'model.safetensors']:
                try:
                    local_fp = hf_hub_download('microsoft/OmniParser-v2.0', filename=f'icon_caption/{f}')
                    dest = tmp_caption / f
                    if not dest.exists():
                        copy2(local_fp, dest)
                except Exception as ee:
                    print(f'[OmniParser] Failed to fetch icon_caption/{f}:', ee)
            if tmp_caption.exists() and not florence_dir.exists():
                tmp_caption.rename(florence_dir)
        except Exception as e2:
            print('[OmniParser] Auto-install fallback also failed:', e2)

    # Update effective paths after attempted install
    cfg['som_model_path'] = cfg.get('som_model_path') or str(default_yolo_path)
    cfg['caption_model_path'] = cfg.get('caption_model_path') or str(florence_dir)
    return cfg


CONFIG = {
    'som_model_path': args.som_model_path,
    'caption_model_name': args.caption_model_name,
    'caption_model_path': args.caption_model_path,
    'BOX_TRESHOLD': args.BOX_TRESHOLD,
    'device': args.device,
}

# If model paths are not provided or missing, attempt auto-install
if not args.no_auto_install:
    print('[OmniParser] Ensuring local model assets (auto-install if missing)...')
    CONFIG = _maybe_auto_install_models(CONFIG, allow_network=True)

app = FastAPI(title="Root OmniParser API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _validate_local_models(cfg: dict) -> dict:
    """Validate required local model paths; do not download.
    Exits with a clear error if paths are missing.
    """
    som_path = cfg.get('som_model_path')
    if not som_path or not os.path.isfile(som_path):
        raise SystemExit(
            f"[OmniParser] Missing YOLO weights. Provide --som_model_path or allow auto-install (expected: weights/icon_detect/model.pt). Got: {som_path!r}"
        )
    print("[OmniParser] YOLO weights at:", som_path)

    cap_dir = cfg.get('caption_model_path')
    if not cap_dir or not os.path.isdir(cap_dir):
        raise SystemExit(
            f"[OmniParser] Missing caption model directory. Provide --caption_model_path or allow auto-install (expected: weights/icon_caption_florence). Got: {cap_dir!r}"
        )
    print("[OmniParser] Caption model directory at:", cap_dir)
    return cfg


print("[OmniParser] Verifying required local model assets...")
CONFIG = _validate_local_models(CONFIG)
print("[OmniParser] Model assets verified locally. Initializing core Omniparser...")
OMNIPARSER_INSTANCE = RootOmniparser(CONFIG)
print("[OmniParser] Core Omniparser initialized.")


class ParseRequest(BaseModel):
    image_base64: Optional[str] = Field(default=None, description="Image as base64 string or data URL")


class Element(BaseModel):
    type: str
    bbox: List[float]
    interactivity: bool
    content: Optional[str] = None


class ParseResponse(BaseModel):
    annotated_image_base64: str
    annotation_list: List[Element]


@app.get('/health')
def health():
    return {"status": "ok"}


@app.get('/config')
def config():
    return CONFIG


def _get_bytes(image: UploadFile | None, body: ParseRequest | None) -> bytes:
    if image is not None:
        return image.file.read()
    if body and body.image_base64:
        s = body.image_base64
        if s.startswith('data:image'):
            try:
                _, s = s.split(',', 1)
            except ValueError:
                raise HTTPException(status_code=400, detail='Malformed data URL')
        try:
            return base64.b64decode(s)
        except Exception:
            raise HTTPException(status_code=400, detail='Invalid base64 image string')
    raise HTTPException(status_code=400, detail='Provide an image file or image_base64')


@app.post('/parse', response_model=ParseResponse)
def parse(image: UploadFile = File(None), body: ParseRequest | None = None):
    img_bytes = _get_bytes(image, body)
    # verify image
    try:
        Image.open(io.BytesIO(img_bytes)).verify()
    except Exception:
        raise HTTPException(status_code=400, detail='Invalid image data')

    image_b64 = base64.b64encode(img_bytes).decode('ascii')
    som_b64, parsed_list = OMNIPARSER_INSTANCE.parse(image_b64)

    # Normalize output fields to match action model format
    out_elems = []
    for item in parsed_list:
        bbox = [float(x) for x in item.get('bbox', [])]
        out_elems.append(
            Element(
                type=item.get('type', 'icon'),
                bbox=bbox,
                interactivity=bool(item.get('interactivity', True)),
                content=item.get('content'),
            )
        )

    return ParseResponse(
        annotated_image_base64=som_b64,
        annotation_list=out_elems,
    )


if __name__ == '__main__':
    # Run using the app object directly to avoid import-by-string issues
    uvicorn.run(app, host=args.host, port=args.port, reload=False)
