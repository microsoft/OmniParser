import modal
from pathlib import Path
import sys
import os
import fastapi
from typing import Optional

# Add the repository root to Python path for imports
repo_root = Path(__file__).parent
sys.path.append(str(repo_root))

from util.utils import get_yolo_model, get_caption_model_processor
from gradio_demo import create_gradio_demo

# Environment configuration with defaults
APP_NAME = os.environ.get("MODAL_APP_NAME", "omniparser-v2")
GPU_CONFIG = os.environ.get("MODAL_GPU_CONFIG", "L4")
CONTAINER_TIMEOUT = int(os.environ.get("MODAL_CONTAINER_TIMEOUT", "120"))
MAX_CONCURRENT_INPUTS = int(os.environ.get("MODAL_MAX_CONCURRENT_INPUTS", "50"))
CONCURRENCY_LIMIT = int(os.environ.get("MODAL_CONCURRENCY_LIMIT", "1")) # Multiple concurrent requests crash the model
GRADIO_PORT = int(os.environ.get("GRADIO_PORT", "7860"))

def create_image():
    """Create and configure the Modal image with all dependencies."""
    return (
        modal.Image.debian_slim()
        .apt_install("libgl1-mesa-glx", "libglib2.0-0")
        .pip_install(
            "torch",
            "torchvision",
            "transformers",
            "accelerate",
            "timm",
            "einops==0.8.0",
            "easyocr",
            "supervision==0.18.0",
            "ultralytics==8.3.70",
            "opencv-python",
            "opencv-python-headless",
            "paddlepaddle",
            "paddleocr",
            "gradio",
            "streamlit>=1.38.0",
            "screeninfo",
            "uiautomation",
            "pyautogui==0.9.54",
            "openai==1.3.5",
            "anthropic[bedrock,vertex]>=0.37.1",
            "dashscope",
            "groq",
            "azure-identity",
            "boto3>=1.28.57",
            "google-auth<3,>=2",
            "httpx>=0.24.0",
            "numpy==1.26.4",
            "dill",
            "jsonschema==4.22.0",
            "ruff==0.6.7",
            "pre-commit==3.8.0",
            "pytest==8.3.3",
            "pytest-asyncio==0.23.6",
        )
        .copy_local_file(
            "weights/icon_detect/model.pt", "/root/weights/icon_detect/model.pt"
        )
        .copy_local_dir(
            "weights/icon_caption_florence", "/root/weights/icon_caption_florence"
        )
        .copy_local_dir("util", "/root/util")
        .copy_local_file("gradio_demo.py", "/root/gradio_demo.py")
    )

# Create Modal app with the specified name
app = modal.App(APP_NAME, image=create_image())

@app.cls(
    gpu=GPU_CONFIG,
    container_idle_timeout=CONTAINER_TIMEOUT,
    allow_concurrent_inputs=MAX_CONCURRENT_INPUTS,
    concurrency_limit=CONCURRENCY_LIMIT,
)
class GradioWebApp:
    """Gradio web application with configurable parameters."""
    
    def __init__(self):
        self.port = GRADIO_PORT

    @modal.enter()
    def startup(self):
        """Initialize models and configure runtime environment."""
        self._configure_torch()
        self._initialize_models()

    def _configure_torch(self):
        """Configure PyTorch runtime settings."""
        import torch
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    def _initialize_models(self):
        """Initialize ML models."""
        self.yolo_model = get_yolo_model(
            model_path="/root/weights/icon_detect/model.pt"
        )
        self.caption_model_processor = get_caption_model_processor(
            model_name="florence2",
            model_name_or_path="/root/weights/icon_caption_florence",
        )

    @modal.asgi_app()
    def fastapi_app(self):
        """Create and configure the FastAPI application with Gradio interface."""
        import gradio as gr
        from fastapi import FastAPI
        from gradio.routes import mount_gradio_app

        web_app = FastAPI()
        demo = create_gradio_demo(
            yolo_model=self.yolo_model,
            caption_model_processor=self.caption_model_processor,
        ).queue(default_concurrency_limit=CONCURRENCY_LIMIT)

        return mount_gradio_app(
            app=web_app,
            blocks=demo,
            path="/",
        )

@app.local_entrypoint()
def main():
    """Local development entrypoint."""
    GradioWebApp.startup.local()

if __name__ == "__main__":
    main()
