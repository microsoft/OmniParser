import modal
from pathlib import Path
import sys
import fastapi

# Add the repository root to Python path for imports
repo_root = Path(__file__).parent
sys.path.append(str(repo_root))

from util.utils import get_yolo_model, get_caption_model_processor
from gradio_demo import create_gradio_demo

# Create image with all dependencies
image = (
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
)

# Copy required files to container
image = (
    image.copy_local_file(
        "weights/icon_detect/model.pt", "/root/weights/icon_detect/model.pt"
    )
    .copy_local_dir(
        "weights/icon_caption_florence", "/root/weights/icon_caption_florence"
    )
    .copy_local_dir("util", "/root/util")
    .copy_local_file("gradio_demo.py", "/root/gradio_demo.py")
)

# Create Modal app
app = modal.App("omniparser-v2", image=image)

@app.cls(gpu="T4", container_idle_timeout=60, concurrency_limit=10)
class GradioWebApp:
    demo = None
    port = 7860

    @modal.enter()
    def startup(self):
        self._configure_torch()
        self._initialize_models()

    def _configure_torch(self):
        import torch

        torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32
        torch.backends.cudnn.benchmark = True  # Enable cuDNN autotuner

    def _initialize_models(self):
        self.yolo_model = get_yolo_model(
            model_path="/root/weights/icon_detect/model.pt"
        )
        self.caption_model_processor = get_caption_model_processor(
            model_name="florence2",
            model_name_or_path="/root/weights/icon_caption_florence",
        )

    @modal.asgi_app()
    def fastapi_app(self):
        import gradio as gr
        from fastapi import FastAPI
        from gradio.routes import mount_gradio_app

        web_app = FastAPI()

        demo = create_gradio_demo(
            yolo_model=self.yolo_model,
            caption_model_processor=self.caption_model_processor,
        )

        return mount_gradio_app(
            app=web_app,
            blocks=demo,
            path="/",
        )

@app.local_entrypoint()
def main():
    GradioWebApp.startup.local()

