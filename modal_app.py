import modal
from typing import Optional
import gradio as gr
import numpy as np
import torch
from PIL import Image
import io
import base64, os
import sys
from pathlib import Path

# Add the repository root to Python path for imports
repo_root = Path(__file__).parent
sys.path.append(str(repo_root))

from util.utils import (
    check_ocr_box,
    get_yolo_model,
    get_caption_model_processor,
    get_som_labeled_img,
)
from gradio_demo import create_gradio_demo, MARKDOWN

# Create Modal stub
stub = modal.Stub("omniparser-v2")

# Create image with all dependencies
image = (
    modal.Image.debian_slim()
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch",
        "easyocr",
        "torchvision",
        "supervision==0.18.0",
        "openai==1.3.5",
        "transformers",
        "ultralytics==8.3.70",
        "azure-identity",
        "numpy==1.26.4",
        "opencv-python",
        "opencv-python-headless",
        "gradio",
        "dill",
        "accelerate",
        "timm",
        "einops==0.8.0",
        "paddlepaddle",
        "paddleocr",
        "ruff==0.6.7",
        "pre-commit==3.8.0",
        "pytest==8.3.3",
        "pytest-asyncio==0.23.6",
        "pyautogui==0.9.54",
        "streamlit>=1.38.0",
        "anthropic[bedrock,vertex]>=0.37.1",
        "jsonschema==4.22.0",
        "boto3>=1.28.57",
        "google-auth<3,>=2",
        "screeninfo",
        "uiautomation",
        "dashscope",
        "groq",
    )
)

# Copy model weights and utils
image = image.copy_local_file(
    "weights/icon_detect/model.pt", "/root/weights/icon_detect/model.pt"
)
image = image.copy_local_dir(
    "weights/icon_caption_florence", "/root/weights/icon_caption_florence"
)
image = image.copy_local_dir("util", "/root/util")
image = image.copy_local_file("gradio_demo.py", "/root/gradio_demo.py")

@stub.function(image=image, gpu="A100", timeout=600)
def run_app():
    """Run the Gradio app in Modal with GPU acceleration"""
    # Initialize models
    yolo_model = get_yolo_model(model_path="/root/weights/icon_detect/model.pt")
    caption_model_processor = get_caption_model_processor(
        model_name="florence2", model_name_or_path="/root/weights/icon_caption_florence"
    )

    # Create Gradio interface with Modal-specific settings
    demo = create_gradio_demo(
        yolo_model=yolo_model,
        caption_model_processor=caption_model_processor,
        image_save_dir="/tmp"  # Use Modal's temp directory
    )

    # Launch the app with Modal's web endpoint configuration
    demo.queue(max_size=10).launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        max_threads=10,
        show_error=True,
    )

@stub.local_entrypoint()
def main():
    run_app.remote()
