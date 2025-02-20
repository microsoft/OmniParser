import modal
from pathlib import Path
import sys
import time
import threading
from typing import Optional

# Add the repository root to Python path for imports
repo_root = Path(__file__).parent
sys.path.append(str(repo_root))

from util.utils import get_yolo_model, get_caption_model_processor
from gradio_demo import create_gradio_demo

# Create Modal stub
stub = modal.Stub("omniparser-v2")

# Create image with all dependencies
image = (
    modal.Image.debian_slim()
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        # Deep learning frameworks
        "torch",
        "torchvision",
        "transformers",
        "accelerate",
        "timm",
        "einops==0.8.0",
        
        # Computer vision & OCR
        "easyocr",
        "supervision==0.18.0", 
        "ultralytics==8.3.70",
        "opencv-python",
        "opencv-python-headless",
        "paddlepaddle",
        "paddleocr",
        
        # UI & visualization
        "gradio",
        "streamlit>=1.38.0",
        "screeninfo",
        "uiautomation",
        "pyautogui==0.9.54",
        
        # AI services & utilities
        "openai==1.3.5",
        "anthropic[bedrock,vertex]>=0.37.1",
        "dashscope",
        "groq",
        "azure-identity",
        "boto3>=1.28.57",
        "google-auth<3,>=2",
        
        # Data processing
        "numpy==1.26.4",
        "dill",
        "jsonschema==4.22.0",
        
        # Development tools
        "ruff==0.6.7",
        "pre-commit==3.8.0",
        "pytest==8.3.3",
        "pytest-asyncio==0.23.6",
    )
)

# Copy required files to container
image = (
    image.copy_local_file("weights/icon_detect/model.pt", "/root/weights/icon_detect/model.pt")
    .copy_local_dir("weights/icon_caption_florence", "/root/weights/icon_caption_florence")
    .copy_local_dir("util", "/root/util")
    .copy_local_file("gradio_demo.py", "/root/gradio_demo.py")
)

@stub.function(
    image=image,
    gpu="H100",
    timeout=21600, # 6 hours
    memory=32768,  # 32GB RAM to match H100's capabilities
    container_idle_timeout=60  # Shutdown container after 60 seconds of inactivity
)
def run_app() -> None:
    """Run the Gradio app in Modal with GPU acceleration."""
    # Initialize models with optimized settings
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for faster matrix operations
    torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner
    
    # Initialize models
    yolo_model = get_yolo_model(model_path="/root/weights/icon_detect/model.pt")
    caption_model_processor = get_caption_model_processor(
        model_name="florence2", 
        model_name_or_path="/root/weights/icon_caption_florence"
    )

    # Create Gradio interface
    demo = create_gradio_demo(
        yolo_model=yolo_model,
        caption_model_processor=caption_model_processor
    )

    # Launch app with H100-optimized settings
    demo.queue(
        max_size=None,
        status_update_rate=0.25,
        default_concurrency_limit=8  # Increased for H100's higher throughput
    ).launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        max_threads=64,  # Increased for better H100 utilization
        show_error=True,
        root_path="/omniparser",
        ssl_verify=False,
    )

@stub.local_entrypoint()
def main() -> None:
    """Main entrypoint for the Modal app."""
    run_app.remote()
