import modal
from pathlib import Path
import sys

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

@stub.function(image=image, gpu="H100", timeout=3600)  # 1 hour timeout
def run_app():
    """Run the Gradio app in Modal with GPU acceleration"""
    import threading
    import time
    
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

    class InactivityMonitor:
        def __init__(self, demo, timeout=60):
            self.demo = demo
            self.timeout = timeout
            self.last_request_time = time.time()
            self.lock = threading.Lock()
            self._start_monitor()
            
        def update(self):
            with self.lock:
                self.last_request_time = time.time()
                
        def _monitor(self):
            while True:
                time.sleep(10)
                with self.lock:
                    if time.time() - self.last_request_time > self.timeout:
                        print("No requests in 1 minute, stopping the app.")
                        self.demo.close()
                        sys.exit()  # Exit the process
                        
        def _start_monitor(self):
            thread = threading.Thread(target=self._monitor, daemon=True)
            thread.start()

    # Create inactivity monitor
    monitor = InactivityMonitor(demo)

    # Launch app with optimized settings
    demo.queue(max_size=20).launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        max_threads=16,
        show_error=True,
        root_path="/omniparser"
    )

    # Update activity timestamp
    monitor.update()

@stub.local_entrypoint()
def main():
    run_app.remote()
