from pathlib import Path
from uuid import uuid4
import requests
from PIL import Image
from .base import BaseAnthropicTool, ToolError
from io import BytesIO

OUTPUT_DIR = "./tmp/outputs"

def get_screenshot(resize: bool = False, target_width: int = 1920, target_height: int = 1080):
    """Capture screenshot by requesting from HTTP endpoint - returns native resolution unless resized"""
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"screenshot_{uuid4().hex}.png"
    
    try:
        response = requests.get('http://localhost:5000/screenshot')
        if response.status_code != 200:
            raise ToolError(f"Failed to capture screenshot: HTTP {response.status_code}")
        
        # (1280, 800)
        screenshot = Image.open(BytesIO(response.content))
        
        if resize and screenshot.size != (target_width, target_height):
            screenshot = screenshot.resize((target_width, target_height))
        screenshot.save(path)
        return screenshot, path
    except Exception as e:
        raise ToolError(f"Failed to capture screenshot: {str(e)}")