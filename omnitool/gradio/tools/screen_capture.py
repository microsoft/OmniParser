import os
from pathlib import Path
from uuid import uuid4
import requests
from PIL import Image
from .base import BaseAnthropicTool, ToolError
from io import BytesIO
import pyautogui

OUTPUT_DIR = "./tmp/outputs"

def get_screenshot(resize: bool = False, target_width: int = 1920, target_height: int = 1080):
    """Capture screenshot by requesting from HTTP endpoint - returns native resolution unless resized"""
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"screenshot_{uuid4().hex}.png"
    
    try:
        screenshot = pyautogui.screenshot()
        size = pyautogui.size()

        target_width = size.width
        target_height = size.height
        
        screenshot = screenshot.resize((target_width, target_height))

        cursor_path = os.path.join(os.path.dirname(__file__), "cursor.png")
        cursor_x, cursor_y = pyautogui.position()
        cursor = Image.open(cursor_path)
        # make the cursor smaller
        cursor = cursor.resize((int(cursor.width / 1.5), int(cursor.height / 1.5)))
        screenshot.paste(cursor, (cursor_x, cursor_y), cursor)

        screenshot.save(path)
        return screenshot, path
    except Exception as e:
        raise ToolError(f"Failed to capture screenshot: {str(e)}")