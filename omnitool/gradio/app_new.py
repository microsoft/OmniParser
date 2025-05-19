"""
The app contains:
- a new UI for the OmniParser AI Agent.
- 
python app_new.py --windows_host_url localhost:8006 --omniparser_server_url localhost:8000
"""

import os
import io
import shutil
import mimetypes
from datetime import datetime
from enum import StrEnum
from functools import partial
from pathlib import Path
from typing import cast, List, Optional
import argparse
import gradio as gr
from anthropic import APIResponse
from anthropic.types import TextBlock
from anthropic.types.beta import BetaMessage, BetaTextBlock, BetaToolUseBlock
from anthropic.types.tool_use_block import ToolUseBlock
from loop import (
    APIProvider,
    sampling_loop_sync,
)
from tools import ToolResult
import requests
from requests.exceptions import RequestException
import base64

CONFIG_DIR = Path("~/.anthropic").expanduser()
API_KEY_FILE = CONFIG_DIR / "api_key"

INTRO_TEXT = '''
<div style="text-align: center; margin-bottom: 10px;">
    <h2>OmniParser AI Agent</h2>
    <p>Turn any vision-language model into an AI agent. We currently support <b>OpenAI (4o/o1/o3-mini), DeepSeek (R1), Qwen (2.5VL) or Anthropic Computer Use (Sonnet)</b>.</p>
    <p>Type a message and press send to start OmniTool. Press stop to pause, and press the trash icon in the chat to clear the message history.</p>
    <p>You can also upload files for analysis using the file upload section.</p>
</div>
'''

def parse_arguments():
    parser = argparse.ArgumentParser(description="Gradio App")
    parser.add_argument("--windows_host_url", type=str, default='localhost:8006')
    parser.add_argument("--omniparser_server_url", type=str, default="localhost:8000")
    parser.add_argument("--run_folder", type=str, default="./tmp/outputs")
    return parser.parse_args()
args = parse_arguments()

# Update upload folder from args if provided
RUN_FOLDER = Path(os.path.join(args.run_folder, datetime.now().strftime('%Y%m%d_%H%M')))
RUN_FOLDER.mkdir(parents=True, exist_ok=True)

class Sender(StrEnum):
    USER = "user"
    BOT = "assistant"
    TOOL = "tool"


def load_existing_files():
    """Load all existing files from the uploads folder"""
    files = []
    if RUN_FOLDER.exists():
        for file_path in RUN_FOLDER.iterdir():
            if file_path.is_file():
                files.append(str(file_path))
    return files

def setup_state(state):
    if "messages" not in state:
        state["messages"] = []
    if "model" not in state:
        state["model"] = "omniparser + gpt-4o-orchestrated"
