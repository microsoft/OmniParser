"""
Streamlit implementation of the OmniTool frontend.
Usage: streamlit run app_streamlit.py -- --windows_host_url localhost:8006 --omniparser_server_url localhost:8000
"""

import os
import io
import shutil
import mimetypes
import argparse
import base64
from datetime import datetime
from pathlib import Path
from typing import cast
from enum import StrEnum
import streamlit as st
from anthropic import APIResponse
from anthropic.types import TextBlock
from anthropic.types.beta import BetaMessage, BetaTextBlock, BetaToolUseBlock
from anthropic.types.tool_use_block import ToolUseBlock
import requests
from requests.exceptions import RequestException

from loop import (
    APIProvider,
    sampling_loop_sync,
)
from tools import ToolResult

# Constants and configurations
CONFIG_DIR = Path("~/.anthropic").expanduser()
API_KEY_FILE = CONFIG_DIR / "api_key"
UPLOAD_FOLDER = Path("./uploads").absolute()
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

class Sender(StrEnum):
    USER = "user"
    BOT = "assistant"
    TOOL = "tool"

def parse_arguments():
    parser = argparse.ArgumentParser(description="Streamlit App")
    parser.add_argument("--windows_host_url", type=str, default='localhost:8006')
    parser.add_argument("--omniparser_server_url", type=str, default="localhost:8000")
    parser.add_argument("--upload_folder", type=str, default="./uploads")
    return parser.parse_known_args()[0]

def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "model" not in st.session_state:
        st.session_state.model = "omniparser + gpt-4o-orchestrated"
    if "provider" not in st.session_state:
        st.session_state.provider = "openai"
    if "api_key" not in st.session_state:
        st.session_state.api_key = os.getenv("OPENAI_API_KEY", "")
    if "anthropic_api_key" not in st.session_state:
        st.session_state.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if "only_n_most_recent_images" not in st.session_state:
        st.session_state.only_n_most_recent_images = 2
    if "responses" not in st.session_state:
        st.session_state.responses = {}
    if "tools" not in st.session_state:
        st.session_state.tools = {}
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if "selected_file" not in st.session_state:
        st.session_state.selected_file = "None"
    if "stop" not in st.session_state:
        st.session_state.stop = False

def get_file_viewer_html(file_path=None, windows_host_url=None):
    """Generate HTML to view a file based on its type"""
    if not file_path:
        # Return the VNC viewer iframe
        return f'<iframe src="http://{windows_host_url}/vnc.html?view_only=1&autoconnect=1&resize=scale" width="100%" height="580" allow="fullscreen"></iframe>'
    
    file_path = Path(file_path)
    if not file_path.exists():
        return f'<div class="error-message">File not found: {file_path.name}</div>'
    
    mime_type, _ = mimetypes.guess_type(file_path)
    file_type = mime_type.split('/')[0] if mime_type else 'unknown'
    file_extension = file_path.suffix.lower()
    
    if file_type == 'image':
        with open(file_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
            return f'<div class="file-viewer"><h3>{file_path.name}</h3><img src="data:{mime_type};base64,{encoded_string}" style="max-width:100%; max-height:500px;"></div>'
    
    elif file_extension in ['.txt', '.py', '.js', '.html', '.css', '.json', '.md', '.csv'] or file_type == 'text':
        try:
            content = file_path.read_text(errors='replace')
            content = content.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            return f'<div class="file-viewer"><h3>{file_path.name}</h3><pre style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow: auto; max-height: 500px; white-space: pre-wrap;"><code>{content}</code></pre></div>'
        except UnicodeDecodeError:
            return f'<div class="error-message">Cannot display binary file: {file_path.name}</div>'
    
    else:
        size_kb = file_path.stat().st_size / 1024
        return f'<div class="file-viewer"><h3>{file_path.name}</h3><p>File type: {mime_type or "Unknown"}</p><p>Size: {size_kb:.2f} KB</p><p>This file type cannot be displayed in the browser.</p></div>'

def handle_file_upload(uploaded_files):
    """Handle file uploads and store them in the upload directory"""
    if uploaded_files:
        for file in uploaded_files:
            file_path = UPLOAD_FOLDER / file.name
            with open(file_path, "wb") as f:
                f.write(file.getvalue())
            if str(file_path) not in st.session_state.uploaded_files:
                st.session_state.uploaded_files.append(str(file_path))

def _api_response_callback(response: APIResponse[BetaMessage]):
    response_id = datetime.now().isoformat()
    st.session_state.responses[response_id] = response

def _tool_output_callback(tool_output: ToolResult, tool_id: str):
    st.session_state.tools[tool_id] = tool_output

def chatbot_output_callback(message, hide_images=False):
    def _render_message(message: str | BetaTextBlock | BetaToolUseBlock | ToolResult, hide_images=False):
        if isinstance(message, str):
            return message
        
        is_tool_result = not isinstance(message, str) and (
            isinstance(message, ToolResult)
            or message.__class__.__name__ == "ToolResult"
        )
        
        if is_tool_result:
            message = cast(ToolResult, message)
            if message.output:
                return message.output
            if message.error:
                return f"Error: {message.error}"
            if message.base64_image and not hide_images:
                return f'<img src="data:image/png;base64,{message.base64_image}">'
        
        elif isinstance(message, (BetaTextBlock, TextBlock)):
            return f"Next step Reasoning: {message.text}"
        
        elif isinstance(message, (BetaToolUseBlock, ToolUseBlock)):
            return None
        
        return message

    rendered_message = _render_message(message, hide_images)
    if rendered_message:
        st.session_state.messages.append({"role": "assistant", "content": rendered_message})

def main():
    args = parse_arguments()
    initialize_session_state()

    # Page configuration
    st.set_page_config(
        page_title="OmniTool",
        page_icon="🤖",
        layout="wide"
    )

    # Custom CSS
    st.markdown("""
        <style>
        .stApp {
            max-width: 100%;
            padding: 1rem;
        }
        .chat-container {
            height: calc(100vh - 200px);
            overflow-y: auto;
            position: relative;
        }
        .viewer-container {
            height: calc(100vh - 200px);
            overflow-y: auto;
        }
        .chat-input-container {
            display: flex;
            align-items: flex-end;
        }
        .icon-button {
            border: none;
            background: none;
            cursor: pointer;
            padding: 0.5rem;
            border-radius: 50%;
            transition: background-color 0.3s;
        }
        .icon-button:hover {
            background-color: #f0f0f0;
        }
        .stButton button {
            border-radius: 50%;
            width: 40px;
            height: 40px;
            padding: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        /* Custom button styles */
        .send-btn {
            background-color: #000 !important;
            color: white !important;
        }
        .stop-btn {
            background-color: #f8f9fa !important;
            color: #d9534f !important;
            border: 1px solid #d9534f !important;
        }
        .upload-btn {
            background-color: #f8f9fa !important;
            color: #0275d8 !important;
            border: 1px solid #0275d8 !important;
        }
        /* Hide the default button styling */
        div[data-testid="stHorizontalBlock"] button[kind="secondary"] {
            background-color: transparent;
            border: none;
        }
        /* Share button positioning */
        .share-button-container {
            position: absolute;
            top: 0;
            right: 0;
            z-index: 100;
        }
        /* Chat header with title and share button */
        .chat-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        /* Input placeholder styling */
        .stTextInput input::placeholder {
            color: #6c757d;
            font-style: italic;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header
    st.title("OmniTool")

    # Sidebar with settings
    with st.sidebar:
        st.header("Settings")
        
        # Model selection
        model = st.selectbox(
            "Model",
            ["omniparser + gpt-4o", "omniparser + o1", "omniparser + o3-mini", 
             "omniparser + R1", "omniparser + qwen2.5vl", "claude-3-5-sonnet-20241022",
             "omniparser + gpt-4o-orchestrated", "omniparser + o1-orchestrated",
             "omniparser + o3-mini-orchestrated", "omniparser + R1-orchestrated",
             "omniparser + qwen2.5vl-orchestrated"],
            index=6
        )
        st.session_state.model = model

        # API settings
        api_key = st.text_input("API Key", value=st.session_state.api_key, type="password")
        st.session_state.api_key = api_key

        # Image settings
        n_images = st.slider("N most recent screenshots", 0, 10, 2)
        st.session_state.only_n_most_recent_images = n_images

        # File viewer selection
        file_options = ["None"]
        if st.session_state.uploaded_files:
            file_options.extend([Path(f).name for f in st.session_state.uploaded_files])
        
        selected_file = st.selectbox(
            "View File",
            options=file_options,
            format_func=lambda x: x
        )
        st.session_state.selected_file = selected_file
        
        view_mode = st.radio("Display Mode", ["OmniTool Computer", "File Viewer"])

    # Main content area with two columns
    col1, col2 = st.columns([2, 3])

    # Chat interface (left column)
    with col1:
        # Chat header with title and share button
        col_header_1, col_header_2 = st.columns([3, 1])
        with col_header_1:
            st.markdown("### Chat")
        with col_header_2:
            share_button = st.button("📤 Share", key="share_btn", help="Share conversation")
            # Apply custom styling with HTML
            st.markdown("""
                <style>
                button[data-testid="share_btn"] {
                    background-color: #f8f9fa !important;
                    color: #0275d8 !important;
                    border: 1px solid #0275d8 !important;
                    border-radius: 4px !important;
                    width: auto !important;
                    height: auto !important;
                    padding: 2px 8px !important;
                    font-size: 0.8rem !important;
                }
                </style>
            """, unsafe_allow_html=True)
        
        # Share functionality
        if share_button:
            # Create a shareable text of the conversation
            conversation_text = ""
            for message in st.session_state.messages:
                if message["role"] == "user":
                    conversation_text += f"User: {message['content']}\n\n"
                else:
                    conversation_text += f"Assistant: {message['content']}\n\n"
            
            # Create a download link
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                label="Download Conversation",
                data=conversation_text,
                file_name=f"omnitool_conversation_{timestamp}.txt",
                mime="text/plain",
                key="download_conversation"
            )
        
        # Display chat messages
        chat_container = st.container(height=450)
        with chat_container:
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.markdown(f"**You:** {message['content']}")
                else:
                    st.markdown(f"**Assistant:** {message['content']}", unsafe_allow_html=True)

        # Chat input and buttons
        user_input = st.text_input(
            "Type your message:", 
            key="user_input", 
            label_visibility="collapsed",
            placeholder="Send message to OmniTool..."
        )
        
        # Button row with icons
        col1_1, col1_2, col1_3, col1_4 = st.columns([6, 1, 1, 1])
        
        with col1_2:
            # Send button with icon - using arrow up icon
            send_button = st.button("⬆️", help="Send message", key="send_btn")
            # Apply custom styling with HTML
            st.markdown("""
                <style>
                button[data-testid="send_btn"] {
                    background-color: black !important;
                    color: white !important;
                }
                </style>
            """, unsafe_allow_html=True)
        
        with col1_3:
            # Stop button with icon
            stop_button = st.button("🛑", help="Stop processing", key="stop_btn")
            # Apply custom styling with HTML
            st.markdown("""
                <style>
                button[data-testid="stop_btn"] {
                    background-color: #f8f9fa !important;
                    color: #d9534f !important;
                    border: 1px solid #d9534f !important;
                }
                </style>
            """, unsafe_allow_html=True)
            
        with col1_4:
            # File upload button with icon
            upload_button = st.button("📎", help="Upload files", key="upload_btn")
            # Apply custom styling with HTML
            st.markdown("""
                <style>
                button[data-testid="upload_btn"] {
                    background-color: #f8f9fa !important;
                    color: #0275d8 !important;
                    border: 1px solid #0275d8 !important;
                }
                </style>
            """, unsafe_allow_html=True)
        
        # File upload area (hidden by default, shown when upload button is clicked)
        if upload_button:
            uploaded_files = st.file_uploader("Upload Files", accept_multiple_files=True, label_visibility="collapsed")
            if uploaded_files:
                handle_file_upload(uploaded_files)
                st.success(f"Uploaded {len(uploaded_files)} file(s)")
                # Update file options
                file_options = ["None"]
                if st.session_state.uploaded_files:
                    file_options.extend([Path(f).name for f in st.session_state.uploaded_files])
                st.rerun()
        
        # Process send button click
        if send_button and user_input:
            # Add user message to state
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Process the message through sampling_loop_sync
            for loop_msg in sampling_loop_sync(
                model=st.session_state.model,
                provider=st.session_state.provider,
                messages=[{"role": "user", "content": [TextBlock(type="text", text=msg["content"])]} for msg in st.session_state.messages],
                output_callback=chatbot_output_callback,
                tool_output_callback=_tool_output_callback,
                api_response_callback=_api_response_callback,
                api_key=st.session_state.api_key,
                only_n_most_recent_images=st.session_state.only_n_most_recent_images,
                max_tokens=16384,
                omniparser_url=args.omniparser_server_url,
                save_folder=str(UPLOAD_FOLDER)
            ):
                if loop_msg is None or st.session_state.stop:
                    break
                st.rerun()
        
        # Process stop button click
        if stop_button:
            st.session_state.stop = True
            st.info("Processing stopped")

    # Viewer interface (right column)
    with col2:
        st.markdown("### Display")
        if view_mode == "OmniTool Computer":
            viewer_html = get_file_viewer_html(windows_host_url=args.windows_host_url)
            st.components.v1.html(
                viewer_html,
                height=600,
                scrolling=True
            )
        else:  # File Viewer mode
            if st.session_state.selected_file and st.session_state.selected_file != "None":
                file_path = next((f for f in st.session_state.uploaded_files 
                                if Path(f).name == st.session_state.selected_file), None)
                if file_path:
                    viewer_html = get_file_viewer_html(file_path=file_path)
                    st.components.v1.html(
                        viewer_html,
                        height=600,
                        scrolling=True
                    )
                else:
                    st.error(f"Could not find file: {st.session_state.selected_file}")
            else:
                st.info("Please select a file to view from the sidebar.")

        # Debug information (temporary)
        with st.expander("Debug Info"):
            st.write("View Mode:", view_mode)
            st.write("Selected File:", st.session_state.selected_file)
            st.write("Available Files:", st.session_state.uploaded_files)
            if view_mode == "File Viewer" and st.session_state.selected_file != "None":
                st.write("File Path:", file_path if 'file_path' in locals() else "Not found")

if __name__ == "__main__":
    main()
