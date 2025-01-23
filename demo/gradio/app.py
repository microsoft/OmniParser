"""
Entrypoint for Gradio, see https://gradio.app/
"""

import platform
import asyncio
import base64
import os
import io
import json
from datetime import datetime
from enum import StrEnum
from functools import partial
from pathlib import Path
from typing import cast, Dict
from PIL import Image
import socket

import gradio as gr
from anthropic import APIResponse
from anthropic.types import TextBlock
from anthropic.types.beta import BetaMessage, BetaTextBlock, BetaToolUseBlock
from anthropic.types.tool_use_block import ToolUseBlock
from computer_use_demo.loop import (
    PROVIDER_TO_DEFAULT_MODEL_NAME,
    APIProvider,
    sampling_loop_sync,
)

from computer_use_demo.tools import ToolResult

CONFIG_DIR = Path("~/.anthropic").expanduser()
API_KEY_FILE = CONFIG_DIR / "api_key"

INTRO_TEXT = '''
🚀🤖✨ It's Play Time!

Welcome to the OmniParser+X Demo! X = [GPT-4o/4o-mini, Claude, Phi, Llama]. Let OmniParser turn your general purpose vision-langauge model to an AI agent. Type a message to play with your beloved assistant.
'''

class Sender(StrEnum):
    USER = "user"
    BOT = "assistant"
    TOOL = "tool"


def setup_state(state):
    if "messages" not in state:
        state["messages"] = []
    if "model" not in state:
        state["model"] = "omniparser + gpt-4o"
    if "provider" not in state:
        state["provider"] = "openai"
    if "openai_api_key" not in state:  # Fetch API keys from environment variables
        state["openai_api_key"] = os.getenv("OPENAI_API_KEY", "")
    if "anthropic_api_key" not in state:
        state["anthropic_api_key"] = os.getenv("ANTHROPIC_API_KEY", "")    
    if "qwen_api_key" not in state:
        state["qwen_api_key"] = os.getenv("QWEN_API_KEY", "")
    if "api_key" not in state:
        state["api_key"] = ""
    if "auth_validated" not in state:
        state["auth_validated"] = False
    if "responses" not in state:
        state["responses"] = {}
    if "tools" not in state:
        state["tools"] = {}
    if "only_n_most_recent_images" not in state:
        state["only_n_most_recent_images"] = 2
    if 'chatbot_messages' not in state:
        state['chatbot_messages'] = []

async def main(state):
    """Render loop for Gradio"""
    setup_state(state)
    return "Setup completed"

def validate_auth(provider: APIProvider, api_key: str | None):
    if provider == APIProvider.ANTHROPIC:
        if not api_key:
            return "Enter your Anthropic API key to continue."
    if provider == APIProvider.BEDROCK:
        import boto3

        if not boto3.Session().get_credentials():
            return "You must have AWS credentials set up to use the Bedrock API."
    if provider == APIProvider.VERTEX:
        import google.auth
        from google.auth.exceptions import DefaultCredentialsError

        if not os.environ.get("CLOUD_ML_REGION"):
            return "Set the CLOUD_ML_REGION environment variable to use the Vertex API."
        try:
            google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        except DefaultCredentialsError:
            return "Your google cloud credentials are not set up correctly."

def load_from_storage(filename: str) -> str | None:
    """Load data from a file in the storage directory."""
    try:
        file_path = CONFIG_DIR / filename
        if file_path.exists():
            data = file_path.read_text().strip()
            if data:
                return data
    except Exception as e:
        print(f"Debug: Error loading {filename}: {e}")
    return None

def save_to_storage(filename: str, data: str) -> None:
    """Save data to a file in the storage directory."""
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        file_path = CONFIG_DIR / filename
        file_path.write_text(data)
        # Ensure only user can read/write the file
        file_path.chmod(0o600)
    except Exception as e:
        print(f"Debug: Error saving {filename}: {e}")

def _api_response_callback(response: APIResponse[BetaMessage], response_state: dict):
    response_id = datetime.now().isoformat()
    response_state[response_id] = response

def _tool_output_callback(tool_output: ToolResult, tool_id: str, tool_state: dict):
    tool_state[tool_id] = tool_output

def chatbot_output_callback(message, chatbot_state, hide_images=False, sender="bot"):
    def _render_message(message: str | BetaTextBlock | BetaToolUseBlock | ToolResult, hide_images=False):
    
        print(f"_render_message: {str(message)[:100]}")
        
        if isinstance(message, str):
            return message
        
        is_tool_result = not isinstance(message, str) and (
            isinstance(message, ToolResult)
            or message.__class__.__name__ == "ToolResult"
            or message.__class__.__name__ == "CLIResult"
        )
        if not message or (
            is_tool_result
            and hide_images
            and not hasattr(message, "error")
            and not hasattr(message, "output")
        ):  # return None if hide_images is True
            return
        # render tool result
        if is_tool_result:
            message = cast(ToolResult, message)
            if message.output:
                return message.output
            if message.error:
                return f"Error: {message.error}"
            if message.base64_image and not hide_images:
                # somehow can't display via gr.Image
                # image_data = base64.b64decode(message.base64_image)
                # return gr.Image(value=Image.open(io.BytesIO(image_data)))
                return f'<img src="data:image/png;base64,{message.base64_image}">'

        elif isinstance(message, BetaTextBlock) or isinstance(message, TextBlock):
            return f"Analysis: {message.text}"
        elif isinstance(message, BetaToolUseBlock) or isinstance(message, ToolUseBlock):
            # return f"Tool Use: {message.name}\nInput: {message.input}"
            return f"Next I will perform the following action: {message.input}"
        else:  
            return message

    def _truncate_string(s, max_length=500):
        """Truncate long strings for concise printing."""
        if isinstance(s, str) and len(s) > max_length:
            return s[:max_length] + "..."
        return s
    # processing Anthropic messages
    message = _render_message(message, hide_images)
    
    if sender == "bot":
        chatbot_state.append((None, message))
    else:
        chatbot_state.append((message, None))
    
    # Create a concise version of the chatbot state for printing
    concise_state = [(_truncate_string(user_msg), _truncate_string(bot_msg))
                        for user_msg, bot_msg in chatbot_state]
    # print(f"chatbot_output_callback chatbot_state: {concise_state} (truncated)")

def process_input(user_input, state):
    # Append the user message to state["messages"]
    state["messages"].append(
        {
            "role": Sender.USER,
            "content": [TextBlock(type="text", text=user_input)],
        }
    )

    # Append the user's message to chatbot_messages with None for the assistant's reply
    state['chatbot_messages'].append((user_input, None))
    yield state['chatbot_messages']  # Yield to update the chatbot UI with the user's message

    print("state")
    print(state)

    # Run sampling_loop_sync with the chatbot_output_callback
    for loop_msg in sampling_loop_sync(
        model=state["model"],
        provider=state["provider"],
        messages=state["messages"],
        output_callback=partial(chatbot_output_callback, chatbot_state=state['chatbot_messages'], hide_images=False),
        tool_output_callback=partial(_tool_output_callback, tool_state=state["tools"]),
        api_response_callback=partial(_api_response_callback, response_state=state["responses"]),
        api_key=state["api_key"],
        only_n_most_recent_images=state["only_n_most_recent_images"],
        selected_screen=0
    ):  
        if loop_msg is None:
            yield state['chatbot_messages']
            print("End of task. Close the loop.")
            break
            
        yield state['chatbot_messages']  # Yield the updated chatbot_messages to update the chatbot UI

with gr.Blocks(theme=gr.themes.Default()) as demo:
    gr.HTML("""
        <style>
        .no-padding {
            padding: 0 !important;
        }
        .no-padding > div {
            padding: 0 !important;
        }
        </style>
    """)
    state = gr.State({})  # Use Gradio's state management

    setup_state(state.value)  # Initialize the state

    # Retrieve screen details
    gr.Markdown("# OmniParser + ✖️ Demo")

    if not os.getenv("HIDE_WARNING", False):
        gr.Markdown(INTRO_TEXT)

    with gr.Accordion("Settings", open=True): 
        with gr.Row():
            with gr.Column():
                model = gr.Dropdown(
                    label="Model",
                    choices=["omniparser + gpt-4o", "omniparser + phi35v", "claude-3-5-sonnet-20241022"],
                    value="omniparser + gpt-4o",  # Set to one of the choices
                    interactive=True,
                )
            with gr.Column():
                only_n_images = gr.Slider(
                    label="N most recent screenshots",
                    minimum=0,
                    maximum=10,
                    step=1,
                    value=2,
                    interactive=True
                )
        with gr.Row():
            with gr.Column(1):
                provider = gr.Dropdown(
                    label="API Provider",
                    choices=[option.value for option in APIProvider],
                    value="openai",
                    interactive=False,
                )
            with gr.Column(2):
                api_key = gr.Textbox(
                    label="API Key",
                    type="password",
                    value=state.value.get("api_key", ""),
                    placeholder="Paste your API key here",
                    interactive=True,
                )
        # hide_images = gr.Checkbox(label="Hide screenshots", value=False)

    with gr.Row():
        with gr.Column(scale=8):
            chat_input = gr.Textbox(show_label=False, placeholder="Type a message to send to Computer Use OOTB...", container=False)
        with gr.Column(scale=1, min_width=50):
            submit_button = gr.Button(value="Send", variant="primary")

    with gr.Row():
        with gr.Column(scale=1):
            chatbot = gr.Chatbot(label="Chatbot History", autoscroll=True, height=580)
        with gr.Column(scale=3):
            # Get the fully qualified domain name of the machine
            machine_fqdn = socket.getfqdn()
            iframe = gr.HTML(
                f'<iframe src="http://{machine_fqdn}:8006/vnc.html?view_only=1&autoconnect=1&resize=scale" width="100%" height="580" allow="fullscreen"></iframe>',
                container=False,
                elem_classes="no-padding"
            )

    def update_model(model_selection, state):
        state["model"] = model_selection
        print(f"Model updated to: {state['model']}")
        
        if model_selection == "claude-3-5-sonnet-20241022":
            provider_choices = [option.value for option in APIProvider if option.value != "openai"]
        elif model_selection == "omniparser + gpt-4o" or model_selection == "omniparser + phi35v":
            provider_choices = ["openai"]
        else:
            provider_choices = [option.value for option in APIProvider]
        default_provider_value = provider_choices[0]
        provider_interactive = len(provider_choices) > 1
        api_key_placeholder = f"{default_provider_value.title()} API Key"

        # Update state
        state["provider"] = default_provider_value
        state["api_key"] = state.get(f"{default_provider_value}_api_key", "")

        # Calls to update other components UI
        provider_update = gr.update(
            choices=provider_choices,
            value=default_provider_value,
            interactive=provider_interactive
        )
        api_key_update = gr.update(
            placeholder=api_key_placeholder,
            value=state["api_key"]
        )

        return provider_update, api_key_update

    def update_only_n_images(only_n_images_value, state):
        state["only_n_most_recent_images"] = only_n_images_value
   
    def update_provider(provider_value, state):
        # Update state
        state["provider"] = provider_value
        state["api_key"] = state.get(f"{provider_value}_api_key", "")
        
        # Calls to update other components UI
        api_key_update = gr.update(
            placeholder=f"{provider_value.title()} API Key",
            value=state["api_key"]
        )
        return api_key_update
                
    def update_api_key(api_key_value, state):
        state["api_key"] = api_key_value
        state[f'{state["provider"]}_api_key'] = api_key_value

    model.change(fn=update_model, inputs=[model, state], outputs=[provider, api_key])
    only_n_images.change(fn=update_only_n_images, inputs=[only_n_images, state], outputs=None)
    provider.change(fn=update_provider, inputs=[provider, state], outputs=api_key)
    api_key.change(fn=update_api_key, inputs=[api_key, state], outputs=None)

    submit_button.click(process_input, [chat_input, state], chatbot)

from fastapi import FastAPI
import uvicorn
from multiprocessing import Process

app = FastAPI()

# Mount the Gradio app under the "/gradio" path
app = gr.mount_gradio_app(app, demo, path="/gradio")

# Optional: Add a root endpoint that redirects to the Gradio interface
@app.get("/")
async def root():
    return {"message": "Welcome to OmniParser Demo API", 
            "gradio_interface": "/gradio"}

# Create a second FastAPI app for VNC
vnc_app = FastAPI()

@vnc_app.get("/")
async def vnc_root():
    return {"message": "VNC Server"}

def run_app(app, host, port):
    uvicorn.run(app, host=host, port=port)

# To run this with uvicorn:
if __name__ == "__main__":
    # Start the main app on port 7889
    p1 = Process(target=run_app, args=(app, "0.0.0.0", 7889))
    # Start the VNC app on port 8006
    p2 = Process(target=run_app, args=(vnc_app, "0.0.0.0", 8006))
    
    p1.start()
    p2.start()
    
    try:
        p1.join()
        p2.join()
    except KeyboardInterrupt:
        p1.terminate()
        p2.terminate()