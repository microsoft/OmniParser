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

import gradio as gr
from anthropic import APIResponse
from anthropic.types import TextBlock
from anthropic.types.beta import BetaMessage, BetaTextBlock, BetaToolUseBlock
from anthropic.types.tool_use_block import ToolUseBlock

from screeninfo import get_monitors

screens = get_monitors()
print(screens)
from loop import (
    PROVIDER_TO_DEFAULT_MODEL_NAME,
    APIProvider,
    sampling_loop_sync,
)

from tools import ToolResult
from tools.computer import get_screen_details
SCREEN_NAMES, SELECTED_SCREEN_INDEX = get_screen_details()
# SELECTED_SCREEN_INDEX = None
# SCREEN_NAMES = None

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
        # state["model"] = "gpt-4o + ShowUI"
        state["model"] = "omniparser + gpt-4o"
        # _reset_model(state)
    if "provider" not in state:
        if state["model"] == "qwen2vl + ShowUI":
            state["provider"] = "DashScopeAPI"
        elif state["model"] == "gpt-4o + ShowUI":
            state["provider"] = "openai"
        else:
            state["provider"] = os.getenv("API_PROVIDER", "anthropic") or "anthropic"

    if "provider_radio" not in state:
        state["provider_radio"] = state["provider"]
    
    if "openai_api_key" not in state:  # Fetch API keys from environment variables
        state["openai_api_key"] = os.getenv("OPENAI_API_KEY", "")
    if "anthropic_api_key" not in state:
        state["anthropic_api_key"] = os.getenv("ANTHROPIC_API_KEY", "")    
    if "qwen_api_key" not in state:
        state["qwen_api_key"] = os.getenv("QWEN_API_KEY", "")
    
    # Set the initial api_key based on the provider
    if "api_key" not in state:
        if state["provider"] == "openai":
            state["api_key"] = state["openai_api_key"]
        elif state["provider"] == "anthropic":
            state["api_key"] = state["anthropic_api_key"]
        elif state["provider"] == "qwen":
            state["api_key"] = state["qwen_api_key"]
        else:
            state["api_key"] = ""
    # print(f"state['api_key']: {state['api_key']}")
    if not state["api_key"]:
        print("API key not found. Please set it in the environment or paste in textbox.")

    if "selected_screen" not in state:
        state['selected_screen'] = SELECTED_SCREEN_INDEX if SCREEN_NAMES else 0

    if "auth_validated" not in state:
        state["auth_validated"] = False
    if "responses" not in state:
        state["responses"] = {}
    if "tools" not in state:
        state["tools"] = {}
    if "only_n_most_recent_images" not in state:
        state["only_n_most_recent_images"] = 10 # 10
    if "custom_system_prompt" not in state:
        state["custom_system_prompt"] = load_from_storage("system_prompt") or ""
        # remove if want to use default system prompt
        device_os_name = "Windows" if platform.system() == "Windows" else "Mac" if platform.system() == "Darwin" else "Linux"
        state["custom_system_prompt"] += f"\n\nNOTE: you are operating a {device_os_name} machine"
    if "hide_images" not in state:
        state["hide_images"] = False
    if 'chatbot_messages' not in state:
        state['chatbot_messages'] = []
    


def _reset_model(state):
    state["model"] = PROVIDER_TO_DEFAULT_MODEL_NAME[cast(APIProvider, state["provider"])]


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
    
    setup_state(state)

    # Append the user message to state["messages"]
    if state["model"] == "gpt-4o + ShowUI" or state["model"] == "qwen2vl + ShowUI":
        state["messages"].append(
            {
                "role": "user",
                "content": [TextBlock(type="text", text=user_input)],
            }
        )
    elif state["model"] == "claude-3-5-sonnet-20241022":
        state["messages"].append(
            {
                "role": Sender.USER,
                "content": [TextBlock(type="text", text=user_input)],
            }
        )
    elif state["model"] == "omniparser + gpt-4o" or state["model"] == "omniparser + phi35v":
        state["messages"].append(
            {
                "role": "user",
                "content": [TextBlock(type="text", text=user_input)],
            }
        )

    # Append the user's message to chatbot_messages with None for the assistant's reply
    state['chatbot_messages'].append((user_input, None))
    yield state['chatbot_messages']  # Yield to update the chatbot UI with the user's message

    # Run sampling_loop_sync with the chatbot_output_callback
    for loop_msg in sampling_loop_sync(
        system_prompt_suffix=state["custom_system_prompt"],
        model=state["model"],
        provider=state["provider"],
        messages=state["messages"],
        output_callback=partial(chatbot_output_callback, chatbot_state=state['chatbot_messages'], hide_images=state["hide_images"]),
        tool_output_callback=partial(_tool_output_callback, tool_state=state["tools"]),
        api_response_callback=partial(_api_response_callback, response_state=state["responses"]),
        api_key=state["api_key"],
        only_n_most_recent_images=state["only_n_most_recent_images"],
        selected_screen=state['selected_screen']
    ):  
        if loop_msg is None:
            yield state['chatbot_messages']
            print("End of task. Close the loop.")
            break
            
        yield state['chatbot_messages']  # Yield the updated chatbot_messages to update the chatbot UI


# with gr.Blocks(theme=gr.themes.Default()) as demo:
with gr.Blocks(theme='YTheme/Minecraft') as demo:
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
                provider = gr.Dropdown(
                    label="API Provider",
                    choices=[option.value for option in APIProvider],
                    value="openai",
                    interactive=False,
                )
            with gr.Column():
                api_key = gr.Textbox(
                    label="API Key",
                    type="password",
                    value=state.value.get("api_key", ""),
                    placeholder="Paste your API key here",
                    interactive=True,
                )
            with gr.Column():
                custom_prompt = gr.Textbox(
                    label="System Prompt Suffix",
                    value="",
                    interactive=True,
                )
            with gr.Column():
                screen_options, primary_index = get_screen_details()
                SCREEN_NAMES = screen_options
                SELECTED_SCREEN_INDEX = primary_index
                screen_selector = gr.Dropdown(
                    label="Select Screen",
                    choices=screen_options,
                    value=screen_options[primary_index] if screen_options else None,
                    interactive=True,
                )
            with gr.Column():
                only_n_images = gr.Slider(
                    label="N most recent screenshots",
                    minimum=0,
                    maximum=10,
                    step=1,
                    value=2,
                    interactive=True,
                )
        # hide_images = gr.Checkbox(label="Hide screenshots", value=False)

    # Define the merged dictionary with task mappings
    # merged_dict = json.load(open("examples/ootb_examples.json", "r"))
    merged_dict = {}

    def update_only_n_images(only_n_images_value, state):
        state["only_n_most_recent_images"] = only_n_images_value
    
    # Callback to update the second dropdown based on the first selection
    def update_second_menu(selected_category):
        return gr.update(choices=list(merged_dict.get(selected_category, {}).keys()))

    # Callback to update the third dropdown based on the second selection
    def update_third_menu(selected_category, selected_option):
        return gr.update(choices=list(merged_dict.get(selected_category, {}).get(selected_option, {}).keys()))

    # Callback to update the textbox based on the third selection
    def update_textbox(selected_category, selected_option, selected_task):
        task_data = merged_dict.get(selected_category, {}).get(selected_option, {}).get(selected_task, {})
        prompt = task_data.get("prompt", "")
        preview_image = task_data.get("initial_state", "")
        task_hint = "Task Hint: " + task_data.get("hint", "")
        return prompt, preview_image, task_hint
    
    # Function to update the global variable when the dropdown changes
    def update_selected_screen(selected_screen_name, state):
        global SCREEN_NAMES
        global SELECTED_SCREEN_INDEX
        SELECTED_SCREEN_INDEX = SCREEN_NAMES.index(selected_screen_name)
        print(f"Selected screen updated to: {SELECTED_SCREEN_INDEX}")
        state['selected_screen'] = SELECTED_SCREEN_INDEX

    def update_model(model_selection, state):
        state["model"] = model_selection
        print(f"Model updated to: {state['model']}")
        
        if model_selection == "claude-3-5-sonnet-20241022":
            # Provider can be any of the current choices except 'openai'
            provider_choices = [option.value for option in APIProvider if option.value != "openai"]
            provider_value = "anthropic"  # Set default to 'anthropic'
            provider_interactive = True
            api_key_placeholder = "claude API key"
        elif model_selection == "omniparser + gpt-4o" or model_selection == "omniparser + phi35v":
            # Provider can be any of the current choices except 'openai'
            provider_choices = ["openai"]
            provider_value = "openai"
            provider_interactive = False
            api_key_placeholder = "openai API key"
        else:
            # Default case
            provider_choices = [option.value for option in APIProvider]
            provider_value = state.get("provider", provider_choices[0])
            provider_interactive = True
            api_key_placeholder = ""

        # Update the provider in state
        state["provider"] = provider_value
        
        # Update api_key in state based on the provider
        if provider_value == "openai":
            state["api_key"] = state.get("openai_api_key", "")
        elif provider_value == "anthropic":
            state["api_key"] = state.get("anthropic_api_key", "")
        elif provider_value == "qwen":
            state["api_key"] = state.get("qwen_api_key", "")
        else:
            state["api_key"] = ""

        # Use gr.update() instead of gr.Dropdown.update()
        provider_update = gr.update(
            choices=provider_choices,
            value=provider_value,
            interactive=provider_interactive
        )

        # Update the API Key textbox
        api_key_update = gr.update(
            placeholder=api_key_placeholder,
            value=state["api_key"]
        )

        return provider_update, api_key_update
    
    def update_api_key_placeholder(provider_value, model_selection):
        if model_selection == "claude-3-5-sonnet-20241022":
            if provider_value == "anthropic":
                return gr.update(placeholder="anthropic API key")
            elif provider_value == "bedrock":
                return gr.update(placeholder="bedrock API key")
            elif provider_value == "vertex":
                return gr.update(placeholder="vertex API key")
            else:
                return gr.update(placeholder="")
        elif model_selection == "gpt-4o + ShowUI":
            return gr.update(placeholder="openai API key")
        else:
            return gr.update(placeholder="")
        
    def update_system_prompt_suffix(system_prompt_suffix, state):
        state["custom_system_prompt"] = system_prompt_suffix


    api_key.change(fn=lambda key: save_to_storage(API_KEY_FILE, key), inputs=api_key)

    with gr.Row():
        # submit_button = gr.Button("Submit")  # Add submit button
        with gr.Column(scale=8):
            chat_input = gr.Textbox(show_label=False, placeholder="Type a message to send to Computer Use OOTB...", container=False)
        with gr.Column(scale=1, min_width=50):
            submit_button = gr.Button(value="Send", variant="primary")

    chatbot = gr.Chatbot(label="Chatbot History", autoscroll=True, height=580)
    
    model.change(fn=update_model, inputs=[model, state], outputs=[provider, api_key])
    provider.change(fn=update_api_key_placeholder, inputs=[provider, model], outputs=api_key)
    screen_selector.change(fn=update_selected_screen, inputs=[screen_selector, state], outputs=None)
    only_n_images.change(fn=update_only_n_images, inputs=[only_n_images, state], outputs=None)
    

    # chat_input.submit(process_input, [chat_input, state], chatbot)
    submit_button.click(process_input, [chat_input, state], chatbot)

demo.launch(share=True, server_port=7861, server_name='0.0.0.0')  # TODO: allowed_paths
