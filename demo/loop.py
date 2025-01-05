"""
Agentic sampling loop that calls the Anthropic API and local implenmentation of anthropic-defined computer use tools.
"""
import time
import json
import asyncio
import platform
from collections.abc import Callable
from datetime import datetime
from enum import StrEnum
from typing import Any, cast, Dict

from anthropic import Anthropic, AnthropicBedrock, AnthropicVertex, APIResponse
from anthropic.types import (
    ToolResultBlockParam,
    TextBlock,
)
from anthropic.types.beta import (
    BetaContentBlock,
    BetaContentBlockParam,
    BetaImageBlockParam,
    BetaMessage,
    BetaMessageParam,
    BetaTextBlockParam,
    BetaToolResultBlockParam,
)
from tools import BashTool, ComputerTool, EditTool, ToolCollection, ToolResult

import torch

from gui_agent.anthropic_agent import AnthropicActor
from executor.anthropic_executor import AnthropicExecutor
from omniparser_agent.vlm_agent import OmniParser, VLMAgent
from tools.colorful_text import colorful_text_showui, colorful_text_vlm
from tools.screen_capture import get_screenshot
from gui_agent.llm_utils.oai import encode_image


BETA_FLAG = "computer-use-2024-10-22"


class APIProvider(StrEnum):
    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"
    VERTEX = "vertex"
    OPENAI = "openai"
    QWEN = "qwen"


PROVIDER_TO_DEFAULT_MODEL_NAME: dict[APIProvider, str] = {
    APIProvider.ANTHROPIC: "claude-3-5-sonnet-20241022",
    APIProvider.BEDROCK: "anthropic.claude-3-5-sonnet-20241022-v2:0",
    APIProvider.VERTEX: "claude-3-5-sonnet-v2@20241022",
    # APIProvider.OPENAI: "gpt-4o",
    # APIProvider.QWEN: "qwen2vl",
}


# This system prompt is optimized for the Docker environment in this repository and
# specific tool combinations enabled.
# We encourage modifying this system prompt to ensure the model has context for the
# environment it is running in, and to provide any additional information that may be
# helpful for the task at hand.
SYSTEM_PROMPT = f"""<SYSTEM_CAPABILITY>
* You are utilizing a Windows system with internet access.
* The current date is {datetime.today().strftime('%A, %B %d, %Y')}.
</SYSTEM_CAPABILITY>
"""

import base64
from PIL import Image
from io import BytesIO

def sampling_loop_sync(
    *,
    model: str,
    provider: APIProvider | None,
    system_prompt_suffix: str,
    messages: list[BetaMessageParam],
    output_callback: Callable[[BetaContentBlock], None],
    tool_output_callback: Callable[[ToolResult, str], None],
    api_response_callback: Callable[[APIResponse[BetaMessage]], None],
    api_key: str,
    only_n_most_recent_images: int | None = 2,
    max_tokens: int = 4096,
    selected_screen: int = 0
):
    """
    Synchronous agentic sampling loop for the assistant/tool interaction of computer use.
    """
    print('in sampling_loop_sync, model:', model)
    if model == "claude-3-5-sonnet-20241022":
        omniparser = OmniParser(url="http://127.0.0.1:8000/send_text/",
                                selected_screen=selected_screen,)
        
        # Register Actor and Executor
        actor = AnthropicActor(
            model=model, 
            provider=provider, 
            system_prompt_suffix=system_prompt_suffix, 
            api_key=api_key, 
            api_response_callback=api_response_callback,
            max_tokens=max_tokens,
            only_n_most_recent_images=only_n_most_recent_images,
            selected_screen=selected_screen
        )

        # from IPython.core.debugger import Pdb; Pdb().set_trace()
        executor = AnthropicExecutor(
            output_callback=output_callback,
            tool_output_callback=tool_output_callback,
            selected_screen=selected_screen
        )
    
    elif model == "omniparser + gpt-4o" or model == "omniparser + phi35v":
        omniparser = OmniParser(url="http://127.0.0.1:8000/send_text/",
                                selected_screen=selected_screen,)

        actor = VLMAgent(
            model=model,
            provider=provider,
            system_prompt_suffix=system_prompt_suffix,
            api_key=api_key,
            api_response_callback=api_response_callback,
            selected_screen=selected_screen,
            output_callback=output_callback,
        )

        executor = AnthropicExecutor(
            output_callback=output_callback,
            tool_output_callback=tool_output_callback,
            selected_screen=selected_screen
        )

    # elif model == "gpt-4o + ShowUI" or model == "qwen2vl + ShowUI":
    #     planner = VLMPlanner(
    #         model=model,
    #         provider=provider,
    #         system_prompt_suffix=system_prompt_suffix,
    #         api_key=api_key,
    #         api_response_callback=api_response_callback,
    #         selected_screen=selected_screen,
    #         output_callback=output_callback,
    #     )
        
    #     if torch.cuda.is_available(): device = torch.device("cuda")
    #     elif torch.backends.mps.is_available(): device = torch.device("mps")
    #     else: device = torch.device("cpu") # support: 'cpu', 'mps', 'cuda'
    #     print(f"showUI-2B inited on device: {device}.")
        
    #     actor = ShowUIActor(
    #         model_path="./showui-2b/",  
    #         # Replace with your local path, e.g., "C:\\code\\ShowUI-2B", "/Users/your_username/ShowUI-2B/".
    #         device=device,  
    #         split='web',  # 'web' or 'phone'
    #         selected_screen=selected_screen,
    #         output_callback=output_callback,
    #     )
        
    #     executor = ShowUIExecutor(
    #         output_callback=output_callback,
    #         tool_output_callback=tool_output_callback,
    #         selected_screen=selected_screen
    #     )
        
    else:
        raise ValueError(f"Model {model} not supported")
    print(f"Model Inited: {model}, Provider: {provider}")
    
    tool_result_content = None
    
    print(f"Start the message loop. User messages: {messages}")
    
    if model == "claude-3-5-sonnet-20241022": # Anthropic loop
        while True:
            parsed_screen = omniparser() # parsed_screen: {"som_image_base64": dino_labled_img, "parsed_content_list": parsed_content_list, "screen_info"}
            import pdb; pdb.set_trace()
            screen_info_block = TextBlock(text='Below is the structured accessibility information of the current UI screen, which includes text and icons you can operate on, take these information into account when you are making the prediction for the next action. Note you will still need to take screenshot to get the image: \n' + parsed_screen['screen_info'], type='text')
            # # messages[-1]['content'].append(screen_info_block)
            screen_info_dict = {"role": "user", "content": [screen_info_block]}
            messages.append(screen_info_dict)
            response = actor(messages=messages)

            for message, tool_result_content in executor(response, messages):
                yield message
        
            if not tool_result_content:
                return messages

            messages.append({"content": tool_result_content, "role": "user"})
    
    elif model == "omniparser + gpt-4o" or model == "omniparser + phi35v":
        while True:
            parsed_screen = omniparser()
            response, vlm_response_json = actor(messages=messages, parsed_screen=parsed_screen)

            for message, tool_result_content in executor(response, messages):
                yield message
        
            if not tool_result_content:
                return messages
            
            # import pdb; pdb.set_trace()
            # messages.append({"role": "user",
            #                  "content": ["History plan:\n" + str(vlm_response_json['Reasoning'])]})

            # messages.append({"content": tool_result_content, "role": "user"})

    elif model == "gpt-4o + ShowUI" or model == "qwen2vl + ShowUI":  # ShowUI loop 
        while True:
            vlm_response = planner(messages=messages)
            
            next_action = json.loads(vlm_response).get("Next Action")
            yield next_action
            
            if next_action == None or next_action == "" or next_action == "None":
                final_sc, final_sc_path = get_screenshot(selected_screen=selected_screen)
                output_callback(f'No more actions from {colorful_text_vlm}. End of task. Final State:\n<img src="data:image/png;base64,{encode_image(str(final_sc_path))}">',
                                sender="bot")
                yield None
                        
            output_callback(f"{colorful_text_vlm} sending action to {colorful_text_showui}:\n{next_action}", sender="bot")
            
            actor_response = actor(messages=next_action)
            yield actor_response
            
            for message, tool_result_content in executor(actor_response, messages):
                time.sleep(1)
                yield message
                
            # since showui executor has no feedback for now, we use "actor_response" to represent its response
            # update messages for the next loop
            messages.append({"role": "user",
                             "content": ["History plan:\n" + str(json.loads(vlm_response)) + 
                                        "\nHistory actions:\n" + str(actor_response["content"])]})
            print(f"End of loop. Messages: {str(messages)[:100000]}. Total cost: $USD{planner.total_cost:.5f}")