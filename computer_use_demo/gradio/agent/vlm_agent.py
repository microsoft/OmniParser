import json
from collections.abc import Callable
from typing import cast, Callable
import uuid
from PIL import Image, ImageDraw
import base64
from io import BytesIO

from anthropic import APIResponse
from anthropic.types import ToolResultBlockParam
from anthropic.types.beta import BetaMessage, BetaTextBlock, BetaToolUseBlock, BetaMessageParam, BetaUsage

from agent.llm_utils.oai import run_oai_interleaved
from agent.llm_utils.groqclient import run_groq_interleaved
import time
import re

OUTPUT_DIR = "./tmp/outputs"

def extract_data(input_string, data_type):
    # Regular expression to extract content starting from '```python' until the end if there are no closing backticks
    pattern = f"```{data_type}" + r"(.*?)(```|$)"
    # Extract content
    # re.DOTALL allows '.' to match newlines as well
    matches = re.findall(pattern, input_string, re.DOTALL)
    # Return the first match if exists, trimming whitespace and ignoring potential closing backticks
    return matches[0][0].strip() if matches else input_string

class VLMAgent:
    def __init__(
        self,
        model: str, 
        provider: str, 
        api_key: str,
        output_callback: Callable, 
        api_response_callback: Callable,
        max_tokens: int = 4096,
        only_n_most_recent_images: int | None = None,
        print_usage: bool = True,
    ):
        if model == "omniparser + gpt-4o":
            self.model = "gpt-4o-2024-11-20"
        elif model == "omniparser + R1":
            self.model = "deepseek-r1-distill-llama-70b"
        else:
            raise ValueError(f"Model {model} not supported")
        

        self.provider = provider
        self.api_key = api_key
        self.api_response_callback = api_response_callback
        self.max_tokens = max_tokens
        self.only_n_most_recent_images = only_n_most_recent_images
        self.output_callback = output_callback

        self.print_usage = print_usage
        self.total_token_usage = 0
        self.total_cost = 0
        self.step_count = 0

        self.system = ''
           
    def __call__(self, messages: list, parsed_screen: list[str, list, dict]):
        self.step_count += 1
        image_base64 = parsed_screen['original_screenshot_base64']
        latency_omniparser = parsed_screen['latency']
        self.output_callback(f'-- Step {self.step_count}: --', sender="bot")
        screen_info = str(parsed_screen['screen_info'])



        screenshot_uuid = parsed_screen['screenshot_uuid']
        screen_width, screen_height = parsed_screen['width'], parsed_screen['height']

        boxids_and_labels = parsed_screen["screen_info"]
        system = self._get_system_prompt(boxids_and_labels)

        # drop looping actions msg, byte image etc
        planner_messages = messages
        planner_messages = _keep_latest_images(planner_messages)
        # if self.only_n_most_recent_images:
        #     _maybe_filter_to_n_most_recent_images(planner_messages, self.only_n_most_recent_images)
        # print(f"filtered_messages: {planner_messages}\n\n", "full messages:", messages)

        if isinstance(planner_messages[-1], dict):
            if not isinstance(planner_messages[-1]["content"], list):
                planner_messages[-1]["content"] = [planner_messages[-1]["content"]]
            planner_messages[-1]["content"].append(f"{OUTPUT_DIR}/screenshot_{screenshot_uuid}.png")
            planner_messages[-1]["content"].append(f"{OUTPUT_DIR}/screenshot_som_{screenshot_uuid}.png")

        start = time.time()
        if "gpt" in self.model:
            vlm_response, token_usage = run_oai_interleaved(
                messages=planner_messages,
                system=system,
                llm=self.model,
                api_key=self.api_key,
                max_tokens=self.max_tokens,
                temperature=0,
            )
            print(f"oai token usage: {token_usage}")
            self.total_token_usage += token_usage
            self.total_cost += (token_usage * 0.15 / 1000000)  # https://openai.com/api/pricing/
        elif "r1" in self.model:
            print(f"Sending messages to Groq: {planner_messages}")
            vlm_response, token_usage = run_groq_interleaved(
                messages=planner_messages,
                system=system,
                llm=self.model,
                api_key=self.api_key,
                max_tokens=self.max_tokens,
            )
            print(f"groq token usage: {token_usage}")
            self.total_token_usage += token_usage
            self.total_cost += (token_usage * 0.99 / 1000000)
        else:
            raise ValueError(f"Model {self.model} not supported")
        latency_vlm = time.time() - start
        self.output_callback(f"LLM: {latency_vlm:.2f}s, OmniParser: {latency_omniparser:.2f}s", sender="bot")

        print(f"{vlm_response}")
        
        if self.print_usage:
            print(f"Total token so far: {self.total_token_usage}. Total cost so far: $USD{self.total_cost:.5f}")
        
        vlm_response_json = extract_data(vlm_response, "json")
        vlm_response_json = json.loads(vlm_response_json)

        img_to_show_base64 = parsed_screen["som_image_base64"]
        if "Box ID" in vlm_response_json:
            bbox = parsed_screen["parsed_content_list"][int(vlm_response_json["Box ID"])]["bbox"]
            vlm_response_json["box_centroid_coordinate"] = [int((bbox[0] + bbox[2]) / 2 * screen_width), int((bbox[1] + bbox[3]) / 2 * screen_height)]
            img_to_show_data = base64.b64decode(img_to_show_base64)
            img_to_show = Image.open(BytesIO(img_to_show_data))

            draw = ImageDraw.Draw(img_to_show)
            x, y = vlm_response_json["box_centroid_coordinate"] 
            radius = 10
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill='red')
            draw.ellipse((x - radius*3, y - radius*3, x + radius*3, y + radius*3), fill=None, outline='red', width=2)

            buffered = BytesIO()
            img_to_show.save(buffered, format="PNG")
            img_to_show_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        self.output_callback(f'<img src="data:image/png;base64,{img_to_show_base64}">', sender="bot")
        self.output_callback(
                    f'<details>'
                    f'  <summary>Screen Info for OmniParser Agent</summary>'
                    f'  <pre>{screen_info}</pre>'
                    f'</details>',
                    sender="bot"
                )
        vlm_plan_str = ""
        for key, value in vlm_response_json.items():
            if key == "Reasoning":
                vlm_plan_str += f'{value}'
            else:
                vlm_plan_str += f'\n{key}: {value}'

        # construct the response so that anthropicExcutor can execute the tool
        response_content = [BetaTextBlock(text=vlm_plan_str, type='text')]
        if 'box_centroid_coordinate' in vlm_response_json:
            move_cursor_block = BetaToolUseBlock(id=f'toolu_{uuid.uuid4()}',
                                            input={'action': 'mouse_move', 'coordinate': vlm_response_json["box_centroid_coordinate"]},
                                            name='computer', type='tool_use')
            response_content.append(move_cursor_block)

        if vlm_response_json["Next Action"] == "type":
            click_block = BetaToolUseBlock(id=f'toolu_{uuid.uuid4()}', input={'action': 'left_click'}, name='computer', type='tool_use')
            sim_content_block = BetaToolUseBlock(id=f'toolu_{uuid.uuid4()}',
                                        input={'action': vlm_response_json["Next Action"], 'text': vlm_response_json["value"]},
                                        name='computer', type='tool_use')
            response_content.extend([click_block, sim_content_block])
        elif vlm_response_json["Next Action"] == "None":
            print("Task paused/completed.")
        else:
            sim_content_block = BetaToolUseBlock(id=f'toolu_{uuid.uuid4()}',
                                            input={'action': vlm_response_json["Next Action"]},
                                            name='computer', type='tool_use')
            response_content.append(sim_content_block)
        response_message = BetaMessage(id=f'toolu_{uuid.uuid4()}', content=response_content, model='', role='assistant', type='message', stop_reason='tool_use', usage=BetaUsage(input_tokens=0, output_tokens=0))
        return response_message, vlm_response_json

    def _api_response_callback(self, response: APIResponse):
        self.api_response_callback(response)

    def _get_system_prompt(self, screen_info: str = ""):
        main_section = f"""
You are using a Windows device.
You are able to use a mouse and keyboard to interact with the computer based on the given task and screenshot.
You can only interact with the desktop GUI (no terminal or application menu access).

You may be given some history plan and actions, this is the response from the previous loop.
You should carefully consider your plan base on the task, screenshot, and history actions.

Here is the list of all detected bounding boxes by IDs on the screen and their description:{screen_info}

Your available "Next Action" only include:
- type: type a string of text.
- left_click: Describe the ui element to be clicked.
- right_click: Describe the ui element to be right clicked.
- double_click: Describe the ui element to be double clicked.
- hover: Describe the ui element to be hovered.
- scroll_up: Scroll the screen up.
- scroll_down: Scroll the screen down.
- wait: Wait for 1 second for the device to load or respond.

Based on the visual information from the screenshot image and the detected bounding boxes, please determine the next action, the Box ID you should operate on, and the value (if the action is 'type') in order to complete the task.

Output format:
```json
{{
    "Reasoning": str, # describe what is in the current screen, taking into account the history, then describe your step-by-step thoughts on how to achieve the task, choose one action from available actions at a time.
    "Next Action": "action_type, action description" | "None" # one action at a time, describe it in short and precisely. 
    "Box ID": n,
    "value": "xxx" # only provide value field if the action is type, else don't include value key
}}
```

One Example:
```json
{{  
    "Reasoning": "The current screen shows google result of amazon, in previous action I have searched amazon on google. Then I need to click on the first search results to go to amazon.com.",
    "Next Action": "left_click",
    "Box ID": m
}}
```

Another Example:
```json
{{
    "Reasoning": "The current screen shows the front page of amazon. There is no previous action. Therefore I need to type "Apple watch" in the search bar.",
    "Next Action": "type",
    "Box ID": n,
    "value": "Apple watch"
}}
```

IMPORTANT NOTES:
1. You should only give a single action at a time.

"""
        thinking_model = "r1" in self.model
        if not thinking_model:
            main_section += """
2. You should give an analysis to the current screen, and reflect on what has been done by looking at the history, then describe your step-by-step thoughts on how to achieve the task.

"""
        else:
            main_section += """
2. In <think> XML tags give an analysis to the current screen, and reflect on what has been done by looking at the history, then describe your step-by-step thoughts on how to achieve the task. In <output> XML tags put the next action prediction JSON.

"""
        main_section += """
3. Attach the next action prediction in the "Next Action".
4. You should not include other actions, such as keyboard shortcuts.
5. When the task is completed, don't complete additional actions. You should say "Next Action": "None" in the json field.
""" 

        return main_section

def _keep_latest_images(messages):
    for i in range(len(messages)-1):
        if isinstance(messages[i]["content"], list):
            for cnt in messages[i]["content"]:
                if isinstance(cnt, str):
                    if cnt.endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif")):
                        messages[i]["content"].remove(cnt)
    return messages


def _maybe_filter_to_n_most_recent_images(
    messages: list[BetaMessageParam],
    images_to_keep: int,
    min_removal_threshold: int = 10,
):
    """
    With the assumption that images are screenshots that are of diminishing value as
    the conversation progresses, remove all but the final `images_to_keep` tool_result
    images in place, with a chunk of min_removal_threshold to reduce the amount we
    break the implicit prompt cache.
    """
    if images_to_keep is None:
        return messages

    tool_result_blocks = cast(
        list[ToolResultBlockParam],
        [
            item
            for message in messages
            for item in (
                message["content"] if isinstance(message["content"], list) else []
            )
            if isinstance(item, dict) and item.get("type") == "tool_result"
        ],
    )

    total_images = sum(
        1
        for tool_result in tool_result_blocks
        for content in tool_result.get("content", [])
        if isinstance(content, dict) and content.get("type") == "image"
    )

    images_to_remove = total_images - images_to_keep
    # for better cache behavior, we want to remove in chunks
    images_to_remove -= images_to_remove % min_removal_threshold

    for tool_result in tool_result_blocks:
        if isinstance(tool_result.get("content"), list):
            new_content = []
            for content in tool_result.get("content", []):
                if isinstance(content, dict) and content.get("type") == "image":
                    if images_to_remove > 0:
                        images_to_remove -= 1
                        continue
                new_content.append(content)
            tool_result["content"] = new_content