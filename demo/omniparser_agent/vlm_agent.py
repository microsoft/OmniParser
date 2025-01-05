import json
import asyncio
import platform
from collections.abc import Callable
from datetime import datetime
from enum import StrEnum
from typing import Any, cast, Dict, Callable
import uuid
import requests
from PIL import Image, ImageDraw
import base64
from io import BytesIO

from anthropic import Anthropic, AnthropicBedrock, AnthropicVertex, APIResponse
from anthropic.types import TextBlock, ToolResultBlockParam
from anthropic.types.beta import BetaMessage, BetaTextBlock, BetaToolUseBlock, BetaMessageParam, BetaUsage

from tools.screen_capture import get_screenshot
from gui_agent.llm_utils.oai import run_oai_interleaved, encode_image
from gui_agent.llm_utils.qwen import run_qwen
from gui_agent.llm_utils.llm_utils import extract_data
from tools.colorful_text import colorful_text_showui, colorful_text_vlm


SYSTEM_PROMPT = f"""<SYSTEM_CAPABILITY>
* You are utilizing a Windows system with internet access.
* The current date is {datetime.today().strftime('%A, %B %d, %Y')}.
</SYSTEM_CAPABILITY>
"""


class OmniParser:
    def __init__(self, 
                 url: str,
                 selected_screen: int = 0) -> None:
        self.url = url
        self.selected_screen = selected_screen

    def __call__(self,):
        screenshot, screenshot_path = get_screenshot(selected_screen=self.selected_screen)
        screenshot_path = str(screenshot_path)
        image_base64 = encode_image(screenshot_path)

        # response = requests.post(self.url, json={"base64_image": image_base64, 'prompt': 'omniparser process'})
        # response_json = response.json()
        # example response_json: {"som_image_base64": dino_labled_img, "parsed_content_list": parsed_content_list, "latency": 0.1}
        # Debug
        response_json = {"som_image_base64": image_base64, "parsed_content_list": ['debug1', 'debug2'], "latency": 0.1}
        print('omniparser latency:', response_json['latency'])
        response_json = self.reformat_messages(response_json)
        return response_json
    
    def reformat_messages(self, response_json: dict):
        parsed_content_list = response_json["parsed_content_list"]
        screen_info = ""
        # Debug
        # for idx, element in enumerate(parsed_content_list):
        #     element['idx'] = idx
        #     if element['type'] == 'text':
        #         # screen_info += f'''<p id={idx} class="text" alt="{element['content']}"> </p>\n'''
        #         screen_info += f'ID: {idx}, Text: {element["content"]}\n'
        #     elif element['type'] == 'icon':
        #         # screen_info += f'''<img id={idx} class="icon" alt="{element['content']}"> </img>\n'''
        #         screen_info += f'ID: {idx}, Icon: {element["content"]}\n'
        response_json['screen_info'] = screen_info
        return response_json



class VLMAgent:
    def __init__(
        self,
        model: str, 
        provider: str, 
        system_prompt_suffix: str, 
        api_key: str,
        output_callback: Callable, 
        api_response_callback: Callable,
        max_tokens: int = 4096,
        only_n_most_recent_images: int | None = None,
        selected_screen: int = 0,
        print_usage: bool = True,
    ):
        if model == "gpt-4o + ShowUI":
            self.model = "gpt-4o-2024-11-20"
        elif model == "gpt-4o-mini + ShowUI":
            self.model = "gpt-4o-mini"  # "gpt-4o-mini"
        elif model == "qwen2vl + ShowUI":
            self.model = "qwen2vl"
        elif model == "omniparser + gpt-4o":
            self.model = "gpt-4o-2024-11-20"
        else:
            raise ValueError(f"Model {model} not supported")
        
        self.provider = provider
        self.system_prompt_suffix = system_prompt_suffix
        self.api_key = api_key
        self.api_response_callback = api_response_callback
        self.max_tokens = max_tokens
        self.only_n_most_recent_images = only_n_most_recent_images
        self.selected_screen = selected_screen
        self.output_callback = output_callback

        self.print_usage = print_usage
        self.total_token_usage = 0
        self.total_cost = 0

        self.system = (
            # f"{SYSTEM_PROMPT}{' ' + system_prompt_suffix if system_prompt_suffix else ''}"
            f"{system_prompt_suffix}"
        )
           
    def __call__(self, messages: list, parsed_screen: list[str, list]):
        # example parsed_screen: {"som_image_base64": dino_labled_img, "parsed_content_list": parsed_content_list, "screen_info"}
        screen_info = parsed_screen["screen_info"]
        # drop looping actions msg, byte image etc
        planner_messages = messages
        # planner_messages = _message_filter_callback(messages)  
        
        print(f"filtered_messages: {planner_messages}\n\n", "full messages:", messages)
        # import pdb; pdb.set_trace()
        planner_messages = _keep_latest_images(planner_messages)
        # if self.only_n_most_recent_images:
        #     _maybe_filter_to_n_most_recent_images(planner_messages, self.only_n_most_recent_images)

        system = self._get_system_prompt(screen_info) + self.system_prompt_suffix

        # Take a screenshot
        screenshot, screenshot_path = get_screenshot(selected_screen=self.selected_screen)
        screen_width, screen_height = screenshot.size
        screenshot_path = str(screenshot_path)
        image_base64 = encode_image(screenshot_path)

        som_image_data = base64.b64decode(parsed_screen['som_image_base64'])
        som_screenshot_path = f"./tmp/outputs/screenshot_som_{uuid.uuid4().hex}.png"
        with open(som_screenshot_path, "wb") as f:
            f.write(som_image_data)

        self.output_callback(f'Screenshot for {colorful_text_vlm}:\n<img src="data:image/png;base64,{image_base64}">',
                             sender="bot")
        self.output_callback(f'Set of Marks Screenshot for {colorful_text_vlm}:\n<img src="data:image/png;base64,{parsed_screen['som_image_base64']}">', sender="bot")


        if isinstance(planner_messages[-1], dict):
            if not isinstance(planner_messages[-1]["content"], list):
                planner_messages[-1]["content"] = [planner_messages[-1]["content"]]
            planner_messages[-1]["content"].append(screenshot_path)
            planner_messages[-1]["content"].append(som_screenshot_path)

        print(f"Sending messages to VLMPlanner : {planner_messages}")

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
            
        elif "qwen" in self.model:
            vlm_response, token_usage = run_qwen(
                messages=planner_messages,
                system=system,
                llm=self.model,
                api_key=self.api_key,
                max_tokens=self.max_tokens,
                temperature=0,
            )
            print(f"qwen token usage: {token_usage}")
            self.total_token_usage += token_usage
            self.total_cost += (token_usage * 0.02 / 7.25 / 1000)  # 1USD=7.25CNY, https://help.aliyun.com/zh/dashscope/developer-reference/tongyi-qianwen-vl-plus-api
        elif "phi" in self.model:
            pass # TODO
        else:
            raise ValueError(f"Model {self.model} not supported")

        print(f"VLMPlanner response: {vlm_response}")
        
        if self.print_usage:
            print(f"VLMPlanner total token usage so far: {self.total_token_usage}. Total cost so far: $USD{self.total_cost:.5f}")
        
        vlm_response_json = extract_data(vlm_response, "json")
        vlm_response_json = json.loads(vlm_response_json)

        # map "box_id" to "idx" in parsed_screen, and output the xy coordinate of bbox
        # TODO add try except for the case when "box_id" is not in the response
        # if 'Box ID' in vlm_response_json:
        try:
            bbox = parsed_screen["parsed_content_list"][int(vlm_response_json["Box ID"])]["bbox"]
            vlm_response_json["coordinate"] = [int((bbox[0] + bbox[2]) / 2 * screen_width), int((bbox[1] + bbox[3]) / 2 * screen_height)]
            # draw a circle on the screenshot image to indicate the action
            self.draw_action(vlm_response_json, image_base64)
        except:
            print("No Box ID in the response.")


        # vlm_plan_str = '\n'.join([f'{key}: {value}' for key, value in json.loads(response).items()])
        vlm_plan_str = ""
        for key, value in vlm_response_json.items():
            if key == "Reasoning":
                vlm_plan_str += f'{value}'
            else:
                vlm_plan_str += f'\n{key}: {value}'
        
        # self.output_callback(f"{colorful_text_vlm}:\n{vlm_plan_str}", sender="bot")

        # construct the response so that anthropicExcutor can execute the tool
        analysis = BetaTextBlock(text=vlm_plan_str, type='text')
        if 'coordinate' in vlm_response_json:
            move_cursor_block = BetaToolUseBlock(id=f'toolu_{uuid.uuid4()}',
                                            input={'action': 'mouse_move', 'coordinate': vlm_response_json["coordinate"]},
                                            name='computer', type='tool_use')
            response_content = [analysis, move_cursor_block]
        else:
            response_content = [analysis]
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

        response = BetaMessage(id=f'toolu_{uuid.uuid4()}', content=response_content, model='', role='assistant', type='message', stop_reason='tool_use', usage=BetaUsage(input_tokens=0, output_tokens=0))
        return response, vlm_response_json


    def _api_response_callback(self, response: APIResponse):
        self.api_response_callback(response)
        

    def reformat_messages(self, messages: list):
        pass

    def _get_system_prompt(self, screen_info: str = ""):
        datetime_str = datetime.now().strftime("%A, %B %d, %Y")
        os_name = platform.system()
        return f"""
You are using an {os_name} device.
You are able to use a mouse and keyboard to interact with the computer based on the given task and screenshot.
You can only interact with the desktop GUI (no terminal or application menu access).

You may be given some history plan and actions, this is the response from the previous loop.
You should carefully consider your plan base on the task, screenshot, and history actions.

Here is the list of all detected bounding boxes by IDs on the screen and their description:{screen_info}

Your available "Next Action" only include:
- type: type a string of text.
- left_click: Describe the ui element to be clicked.
- enter: Press an enter key.
- escape: Press an ESCAPE key.
- hover: Describe the ui element to be hovered.
- scroll: Scroll the screen, you must specify up or down.
- press: Describe the ui element to be pressed.

Based on the visual information from the screenshot image and the detected bounding boxes, please determine the next action, the Box ID you should operate on, and the value (if the action is 'type') in order to complete the task.

Output format:
```json
{{
    "Reasoning": str, # describe what is in the current screen, taking into account the history, then describe your step-by-step thoughts on how to achieve the task, choose one action from available actions at a time.
    "Next Action": "action_type, action description" | "None" # one action at a time, describe it in short and precisely. 
    'Box ID': n,
    'value': "xxx" # if the action is type, you should provide the text to type.
}}
```

One Example:
```json
{{  
    "Reasoning": "The current screen shows google result of amazon, in previous action I have searched amazon on google. Then I need to click on the first search results to go to amazon.com.",
    "Next Action": "left_click",
    'Box ID': m,
}}
```

Another Example:
```json
{{
    "Reasoning": "The current screen shows the front page of amazon. There is no previous action. Therefore I need to type "Apple watch" in the search bar.",
    "Next Action": "type",
    'Box ID': n,
    'value': "Apple watch"
}}
```

IMPORTANT NOTES:
1. You should only give a single action at a time.
2. You should give an analysis to the current screen, and reflect on what has been done by looking at the history, then describe your step-by-step thoughts on how to achieve the task.
3. Attach the next action prediction in the "Next Action".
4. You should not include other actions, such as keyboard shortcuts.
5. When the task is completed, you should say "Next Action": "None" in the json field.
""" 
    def draw_action(self, vlm_response_json, image_base64):
        # draw a circle using the coordinate in parsed_screen['som_image_base64']
        image_data = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_data))

        draw = ImageDraw.Draw(image)
        x, y = vlm_response_json["coordinate"] 
        radius = 10
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline='red', width=3)
        buffered = BytesIO()
        image.save('demo.png')
        image.save(buffered, format="PNG")
        image_with_circle_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        self.output_callback(f'Action performed on the Screenshot (red circle), for {colorful_text_vlm}:\n<img src="data:image/png;base64,{image_with_circle_base64}">', sender="bot")


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


def _message_filter_callback(messages):
    filtered_list = []
    try:
        for msg in messages:
            if msg.get('role') in ['user']:
                if not isinstance(msg["content"], list):
                    msg["content"] = [msg["content"]]
                if isinstance(msg["content"][0], TextBlock):
                    filtered_list.append(str(msg["content"][0].text))  # User message
                elif isinstance(msg["content"][0], str):
                    filtered_list.append(msg["content"][0])  # User message
                else:
                    print("[_message_filter_callback]: drop message", msg)
                    continue                

            # elif msg.get('role') in ['assistant']:
            #     if isinstance(msg["content"][0], TextBlock):
            #         msg["content"][0] = str(msg["content"][0].text)
            #     elif isinstance(msg["content"][0], BetaTextBlock):
            #         msg["content"][0] = str(msg["content"][0].text)
            #     elif isinstance(msg["content"][0], BetaToolUseBlock):
            #         msg["content"][0] = str(msg['content'][0].input)
            #     elif isinstance(msg["content"][0], Dict) and msg["content"][0]["content"][-1]["type"] == "image":
            #         msg["content"][0] = f'<img src="data:image/png;base64,{msg["content"][0]["content"][-1]["source"]["data"]}">'
            #     else:
            #         print("[_message_filter_callback]: drop message", msg)
            #         continue
            #     filtered_list.append(msg["content"][0])  # User message
                
            else:
                print("[_message_filter_callback]: drop message", msg)
                continue
            
    except Exception as e:
        print("[_message_filter_callback]: error", e)
                
    return filtered_list