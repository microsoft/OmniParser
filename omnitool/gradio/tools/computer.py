import base64
import time
from enum import StrEnum
from typing import Literal, TypedDict

from PIL import Image

from anthropic.types.beta import BetaToolComputerUse20241022Param

from .base import BaseAnthropicTool, ToolError, ToolResult
from .screen_capture import get_screenshot
import requests
import re

OUTPUT_DIR = "./tmp/outputs"

TYPING_DELAY_MS = 12
TYPING_GROUP_SIZE = 50

Action = Literal[
    "key",
    "type",
    "mouse_move",
    "left_click",
    "left_click_drag",
    "right_click",
    "middle_click",
    "double_click",
    "screenshot",
    "cursor_position",
    "hover",
    "wait"
]


class Resolution(TypedDict):
    width: int
    height: int


MAX_SCALING_TARGETS: dict[str, Resolution] = {
    "XGA": Resolution(width=1024, height=768),  # 4:3
    "WXGA": Resolution(width=1280, height=800),  # 16:10
    "FWXGA": Resolution(width=1366, height=768),  # ~16:9
}


class ScalingSource(StrEnum):
    COMPUTER = "computer"
    API = "api"


class ComputerToolOptions(TypedDict):
    display_height_px: int
    display_width_px: int
    display_number: int | None


def chunks(s: str, chunk_size: int) -> list[str]:
    return [s[i : i + chunk_size] for i in range(0, len(s), chunk_size)]

class ComputerTool(BaseAnthropicTool):
    """
    A tool that allows the agent to interact with the screen, keyboard, and mouse of the current computer.
    Adapted for Windows using 'pyautogui'.
    """

    name: Literal["computer"] = "computer"
    api_type: Literal["computer_20241022"] = "computer_20241022"
    width: int
    height: int
    display_num: int | None

    _screenshot_delay = 2.0
    _scaling_enabled = True

    @property
    def options(self) -> ComputerToolOptions:
        width, height = self.scale_coordinates(
            ScalingSource.COMPUTER, self.width, self.height
        )
        return {
            "display_width_px": width,
            "display_height_px": height,
            "display_number": self.display_num,
        }

    def to_params(self) -> BetaToolComputerUse20241022Param:
        return {"name": self.name, "type": self.api_type, **self.options}

    def __init__(self, is_scaling: bool = False):
        super().__init__()

        # Get screen width and height using Windows command
        self.display_num = None
        self.offset_x = 0
        self.offset_y = 0
        self.is_scaling = is_scaling
        self.width, self.height = self.get_screen_size()
        print(f"screen size: {self.width}, {self.height}")

        self.key_conversion = {"Page_Down": "pagedown",
                               "Page_Up": "pageup",
                               "Super_L": "win",
                               "Escape": "esc"}


    async def __call__(
        self,
        *,
        action: Action,
        text: str | None = None,
        coordinate: tuple[int, int] | None = None,
        **kwargs,
    ):
        print(f"action: {action}, text: {text}, coordinate: {coordinate}, is_scaling: {self.is_scaling}")
        if action in ("mouse_move", "left_click_drag"):
            if coordinate is None:
                raise ToolError(f"coordinate is required for {action}")
            if text is not None:
                raise ToolError(f"text is not accepted for {action}")
            if not isinstance(coordinate, (list, tuple)) or len(coordinate) != 2:
                raise ToolError(f"{coordinate} must be a tuple of length 2")
            # if not all(isinstance(i, int) and i >= 0 for i in coordinate):
            if not all(isinstance(i, int) for i in coordinate):
                raise ToolError(f"{coordinate} must be a tuple of non-negative ints")
            
            if self.is_scaling:
                x, y = self.scale_coordinates(
                    ScalingSource.API, coordinate[0], coordinate[1]
                )
            else:
                x, y = coordinate

            # print(f"scaled_coordinates: {x}, {y}")
            # print(f"offset: {self.offset_x}, {self.offset_y}")
            
            # x += self.offset_x # TODO - check if this is needed
            # y += self.offset_y

            print(f"mouse move to {x}, {y}")
            
            if action == "mouse_move":
                self.send_to_vm(f"pyautogui.moveTo({x}, {y})")
                return ToolResult(output=f"Moved mouse to ({x}, {y})")
            elif action == "left_click_drag":
                current_x, current_y = self.send_to_vm("pyautogui.position()")
                self.send_to_vm(f"pyautogui.dragTo({x}, {y}, duration=0.5)")
                return ToolResult(output=f"Dragged mouse from ({current_x}, {current_y}) to ({x}, {y})")

        if action in ("key", "type"):
            if text is None:
                raise ToolError(f"text is required for {action}")
            if coordinate is not None:
                raise ToolError(f"coordinate is not accepted for {action}")
            if not isinstance(text, str):
                raise ToolError(output=f"{text} must be a string")

            if action == "key":
                # Handle key combinations
                keys = text.split('+')
                for key in keys:
                    key = self.key_conversion.get(key.strip(), key.strip())
                    key = key.lower()
                    self.send_to_vm(f"pyautogui.keyDown('{key}')")  # Press down each key
                for key in reversed(keys):
                    key = self.key_conversion.get(key.strip(), key.strip())
                    key = key.lower()
                    self.send_to_vm(f"pyautogui.keyUp('{key}')")    # Release each key in reverse order
                return ToolResult(output=f"Pressed keys: {text}")
            
            elif action == "type":
                # default click before type TODO: check if this is needed
                self.send_to_vm("pyautogui.click()")
                self.send_to_vm(f"pyautogui.typewrite('{text}', interval={TYPING_DELAY_MS / 1000})")
                self.send_to_vm("pyautogui.press('enter')")
                screenshot_base64 = (await self.screenshot()).base64_image
                return ToolResult(output=text, base64_image=screenshot_base64)

        if action in (
            "left_click",
            "right_click",
            "double_click",
            "middle_click",
            "screenshot",
            "cursor_position",
            "left_press",
        ):
            if text is not None:
                raise ToolError(f"text is not accepted for {action}")
            if coordinate is not None:
                raise ToolError(f"coordinate is not accepted for {action}")

            if action == "screenshot":
                return await self.screenshot()
            elif action == "cursor_position":
                x, y = self.send_to_vm("pyautogui.position()")
                x, y = self.scale_coordinates(ScalingSource.COMPUTER, x, y)
                return ToolResult(output=f"X={x},Y={y}")
            else:
                if action == "left_click":
                    self.send_to_vm("pyautogui.click()")
                elif action == "right_click":
                    self.send_to_vm("pyautogui.rightClick()")
                elif action == "middle_click":
                    self.send_to_vm("pyautogui.middleClick()")
                elif action == "double_click":
                    self.send_to_vm("pyautogui.doubleClick()")
                elif action == "left_press":
                    self.send_to_vm("pyautogui.mouseDown()")
                    time.sleep(1)
                    self.send_to_vm("pyautogui.mouseUp()")
                return ToolResult(output=f"Performed {action}")
        if action in ("scroll_up", "scroll_down"):
            if action == "scroll_up":
                self.send_to_vm("pyautogui.scroll(100)")
            elif action == "scroll_down":
                self.send_to_vm("pyautogui.scroll(-100)")
            return ToolResult(output=f"Performed {action}")
        if action == "hover":
            return ToolResult(output=f"Performed {action}")
        if action == "wait":
            time.sleep(1)
            return ToolResult(output=f"Performed {action}")
        raise ToolError(f"Invalid action: {action}")

    def send_to_vm(self, action: str):
        """
        Executes a python command on the server. Only return tuple of x,y when action is "pyautogui.position()"
        """
        prefix = "import pyautogui; pyautogui.FAILSAFE = False;"
        command_list = ["python", "-c", f"{prefix} {action}"]
        parse = action == "pyautogui.position()"
        if parse:
            command_list[-1] = f"{prefix} print({action})"

        try:
            print(f"sending to vm: {command_list}")
            response = requests.post(
                f"http://localhost:5000/execute", 
                headers={'Content-Type': 'application/json'},
                json={"command": command_list},
                timeout=90
            )
            time.sleep(0.7) # avoid async error as actions take time to complete
            print(f"action executed")
            if response.status_code != 200:
                raise ToolError(f"Failed to execute command. Status code: {response.status_code}")
            if parse:
                output = response.json()['output'].strip()
                match = re.search(r'Point\(x=(\d+),\s*y=(\d+)\)', output)
                if not match:
                    raise ToolError(f"Could not parse coordinates from output: {output}")
                x, y = map(int, match.groups())
                return x, y
        except requests.exceptions.RequestException as e:
            raise ToolError(f"An error occurred while trying to execute the command: {str(e)}")

    async def screenshot(self):
        if not hasattr(self, 'target_dimension'):
            screenshot = self.padding_image(screenshot)
            self.target_dimension = MAX_SCALING_TARGETS["WXGA"]
        width, height = self.target_dimension["width"], self.target_dimension["height"]
        screenshot, path = get_screenshot(resize=True, target_width=width, target_height=height)
        time.sleep(0.7) # avoid async error as actions take time to complete
        return ToolResult(base64_image=base64.b64encode(path.read_bytes()).decode())

    def padding_image(self, screenshot):
        """Pad the screenshot to 16:10 aspect ratio, when the aspect ratio is not 16:10."""
        _, height = screenshot.size
        new_width = height * 16 // 10

        padding_image = Image.new("RGB", (new_width, height), (255, 255, 255))
        # padding to top left
        padding_image.paste(screenshot, (0, 0))
        return padding_image

    def scale_coordinates(self, source: ScalingSource, x: int, y: int):
        """Scale coordinates to a target maximum resolution."""
        if not self._scaling_enabled:
            return x, y
        ratio = self.width / self.height
        target_dimension = None

        for target_name, dimension in MAX_SCALING_TARGETS.items():
            # allow some error in the aspect ratio - not ratios are exactly 16:9
            if abs(dimension["width"] / dimension["height"] - ratio) < 0.02:
                if dimension["width"] < self.width:
                    target_dimension = dimension
                    self.target_dimension = target_dimension
                    # print(f"target_dimension: {target_dimension}")
                break

        if target_dimension is None:
            # TODO: currently we force the target to be WXGA (16:10), when it cannot find a match
            target_dimension = MAX_SCALING_TARGETS["WXGA"]
            self.target_dimension = MAX_SCALING_TARGETS["WXGA"]

        # should be less than 1
        x_scaling_factor = target_dimension["width"] / self.width
        y_scaling_factor = target_dimension["height"] / self.height
        if source == ScalingSource.API:
            if x > self.width or y > self.height:
                raise ToolError(f"Coordinates {x}, {y} are out of bounds")
            # scale up
            return round(x / x_scaling_factor), round(y / y_scaling_factor)
        # scale down
        return round(x * x_scaling_factor), round(y * y_scaling_factor)

    def get_screen_size(self):
        """Return width and height of the screen"""
        try:
            response = requests.post(
                f"http://localhost:5000/execute",
                headers={'Content-Type': 'application/json'},
                json={"command": ["python", "-c", "import pyautogui; print(pyautogui.size())"]},
                timeout=90
            )
            if response.status_code != 200:
                raise ToolError(f"Failed to get screen size. Status code: {response.status_code}")
            
            output = response.json()['output'].strip()
            match = re.search(r'Size\(width=(\d+),\s*height=(\d+)\)', output)
            if not match:
                raise ToolError(f"Could not parse screen size from output: {output}")
            width, height = map(int, match.groups())
            return width, height
        except requests.exceptions.RequestException as e:
            raise ToolError(f"An error occurred while trying to get screen size: {str(e)}")