import asyncio
from typing import Any, Dict, cast
from collections.abc import Callable
from anthropic.types.beta import (
    BetaContentBlock,
    BetaContentBlockParam,
    BetaImageBlockParam,
    BetaMessage,
    BetaMessageParam,
    BetaTextBlockParam,
    BetaToolResultBlockParam,
)
from anthropic.types import TextBlock
from anthropic.types.beta import BetaMessage, BetaTextBlock, BetaToolUseBlock
from tools import ComputerTool, ToolCollection, ToolResult


class AnthropicExecutor:
    def __init__(
        self, 
        output_callback: Callable[[BetaContentBlockParam], None], 
        tool_output_callback: Callable[[Any, str], None],
    ):
        self.tool_collection = ToolCollection(
            ComputerTool()
        )
        self.output_callback = output_callback
        self.tool_output_callback = tool_output_callback

    def __call__(self, response: BetaMessage, messages: list[BetaMessageParam]):
        new_message = {
            "role": "assistant",
            "content": cast(list[BetaContentBlockParam], response.content),
        }
        if new_message not in messages:
            messages.append(new_message)
        else:
            print("new_message already in messages, there are duplicates.")
        
        tool_result_content: list[BetaToolResultBlockParam] = []
        for content_block in cast(list[BetaContentBlock], response.content):
            self.output_callback(content_block, sender="bot")
            # Execute the tool
            if content_block.type == "tool_use":
                # Run the asynchronous tool execution in a synchronous context
                result = asyncio.run(self.tool_collection.run(
                    name=content_block.name,
                    tool_input=cast(dict[str, Any], content_block.input),
                ))
                
                self.output_callback(result, sender="bot")
                
                tool_result_content.append(
                    _make_api_tool_result(result, content_block.id)
                )
                self.tool_output_callback(result, content_block.id)

            # Craft messages based on the content_block
            # Note: to display the messages in the gradio, you should organize the messages in the following way (user message, bot message)
            
            display_messages = _message_display_callback(messages)
            # display_messages = []
            
            # Send the messages to the gradio
            for user_msg, bot_msg in display_messages:
                # yield [user_msg, bot_msg], tool_result_content
                yield [None, None], tool_result_content

        if not tool_result_content:
            return messages
        
        return tool_result_content

def _message_display_callback(messages):
    display_messages = []
    for msg in messages:
        try:
            if isinstance(msg["content"][0], TextBlock):
                display_messages.append((msg["content"][0].text, None))  # User message
            elif isinstance(msg["content"][0], BetaTextBlock):
                display_messages.append((None, msg["content"][0].text))  # Bot message
            elif isinstance(msg["content"][0], BetaToolUseBlock):
                display_messages.append((None, f"Tool Use: {msg['content'][0].name}\nInput: {msg['content'][0].input}"))  # Bot message
            elif isinstance(msg["content"][0], Dict) and msg["content"][0]["content"][-1]["type"] == "image":
                display_messages.append((None, f'<img src="data:image/png;base64,{msg["content"][0]["content"][-1]["source"]["data"]}">'))  # Bot message
            else:
                print(msg["content"][0])
        except Exception as e:
            print("error", e)
            pass
    return display_messages

def _make_api_tool_result(
    result: ToolResult, tool_use_id: str
) -> BetaToolResultBlockParam:
    """Convert an agent ToolResult to an API ToolResultBlockParam."""
    tool_result_content: list[BetaTextBlockParam | BetaImageBlockParam] | str = []
    is_error = False
    if result.error:
        is_error = True
        tool_result_content = _maybe_prepend_system_tool_result(result, result.error)
    else:
        if result.output:
            tool_result_content.append(
                {
                    "type": "text",
                    "text": _maybe_prepend_system_tool_result(result, result.output),
                }
            )
        if result.base64_image:
            tool_result_content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": result.base64_image,
                    },
                }
            )
    return {
        "type": "tool_result",
        "content": tool_result_content,
        "tool_use_id": tool_use_id,
        "is_error": is_error,
    }


def _maybe_prepend_system_tool_result(result: ToolResult, result_text: str):
    if result.system:
        result_text = f"<system>{result.system}</system>\n{result_text}"
    return result_text