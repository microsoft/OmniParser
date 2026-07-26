import os
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import Optional
from PIL import Image
from pprint import pprint

from .utils import is_image_path, encode_image

class Action(BaseModel):
    reasoning: str = Field(..., alias="Reasoning")
    next_action: str = Field(..., alias="Next Action")
    box_id: str | None = Field(None, alias="Box ID")
    value: str | None = None

def run_gemini_interleaved(messages: list, system: str, model_name: str, api_key: str, max_tokens: int, temperature=0):    
    """
    Run a chat completion through Google Gemini's API
    """
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set")
    
    client = genai.Client(
        api_key=api_key,
    )

    generate_content_config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
        response_mime_type="application/json",
        response_schema=Action,
        system_instruction=[
            types.Part.from_text(text=system),
        ],
    )

    contents = []

    if type(messages) == list:
        for item in messages:
            if isinstance(item, dict):
                for cnt in item["content"]:
                    if isinstance(cnt, str):
                        if is_image_path(cnt):
                            contents.append(Image.open(cnt))
                        else:
                            contents.append(cnt)
                    else:
                        contents.append(str(cnt))
                
            else:  # str
                contents.append(str(cnt))

    elif isinstance(messages, str):
        contents.push(messages)

    try:
        response = client.models.generate_content(
            model=model_name, 
            contents=contents,
            config=generate_content_config
        )
        final_answer = response.text
        token_usage = response.usage_metadata.total_token_count

        return final_answer, token_usage
    except Exception as e:
        print(f"Error in interleaved Gemini: {e}")

        return str(e), 0
