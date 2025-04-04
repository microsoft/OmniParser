import os
from google import genai
from google.genai import types
import tiktoken

from .utils import is_image_path, encode_image

def estimate_token_count(text):
    """Estimates the token count of a text string using tiktoken.
       Adapt this for Gemini's specific vocabulary if necessary."""

    # IMPORTANT:  tiktoken is primarily for OpenAI models.
    # You need to be aware of potential inaccuracies if Gemini
    # uses a significantly different tokenization scheme.

    try:
        encoding = tiktoken.get_encoding("cl100k_base") # This is a good starting point, but research Gemini tokenizer
        tokens = encoding.encode(text)
        return len(tokens)
    except Exception as e:
        print(f"Error estimating token count: {e}")
        return None  # or a reasonable default

def run_gemini_interleaved(messages: list, system: str, model_name: str, api_key: str, temperature=0):    
    """
    Run a chat completion through Gemini's API, ignoring any images in the messages.
    """
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set")
    
    client = genai.Client(
        api_key=api_key,
    )

    generate_content_config = types.GenerateContentConfig(
        temperature=temperature,
        response_mime_type="application/json",
        system_instruction=[
            types.Part.from_text(text=system),
        ],
    )

    contents = []

    if type(messages) == list:
        for item in messages:
            parts = []
            if isinstance(item, dict):
                for cnt in item["content"]:
                    if isinstance(cnt, str):
                        parts.append(types.Part.from_text(text=cnt))
                    else:
                        # in this case it is a text block from anthropic
                        parts.append(types.Part.from_text(text=str(cnt)))
                
            else:  # str
                parts.append(types.Part.from_text(text=str(item)))

            content = (types.Content(
                role="user",
                parts=parts
            ))
            
            contents.append(content)

    
    elif isinstance(messages, str):
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=messages)]
            )
        ]

    try:
        response = client.models.generate_content(
            model=model_name, 
            contents=contents,
            config=generate_content_config
        )
        final_answer = response.text
        token_usage = estimate_token_count(final_answer)

        return final_answer, token_usage
    except Exception as e:
        print(f"Error in interleaved Gemini: {e}")

        return str(e), 0
