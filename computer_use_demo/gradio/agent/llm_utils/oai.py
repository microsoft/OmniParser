import os
import logging
import base64
import requests
from .utils import is_image_path, encode_image

def run_oai_interleaved(messages: list, system: str, llm: str, api_key: str, max_tokens=256, temperature=0):
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set")
    
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {api_key}"}

    final_messages = [{"role": "system", "content": system}]

    if type(messages) == list:
        for item in messages:
            contents = []
            if isinstance(item, dict):
                for cnt in item["content"]:
                    if isinstance(cnt, str):
                        if is_image_path(cnt):
                            base64_image = encode_image(cnt)
                            content = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        else:
                            content = {"type": "text", "text": cnt}
                    else:
                        # in this case it is a text block from anthropic
                        content = {"type": "text", "text": str(cnt)}
                        
                    contents.append(content)
                    
                message = {"role": 'user', "content": contents}
            else:  # str
                contents.append({"type": "text", "text": item})
                message = {"role": "user", "content": contents}
            
            final_messages.append(message)

    
    elif isinstance(messages, str):
        final_messages = [{"role": "user", "content": messages}]

    payload = {
        "model": llm,
        "messages": final_messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    )

    try:
        text = response.json()['choices'][0]['message']['content']
        token_usage = int(response.json()['usage']['total_tokens'])
        return text, token_usage
    except Exception as e:
        print(f"Error in interleaved openAI: {e}. This may due to your invalid OPENAI_API_KEY. Please check the response: {response.json()} ")
        return response.json()