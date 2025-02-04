import os
import logging
import base64
import requests
from .utils import is_image_path, encode_image

def run_oai_interleaved(messages: list, system: str, model_name: str, api_key: str, max_tokens=256, temperature=0, provider_base_url: str = "https://api.openai.com/v1"):    
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {api_key}"}
    final_messages = [{"role": "system", "content": system}]

    if type(messages) == list:
        for item in messages:
            contents = []
            if isinstance(item, dict):
                for cnt in item["content"]:
                    if isinstance(cnt, str):
                        if is_image_path(cnt) and 'o3-mini' not in model_name:
                            # 03 mini does not support images
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
        "model": model_name,
        "messages": final_messages,
    }
    if 'o1' in model_name or 'o3-mini' in model_name:
        payload['reasoning_effort'] = 'low'
        payload['max_completion_tokens'] = max_tokens
    else:
        payload['max_tokens'] = max_tokens

    response = requests.post(
        f"{provider_base_url}/chat/completions", headers=headers, json=payload
    )


    try:
        text = response.json()['choices'][0]['message']['content']
        token_usage = int(response.json()['usage']['total_tokens'])
        return text, token_usage
    except Exception as e:
        print(f"Error in interleaved openAI: {e}. This may due to your invalid API key. Please check the response: {response.json()} ")
        return response.json()