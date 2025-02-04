from groq import Groq
import os
from .utils import is_image_path

def run_groq_interleaved(messages: list, system: str, model_name: str, api_key: str, max_tokens=256, temperature=0.6):
    """
    Run a chat completion through Groq's API, ignoring any images in the messages.
    """
    api_key = api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set")
    
    client = Groq(api_key=api_key)
    # avoid using system messages for R1
    final_messages = [{"role": "user", "content": system}]

    if isinstance(messages, list):
        for item in messages:
            if isinstance(item, dict):
                # For dict items, concatenate all text content, ignoring images
                text_contents = []
                for cnt in item["content"]:
                    if isinstance(cnt, str):
                        if not is_image_path(cnt):  # Skip image paths
                            text_contents.append(cnt)
                    else:
                        text_contents.append(str(cnt))
                
                if text_contents:  # Only add if there's text content
                    message = {"role": "user", "content": " ".join(text_contents)}
                    final_messages.append(message)
            else:  # str
                message = {"role": "user", "content": item}
                final_messages.append(message)
    
    elif isinstance(messages, str):
        final_messages.append({"role": "user", "content": messages})

    try:
        completion = client.chat.completions.create(
            model="deepseek-r1-distill-llama-70b",
            messages=final_messages,
            temperature=0.6,
            max_completion_tokens=max_tokens,
            top_p=0.95,
            stream=False,
            reasoning_format="raw"
        )
        
        response = completion.choices[0].message.content
        final_answer = response.split('</think>\n')[-1] if '</think>' in response else response
        final_answer = final_answer.replace("<output>", "").replace("</output>", "")
        token_usage = completion.usage.total_tokens
        
        return final_answer, token_usage
    except Exception as e:
        print(f"Error in interleaved Groq: {e}")

        return str(e), 0