import os
import re
import ast
import base64
from io import BytesIO
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

import openai
from openai import BadRequestError

model_name = "gpt-4o-2024-05-13"
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")

def convert_pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


from models.utils import get_som_labeled_img, check_ocr_box, get_caption_model_processor, get_yolo_model
import torch
from ultralytics import YOLO
from PIL import Image
device = 'cuda' if torch.cuda.is_available() else 'cpu'
SOM_MODEL_PATH='...'
CAPTION_MODEL_PATH='...'
som_model = get_yolo_model(SOM_MODEL_PATH)

som_model.to(device)
print('model to {}'.format(device))

# two choices for caption model: fine-tuned blip2 or florence2
import importlib
caption_model_processor = get_caption_model_processor(model_name="florence2", model_name_or_path="CAPTION_MODEL_PATH", device=device)

def omniparser_parse(image, image_path):
    box_overlay_ratio = max(image.size) / 3200
    draw_bbox_config = {
        'text_scale': 0.8 * box_overlay_ratio,
        'text_thickness': max(int(2 * box_overlay_ratio), 1),
        'text_padding': max(int(3 * box_overlay_ratio), 1),
        'thickness': max(int(3 * box_overlay_ratio), 1),
    }
    BOX_TRESHOLD = 0.05

    ocr_bbox_rslt, is_goal_filtered = check_ocr_box(image_path, display_img = False, output_bb_format='xyxy', goal_filtering=None, easyocr_args={'paragraph': False, 'text_threshold':0.5, 'canvas_size':max(image.size), 'decoder':'beamsearch', 'beamWidth':10, 'batch_size':256}, use_paddleocr=False)
    text, ocr_bbox = ocr_bbox_rslt

    dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(image_path, som_model, BOX_TRESHOLD = BOX_TRESHOLD, output_coord_in_ratio=True, ocr_bbox=ocr_bbox,draw_bbox_config=draw_bbox_config, caption_model_processor=caption_model_processor, ocr_text=text,use_local_semantics=True, iou_threshold=0.7, scale_img=False, batch_size=128)
    return dino_labled_img, label_coordinates, parsed_content_list
    
def reformat_messages(parsed_content_list):
    screen_info = ""
    for idx, element in enumerate(parsed_content_list):
        element['idx'] = idx
        if element['type'] == 'text':
            screen_info += f'''<p id={idx} class="text" alt="{element['content']}"> </p>\n'''
            # screen_info += f'ID: {idx}, Text: {element["content"]}\n'
        elif element['type'] == 'icon':
            screen_info += f'''<img id={idx} class="icon" alt="{element['content']}"> </img>\n'''
            # screen_info += f'ID: {idx}, Icon: {element["content"]}\n'
    return screen_info

PROMPT_TEMPLATE_SEECLICK_PARSED_CONTENT = '''Please generate the next move according to the UI screenshot and task instruction. You will be presented with a screenshot image. Also you will be given each bounding box's description in a list. To complete the task, You should choose a related bbox to click based on the bbox descriptions. 
Task instruction: {}. 
Here is the list of all detected bounding boxes by IDs and their descriptions: {}. Keep in mind the description for Text Boxes are likely more accurate than the description for Icon Boxes. 
Requirement: 1. You should first give a reasonable description of the current screenshot, and give a short analysis of how can the user task be achieved. 2. Then make an educated guess of bbox id to click in order to complete the task based on the bounding boxes descriptions. 3. Your answer should follow the following format: {{"Analysis": xxx, "Click BBox ID": "y"}}. Do not include any other info. Some examples: {}. The task is to {}. Retrieve the bbox id where its description matches the task instruction. Now start your answer:'''

# PROMPT_TEMPLATE_SEECLICK_PARSED_CONTENT_v1 = "The instruction is to {}. \nHere is the list of all detected bounding boxes by IDs and their descriptions: {}. \nKeep in mind the description for Text Boxes are likely more accurate than the description for Icon Boxes. \n Requirement: 1. You should first give a reasonable description of the current screenshot, and give a step by step analysis of how can the user task be achieved. 2. Then make an educated guess of bbox id to click in order to complete the task using both the visual information from the screenshot image and the bounding boxes descriptions. 3. Your answer should follow the following format: {{'Analysis': 'xxx', 'Click BBox ID': 'y'}}. Please do not include any other info."
PROMPT_TEMPLATE_SEECLICK_PARSED_CONTENT_v1 = "The instruction is to {}. \nHere is the list of all detected bounding boxes by IDs and their descriptions: {}. \nKeep in mind the description for Text Boxes are likely more accurate than the description for Icon Boxes. \n Requirement: 1. You should first give a reasonable description of the current screenshot, and give a some analysis of how can the user instruction be achieved by a single click. 2. Then make an educated guess of bbox id to click in order to complete the task using both the visual information from the screenshot image and the bounding boxes descriptions. REMEMBER: the task instruction must be achieved by one single click. 3. Your answer should follow the following format: {{'Analysis': 'xxx', 'Click BBox ID': 'y'}}. Please do not include any other info."


FEWSHOT_EXAMPLE = '''Example 1: Task instruction: Next page. \n{"Analysis": "Based on the screenshot and icon descriptions, I should click on the next page icon, which is labeled with box ID x in the bounding box list", "Click BBox ID": "x"}\n\n
Example 2: Task instruction: Search on google. \n{"Analysis": "Based on the screenshot and icon descriptions, I should click on the 'Search' box, which is labeled with box ID y in the bounding box list", "Click BBox ID": "y"}'''




from azure.identity import AzureCliCredential, DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI
from models.utils import get_pred_phi3v, extract_dict_from_text, get_phi3v_model_dict

class GPT4XModel():
    def __init__(self, model_name="gpt-4o-2024-05-13", use_managed_identity=False):
        self.client = openai.OpenAI(
            api_key=OPENAI_KEY,
        )
        self.model_name = model_name
        if model_name == 'phi35v':
            self.model_dict = get_phi3v_model_dict()

    def load_model(self):
        pass
    
    def set_generation_config(self, **kwargs):
        self.override_generation_config.update(kwargs)

    def ground_only_positive_phi35v(self, instruction, image):
        if isinstance(image, str):
            image_path = image
            assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."
            image = Image.open(image_path).convert('RGB')
        assert isinstance(image, Image.Image), "Invalid input image."
        
        base64_image = convert_pil_image_to_base64(image)
        dino_labled_img, label_coordinates, parsed_content_list = omniparser_parse(image, image_path)
        screen_info = reformat_messages(parsed_content_list)
        prompt_origin = PROMPT_TEMPLATE_SEECLICK_PARSED_CONTENT.format(instruction, screen_info, FEWSHOT_EXAMPLE, instruction)
        # prompt_origin = PROMPT_TEMPLATE_SEECLICK_PARSED_CONTENT_v1.format(instruction, screen_info)

        # Use the get_pred_phi3v function to get predictions
        icon_id, bbox, click_point, response_text = get_pred_phi3v(prompt_origin, (base64_image, dino_labled_img), label_coordinates, id_key='Click ID', model_dict=self.model_dict)

        result_dict = {
            "result": "positive",
            "bbox": bbox,
            "point": click_point,
            "raw_response": response_text,
            'dino_labled_img': dino_labled_img,
            'screen_info': screen_info,
        }
        
        return result_dict

    def ground_only_positive(self, instruction, image):
        if isinstance(image, str):
            image_path = image
            assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."
            image = Image.open(image_path).convert('RGB')
        assert isinstance(image, Image.Image), "Invalid input image."
        
        base64_image = convert_pil_image_to_base64(image)
        dino_labled_img, label_coordinates, parsed_content_list = omniparser_parse(image, image_path)
        screen_info = reformat_messages(parsed_content_list)
        # prompt_origin = PROMPT_TEMPLATE_SEECLICK_PARSED_CONTENT.format(screen_info, FEWSHOT_EXAMPLE, instruction)
        prompt_origin = PROMPT_TEMPLATE_SEECLICK_PARSED_CONTENT_v1.format(instruction, screen_info)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": [
                            # {"type": "text", "text": "You are an expert in using electronic devices and interacting with graphic interfaces. You should not call any external tools."}
                            {"type": "text", "text": '''You are an expert at completing instructions on GUI screens. 
               You will be presented with two images. The first is the original screenshot. The second is the same screenshot with some numeric tags. You will also be provided with some descriptions of the bbox, and your task is to choose the numeric bbox idx you want to click in order to complete the user instruction.'''}
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text", 
                                "text": prompt_origin

                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}",
                                }
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{dino_labled_img}",
                                }
                            },
                        ],
                    }
                ],
                temperature=self.override_generation_config['temperature'],
                max_tokens=2048,
            )
            response_text = response.choices[0].message.content
        except BadRequestError as e:
            print("OpenAI BadRequestError:", e)
            return None

        # Extract bounding box
        # print("------")
        # print(grounding_prompt)
        print("------")
        print(response_text)
        # print("------")
        # Try getting groundings
        # bbox = extract_first_bounding_box(response_text)
        # click_point = extract_first_point(response_text)
        
        # if not click_point and bbox:
        #     click_point = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        response_text = response_text.replace('```json', '').replace('```', '') #TODO: fix this

        try:
            response_text = ast.literal_eval(response_text)
            
            icon_id = response_text['Click BBox ID']
            bbox = label_coordinates[str(icon_id)]
            click_point = [bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2]
        except:
            print('error parsing, use regex to parse!!!')
            response_text = extract_dict_from_text(response_text)
            icon_id = response_text['Click BBox ID']
            bbox = label_coordinates[str(icon_id)]
            click_point = [bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2]

        result_dict = {
            "result": "positive",
            "bbox": bbox,
            "point": click_point,
            "raw_response": response_text,
            'dino_labled_img': dino_labled_img,
            'screen_info': screen_info,
        }
        
        return result_dict

    def ground_allow_negative(self, instruction, image=None):
        if isinstance(image, str):
            image_path = image
            assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."
            image = Image.open(image_path).convert('RGB')
        assert isinstance(image, Image.Image), "Invalid input image."
        
        base64_image = convert_pil_image_to_base64(image)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": "You are an expert in using electronic devices and interacting with graphic interfaces. You should not call any external tools."}
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}",
                                }
                            },
                            {
                                "type": "text", 
                                "text": "You are asked to find the bounding box of an UI element in the given screenshot corresponding to a given instruction.\n"
                                        "Don't output any analysis. Output your result in the format of [[x0,y0,x1,y1]], with x and y ranging from 0 to 1. \n"
                                        "If such element does not exist, output only the text 'Target not existent'.\n"
                                        "The instruction is:\n"
                                        f"{instruction}\n"
                            }
                        ],
                    }
                ],
                temperature=self.override_generation_config['temperature'],
                max_tokens=2048,
            )
            response_text = response.choices[0].message.content
        except BadRequestError as e:
            print("OpenAI BadRequestError:", e)
            return {
                "result": "failed"
            }

        # Extract bounding box
        # print("------")
        # print(grounding_prompt)
        print("------")
        print(response_text)
        # print("------")

        if "not existent" in response_text.lower():
            return {
                "result": "negative",
                "bbox": None,
                "point": None,
                "raw_response": response_text
            }
        
        # Try getting groundings
        bbox = extract_first_bounding_box(response_text)
        click_point = extract_first_point(response_text)
        
        if not click_point and bbox:
            click_point = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]

        result_dict = {
            "result": "positive" if bbox or click_point else "negative",
            "bbox": bbox,
            "point": click_point,
            "raw_response": response_text
        }
        
        return result_dict

    
    def ground_with_uncertainty(self, instruction, image=None):
        if isinstance(image, str):
            image_path = image
            assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."
            image = Image.open(image_path).convert('RGB')
        assert isinstance(image, Image.Image), "Invalid input image."
        
        base64_image = convert_pil_image_to_base64(image)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": "You are an expert in using electronic devices and interacting with graphic interfaces. You should not call any external tools."}
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}",
                                }
                            },
                            {
                                "type": "text", 
                                "text": "You are asked to find the bounding box of an UI element in the given screenshot corresponding to a given instruction.\n"
                                        "- If such element does not exist in the screenshot, output only the text 'Target not existent'."

                                        "- If you are sure such element exists and you are confident in finding it, output your result in the format of [[x0,y0,x1,y1]], with x and y ranging from 0 to 1. \n"
                                        "Please find out the bounding box of the UI element corresponding to the following instruction: \n"
                                        "The instruction is:\n"
                                        f"{instruction}\n"
                                        
                            }
                        ],
                    }
                ],
                temperature=self.override_generation_config['temperature'],
                max_tokens=2048,
            )
            response_text = response.choices[0].message.content
        except BadRequestError as e:
            print("OpenAI BadRequestError:", e)
            return {
                "result": "failed"
            }

        # Extract bounding box
        # print("------")
        # print(grounding_prompt)
        print("------")
        print(response_text)
        # print("------")

        if "not found" in response_text.lower():
            return {
                "result": "negative",
                "bbox": None,
                "point": None,
                "raw_response": response_text
            }
        
        # Try getting groundings
        bbox = extract_first_bounding_box(response_text)
        click_point = extract_first_point(response_text)
        
        if not click_point and bbox:
            click_point = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]

        result_dict = {
            "result": "positive",
            "bbox": bbox,
            "point": click_point,
            "raw_response": response_text
        }
        
        return result_dict

def extract_first_bounding_box(text):
    # Regular expression pattern to match the first bounding box in the format [[x0,y0,x1,y1]]
    # This captures the entire float value using \d for digits and optional decimal points
    pattern = r"\[\[(\d+\.\d+|\d+),(\d+\.\d+|\d+),(\d+\.\d+|\d+),(\d+\.\d+|\d+)\]\]"
    
    # Search for the first match in the text
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        # Capture the bounding box coordinates as floats
        bbox = [float(match.group(1)), float(match.group(2)), float(match.group(3)), float(match.group(4))]
        return bbox
    return None


def extract_first_point(text):
    # Regular expression pattern to match the first point in the format [[x0,y0]]
    # This captures the entire float value using \d for digits and optional decimal points
    pattern = r"\[\[(\d+\.\d+|\d+),(\d+\.\d+|\d+)\]\]"
    
    # Search for the first match in the text
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        point = [float(match.group(1)), float(match.group(2))]
        return point
    
    return None
