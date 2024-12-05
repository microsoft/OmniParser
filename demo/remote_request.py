# uvicorn test:app --host 0.0.0.0 --port 8000 --reload
from fastapi import FastAPI
from pydantic import BaseModel

print('hello')
app = FastAPI()

class Item(BaseModel):
    base64_image: str
    prompt: str
    # goal: str
    # idx: int
    # step_logs: str

@app.post("/send_text/")
async def send_text(item: Item):
    print(item.prompt)
    # print(item.base64_image)
    return {"text": item.prompt}



from utils import get_som_labeled_img, check_ocr_box, get_caption_model_processor,  get_dino_model, get_yolo_model
import torch
from ultralytics import YOLO
from PIL import Image
from typing import Dict, Tuple, List
import io
import base64


config = {
    'som_model_path': 'finetuned_icon_detect.pt',
    'device': 'cpu',
    'caption_model_path': 'Salesforce/blip2-opt-2.7b',
    'draw_bbox_config': {
        'text_scale': 0.8,
        'text_thickness': 2,
        'text_padding': 3,
        'thickness': 3,
    },
    'BOX_TRESHOLD': 0.05
}


class Omniparser(object):
    def __init__(self, config: Dict):
        self.config = config
        
        self.som_model = get_yolo_model(model_path=config['som_model_path'])
        device='gpu' if torch.cuda.is_available() else 'cpu'
        self.caption_model_processor = get_caption_model_processor(model_name="florence2", model_name_or_path="weights/icon_caption_florence", device=device)

    def parse(self, image_path: str):
        print('Parsing image:', image_path)
        ocr_bbox_rslt, is_goal_filtered = check_ocr_box(image_path, display_img = False, output_bb_format='xyxy', goal_filtering=None, easyocr_args={'paragraph': False, 'text_threshold':0.5}, use_paddleocr=True)
        text, ocr_bbox = ocr_bbox_rslt
        dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(image_path, self.som_model, BOX_TRESHOLD = BOX_TRESHOLD, output_coord_in_ratio=True, ocr_bbox=ocr_bbox,draw_bbox_config=draw_bbox_config, caption_model_processor=self.caption_model_processor, ocr_text=text,use_local_semantics=True, iou_threshold=0.7, scale_img=False, batch_size=128)

        image = Image.open(io.BytesIO(base64.b64decode(dino_labled_img)))
        # formating output
        return_list = [{'from': 'omniparser', 'shape': {'x':coord[0], 'y':coord[1], 'width':coord[2], 'height':coord[3]},
                        'text': parsed_content_list[i].split(': ')[1], 'type':'text'} for i, (k, coord) in enumerate(label_coordinates.items()) if i < len(parsed_content_list)]
        return_list.extend(
            [{'from': 'omniparser', 'shape': {'x':coord[0], 'y':coord[1], 'width':coord[2], 'height':coord[3]},
                        'text': 'None', 'type':'icon'} for i, (k, coord) in enumerate(label_coordinates.items()) if i >= len(parsed_content_list)]
              )

        return [image, return_list]