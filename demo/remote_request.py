# uvicorn remote_request:app --host 0.0.0.0 --port 8000 --reload

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import get_som_labeled_img, check_ocr_box, get_caption_model_processor, get_yolo_model
import torch
from PIL import Image
from typing import Dict, Tuple, List
import base64


config = {
    'som_model_path': '../weights/icon_detect_v1_5/model_v1_5.pt',
    'device': 'cpu',
    'caption_model_name': 'florence2',
    'caption_model_path': '../weights/icon_caption_florence',
    'BOX_TRESHOLD': 0.05
}


class Omniparser(object):
    def __init__(self, config: Dict):
        self.config = config
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.som_model = get_yolo_model(model_path=config['som_model_path'])
        self.caption_model_processor = get_caption_model_processor(model_name=config['caption_model_name'], model_name_or_path=config['caption_model_path'], device=device)
        print('Omniparser initialized!!!')

    def parse(self, image_base64: str):
        image_path = '../imgs/demo_image.jpg'
        with open(image_path, "wb") as fh:
            fh.write(base64.b64decode(image_base64))
        print('Parsing image:', image_path)

        image = Image.open(image_path)
        print('image size:', image.size)
        
        box_overlay_ratio = max(image.size) / 3200
        draw_bbox_config = {
            'text_scale': 0.8 * box_overlay_ratio,
            'text_thickness': max(int(2 * box_overlay_ratio), 1),
            'text_padding': max(int(3 * box_overlay_ratio), 1),
            'thickness': max(int(3 * box_overlay_ratio), 1),
        }
        BOX_TRESHOLD = config['BOX_TRESHOLD']

        ocr_bbox_rslt, is_goal_filtered = check_ocr_box(image_path, display_img = False, output_bb_format='xyxy', goal_filtering=None, easyocr_args={'paragraph': False, 'text_threshold':0.8}, use_paddleocr=True)
        text, ocr_bbox = ocr_bbox_rslt
        dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(image_path, self.som_model, BOX_TRESHOLD = BOX_TRESHOLD, output_coord_in_ratio=True, ocr_bbox=ocr_bbox,draw_bbox_config=draw_bbox_config, caption_model_processor=self.caption_model_processor, ocr_text=text,use_local_semantics=True, iou_threshold=0.7, scale_img=False, batch_size=128)
        with open('../imgs/demo_image_som.jpg', "wb") as fh:
            fh.write(base64.b64decode(dino_labled_img))

        return dino_labled_img, parsed_content_list
    

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    base64_image: str
    prompt: str

Omniparser = Omniparser(config)

@app.post("/send_text/")
async def send_text(item: Item):
    print('start parsing...')
    import time
    start = time.time()
    dino_labled_img, parsed_content_list = Omniparser.parse(item.base64_image)
    latency = time.time() - start
    print('time:', latency)
    return {"som_image_base64": dino_labled_img, "parsed_content_list": parsed_content_list, 'latency': latency}