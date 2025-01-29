'''
python -m remote_request --som_model_path ../weights/icon_detect_v1_5/model_v1_5.pt --caption_model_name florence2 --caption_model_path ../weights/icon_caption_florence --device cuda --BOX_TRESHOLD 0.05
'''

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
from utils import get_som_labeled_img, get_caption_model_processor, get_yolo_model, check_ocr_box
import torch
from PIL import Image
from typing import Dict, Tuple, List
import base64
import io
from fastapi import FastAPI
from pydantic import BaseModel
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Omniparser API')
    parser.add_argument('--som_model_path', type=str, default='../weights/icon_detect_v1_5/model_v1_5.pt', help='Path to the som model')
    parser.add_argument('--caption_model_name', type=str, default='florence2', help='Name of the caption model')
    parser.add_argument('--caption_model_path', type=str, default='../weights/icon_caption_florence', help='Path to the caption model')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the model')
    parser.add_argument('--BOX_TRESHOLD', type=float, default=0.05, help='Threshold for box detection')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host for the API')
    parser.add_argument('--port', type=int, default=8000, help='Port for the API')
    args = parser.parse_args()
    return args

args = parse_arguments()
config = vars(args)


# config = {
#     'som_model_path': '../weights/icon_detect_v1_5/model_v1_5.pt',
#     'device': 'cpu',
#     'caption_model_name': 'florence2',
#     'caption_model_path': '../weights/icon_caption_florence',
#     'BOX_TRESHOLD': 0.05
# }


class Omniparser(object):
    def __init__(self, config: Dict):
        self.config = config
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.som_model = get_yolo_model(model_path=config['som_model_path'])
        self.caption_model_processor = get_caption_model_processor(model_name=config['caption_model_name'], model_name_or_path=config['caption_model_path'], device=device)
        print('Omniparser initialized!!!')

    def parse(self, image_base64: str):
        # Convert base64 to image directly without saving to disk
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes))
        print('image size:', image.size)
        
        box_overlay_ratio = max(image.size) / 3200
        draw_bbox_config = {
            'text_scale': 0.8 * box_overlay_ratio,
            'text_thickness': max(int(2 * box_overlay_ratio), 1),
            'text_padding': max(int(3 * box_overlay_ratio), 1),
            'thickness': max(int(3 * box_overlay_ratio), 1),
        }
        BOX_TRESHOLD = config['BOX_TRESHOLD']

        (text, ocr_bbox), _ = check_ocr_box(image, display_img=False, output_bb_format='xyxy', easyocr_args={'text_threshold': 0.8}, use_paddleocr=False)
        dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(image, self.som_model, BOX_TRESHOLD = BOX_TRESHOLD, output_coord_in_ratio=True, ocr_bbox=ocr_bbox,draw_bbox_config=draw_bbox_config, caption_model_processor=self.caption_model_processor, ocr_text=text,use_local_semantics=True, iou_threshold=0.7, scale_img=False, batch_size=128)

        return dino_labled_img, parsed_content_list
    



app = FastAPI()

class Item(BaseModel):
    base64_image: str
    prompt: str

Omniparser = Omniparser(config)

@app.post("/send_text/")
async def send_text(item: Item):
    print('start parsing...')
    
    start = time.time()
    dino_labled_img, parsed_content_list = Omniparser.parse(item.base64_image)
    latency = time.time() - start
    print('time:', latency)
    return {"som_image_base64": dino_labled_img, "parsed_content_list": parsed_content_list, 'latency': latency}

@app.get("/")
async def root():
    return {"message": "Omniparser API ready"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("remote_request:app", host=args.host, port=args.port, reload=True)