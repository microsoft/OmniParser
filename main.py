from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from typing import Optional
import torch
from PIL import Image
import io
import base64
import os
from util.utils import check_ocr_box, get_yolo_model, get_caption_model_processor, get_som_labeled_img
import logging

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Initialize models
logger.info("Initializing models...")
yolo_model = get_yolo_model(model_path='weights/icon_detect/model.pt')
caption_model_processor = get_caption_model_processor(model_name="florence2", model_name_or_path="weights/icon_caption_florence")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Models initialized. Using device: {DEVICE}")

app = FastAPI(title="OmniParser API")

def process_image(image: Image.Image, box_threshold: float, iou_threshold: float, 
      use_paddleocr: bool, imgsz: int) -> str:
    logger.info(f"Processing image with params: box_threshold={box_threshold}, iou_threshold={iou_threshold}, use_paddleocr={use_paddleocr}, imgsz={imgsz}")
    ocr_bbox_rslt, _ = check_ocr_box(image, display_img=False, output_bb_format='xyxy', goal_filtering=None, easyocr_args={'paragraph': False, 'text_threshold': 0.9}, use_paddleocr=use_paddleocr)
    text, ocr_bbox = ocr_bbox_rslt
    _, _, parsed_content_list = get_som_labeled_img(
        image, yolo_model, BOX_TRESHOLD=box_threshold, output_coord_in_ratio=True,
        ocr_bbox=ocr_bbox, draw_bbox_config={}, caption_model_processor=caption_model_processor,
        ocr_text=text, iou_threshold=iou_threshold, imgsz=imgsz
    )
    return '\n'.join([f'icon {i}: ' + str(v) for i, v in enumerate(parsed_content_list)])

@app.post("/process")
async def process_endpoint(
    file: UploadFile = File(...),
    box_threshold: float = Form(...),
    iou_threshold: float = Form(...),
    use_paddleocr: str = Form(...),
    imgsz: int = Form(...)
):
    use_paddleocr_bool = use_paddleocr.lower() in ('true', '1')
    parsed_content = process_image(Image.open(file.file).convert("RGB"), box_threshold, iou_threshold, use_paddleocr_bool, imgsz)
    return { "parsed_elements": parsed_content }

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting OmniParser API server for local testing...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
