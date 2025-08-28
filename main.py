from fastapi import FastAPI, UploadFile, File
from typing import Optional
import torch
from PIL import Image
import io
import base64
import os
from util.utils import check_ocr_box, get_yolo_model, get_caption_model_processor, get_som_labeled_img

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Initialize models
yolo_model = get_yolo_model(model_path='weights/icon_detect/model.pt')
caption_model_processor = get_caption_model_processor(model_name="florence2", model_name_or_path="weights/icon_caption_florence")
DEVICE = torch.device('cuda')

app = FastAPI(title="OmniParser API")

def process_image(image: Image.Image, box_threshold: float, iou_threshold: float, use_paddleocr: bool, imgsz: int) -> tuple[Image.Image, str]:
    """
    Process the image and return the annotated image and parsed elements.
    """
    box_overlay_ratio = image.size[0] / 3200
    draw_bbox_config = {
        'text_scale': 0.8 * box_overlay_ratio,
        'text_thickness': max(int(2 * box_overlay_ratio), 1),
        'text_padding': max(int(3 * box_overlay_ratio), 1),
        'thickness': max(int(3 * box_overlay_ratio), 1),
    }

    ocr_bbox_rslt, is_goal_filtered = check_ocr_box(image, display_img=False, output_bb_format='xyxy', goal_filtering=None, easyocr_args={'paragraph': False, 'text_threshold': 0.9}, use_paddleocr=use_paddleocr)
    text, ocr_bbox = ocr_bbox_rslt
    dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
        image, yolo_model, BOX_TRESHOLD=box_threshold, output_coord_in_ratio=True, ocr_bbox=ocr_bbox,
        draw_bbox_config=draw_bbox_config, caption_model_processor=caption_model_processor, ocr_text=text,
        iou_threshold=iou_threshold, imgsz=imgsz
    )
    annotated_image = Image.open(io.BytesIO(base64.b64decode(dino_labled_img)))
    parsed_content_str = '\n'.join([f'icon {i}: ' + str(v) for i, v in enumerate(parsed_content_list)])
    return annotated_image, parsed_content_str

@app.post("/process")
async def process_endpoint(
    file: UploadFile = File(...),
    box_threshold: float = 0.05,
    iou_threshold: float = 0.1,
    use_paddleocr: bool = True,
    imgsz: int = 640
):
    """
    Endpoint to upload an image and get parsed elements.
    """
    try:
        # Read the uploaded image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Process the image
        annotated_image, parsed_content = process_image(image, box_threshold, iou_threshold, use_paddleocr, imgsz)

        # Convert annotated image to bytes for return
        img_byte_arr = io.BytesIO()
        annotated_image.save(img_byte_arr, format='PNG')
        img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

        # Return JSON response with image and parsed content
        return {
            "annotated_image": img_base64,
            "parsed_elements": parsed_content
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)