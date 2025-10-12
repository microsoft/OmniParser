import io
import os
import numpy as np
import time
import torch
import logging
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from util.utils import check_ocr_box, get_yolo_model, get_caption_model_processor, get_som_labeled_img, \
    prepare_structured_results, convert_ocr_results_to_json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Load models and processors with error handling
try:
    yolo_model = get_yolo_model(model_path=os.getenv('YOLO_MODEL_PATH', 'weights/icon_detect/model.pt'))
    caption_model_processor = get_caption_model_processor(
        model_name="florence2", 
        model_name_or_path=os.getenv('CAPTION_MODEL_PATH', 'weights/icon_caption_florence')
    )
except Exception as e:
    logger.error(f"Error loading models: {e}")
    raise

DEVICE = torch.device(os.getenv('DEVICE', 'cuda'))

# Define the processing function
def process_image(
    image_input: Image.Image,
    image_name: str,
    box_threshold: float = 0.05,
    iou_threshold: float = 0.1,
    use_paddleocr: bool = True,
    imgsz: int = 640) -> tuple:
    start_time = time.time()

    try:
        # Save the input image
        image_save_path = 'imgs/saved_image_input.png'
        image_input.save(image_save_path)
        with Image.open(image_save_path) as image:
            box_overlay_ratio = image.size[0] / 3200
            draw_bbox_config = {
                'text_scale': 0.8 * box_overlay_ratio,
                'text_thickness': max(int(2 * box_overlay_ratio), 1),
                'text_padding': max(int(3 * box_overlay_ratio), 1),
                'thickness': max(int(3 * box_overlay_ratio), 1),
            }

            # ocr_bbox_rslt, is_goal_filtered = check_ocr_box(
            #     image_save_path, display_img=False, output_bb_format='xyxy',
            #     goal_filtering=None, easyocr_args={'paragraph': False, 'text_threshold': 0.9},
            #     use_paddleocr=use_paddleocr
            # )
            ocr_bbox_rslt, is_goal_filtered = check_ocr_box(image_save_path, display_img = False, output_bb_format='xyxy', goal_filtering=None, easyocr_args={'paragraph': False, 'text_threshold':0.8}, use_paddleocr=False)

            text, ocr_bbox = ocr_bbox_rslt
            print(text, ocr_bbox, 'text, ocr_bbox')
            # Convert OCR results to JSON format
            ocr_json = convert_ocr_results_to_json(image_name, text, ocr_bbox)

            # dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
            #     image_save_path, yolo_model, BOX_TRESHOLD=box_threshold,
            #     output_coord_in_ratio=True, ocr_bbox=ocr_bbox, draw_bbox_config=draw_bbox_config,
            #     caption_model_processor=caption_model_processor, ocr_text=text,
            #     iou_threshold=iou_threshold, imgsz=imgsz
            # )
            dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(image_save_path, yolo_model, BOX_TRESHOLD = box_threshold, output_coord_in_ratio=True, ocr_bbox=ocr_bbox,draw_bbox_config=draw_bbox_config, caption_model_processor=caption_model_processor, ocr_text=text,use_local_semantics=True, iou_threshold=0.7, scale_img=False, batch_size=128)


            # Use the new function
            structured_results = prepare_structured_results(parsed_content_list, label_coordinates, image_input)

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

    end_time = time.time()
    processing_time = end_time - start_time

    return label_coordinates, parsed_content_list, processing_time, ocr_bbox_rslt, structured_results, ocr_json, dino_labled_img

@app.post("/process")
async def process_route(image: UploadFile = File(...), img_name: str = Form(...)):
    try:
        image_data = await image.read()
        image_input = Image.open(io.BytesIO(image_data))
    except Exception as e:
        logger.error(f"Error reading image: {e}")
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Process the image
    label_coords, content_list, processing_time, ocr_bbox_rslt, structured_results, ocr_json, dino_labeled_img = process_image(
        image_input=image_input,
        image_name=img_name,
        box_threshold=0.05,
        iou_threshold=0.1,
        use_paddleocr=True,
        imgsz=640,
    )

    # Convert any ndarray objects to lists
    label_coords = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in label_coords.items()}

    print(structured_results, 'structured_results')

    # Return the results as JSON
    return JSONResponse(content={
        "based64_image": dino_labeled_img,
        "ocr_json": ocr_json,
        "structured_results": structured_results
    })

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=int(os.getenv('PORT', 5003)))
