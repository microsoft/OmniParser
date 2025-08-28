import os
import sys
import io
import traceback
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse

# Add the parent directory of 'util' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from util.omniparser import get_som_labeled_img, initialize_models
from util.utils import check_ocr_box

# --- Initialize Models ---
# It's better to initialize models once when the application starts
yolo_model, caption_model_processor = initialize_models()

app = FastAPI()

def process_image_with_logging(image, box_threshold, iou_threshold, use_paddleocr, imgsz):
    """
    This function contains the core image processing logic.
    It's designed to be testable and to log every step.
    """
    log = ["--- Inside process_image_with_logging ---"]
    try:
        log.append(f"Step 1: Received parameters: box_threshold={box_threshold}, iou_threshold={iou_threshold}, use_paddleocr={use_paddleocr}, imgsz={imgsz}")

        # --- OCR Check Case ---
        log.append("Step 2: Starting OCR check...")
        ocr_bbox_rslt, is_goal_filtered = check_ocr_box(image, display_img=False,
          output_bb_format='xyxy', goal_filtering=None, easyocr_args={'paragraph': False,
          'text_threshold': 0.9}, use_paddleocr=use_paddleocr)
        text, ocr_bbox = ocr_bbox_rslt
        log.append(f"Step 3: OCR check complete. Found {len(text)} text elements.")

        # --- Main Model Case ---
        log.append("Step 4: Starting main model (get_som_labeled_img)...")
        _, _, parsed_content_list = get_som_labeled_img(
            image, yolo_model, BOX_TRESHOLD=box_threshold, output_coord_in_ratio=True,
            ocr_bbox=ocr_bbox, draw_bbox_config={}, # draw_bbox_config is not needed for data-only output
            caption_model_processor=caption_model_processor, ocr_text=text,
            iou_threshold=iou_threshold, imgsz=imgz
        )
        log.append(f"Step 5: Main model finished. It returned a list with {len(parsed_content_list)} elements.")

        # --- Final Formatting Case ---
        if not parsed_content_list:
            log.append("WARNING: The final list of elements is empty. The model did not detect any objects that met the criteria.")

        parsed_content_str = '\n'.join([f'icon {i}: ' + str(v) for i, v in enumerate(parsed_content_list)])
        log.append("Step 6: Successfully formatted the final list into a string.")

        return parsed_content_str, log

    except Exception as e:
        log.append(f"\n!!!!!! AN ERROR OCCURRED INSIDE process_image_with_logging !!!!!!")
        log.append(f"Error Type: {type(e)}")
        log.append(f"Error Message: {str(e)}")
        log.append("--- Full Traceback ---")
        log.append(traceback.format_exc())
        return "", log # Return an empty string but the full log of the crash

@app.post("/process")
async def process_endpoint(
    file: UploadFile = File(...),
    box_threshold: float = Form(...),
    iou_threshold: float = Form(...),
    use_paddleocr: bool = Form(...),
    imgsz: int = Form(...)
):
    """
    This endpoint now acts as a "black box recorder".
    It logs everything and returns the log in the response.
    """
    master_log = ["--- Server received request in process_endpoint ---"]
    try:
        # --- Parameter Reception Case ---
        master_log.append(f"Received box_threshold: {box_threshold} (type: {type(box_threshold)})")
        master_log.append(f"Received iou_threshold: {iou_threshold} (type: {type(iou_threshold)})")
        master_log.append(f"Received use_paddleocr: {use_paddleocr} (type: {type(use_paddleocr)})")
        master_log.append(f"Received imgsz: {imgsz} (type: {type(imgsz)})")

        # --- Image Loading Case ---
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        master_log.append(f"Image successfully loaded. Size: {image.size}")

        # --- Call Core Logic Case ---
        parsed_content, function_log = process_image_with_logging(image, box_threshold, iou_threshold, use_paddleocr, imgsz)
        master_log.extend(function_log) # Add the detailed log from the function

        master_log.append("\n--- FINAL SUMMARY ---")
        master_log.append(f"Final content string length: {len(parsed_content)}")

        # We will return the log itself for debugging
        return { "debug_log": "\n".join(master_log), "parsed_elements": parsed_content }

    except Exception as e:
        master_log.append(f"\n!!!!!! AN UNEXPECTED ERROR OCCURRED IN process_endpoint !!!!!!")
        master_log.append(f"Error Type: {type(e)}")
        master_log.append(f"Error Message: {str(e)}")
        master_log.append("--- Full Traceback ---")
        master_log.append(traceback.format_exc())

        return { "debug_log": "\n".join(master_log), "parsed_elements": "" }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
