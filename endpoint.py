import numpy as np
from PIL import Image
import io
import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
from typing import Dict, Tuple, Optional
import os
import logging
import sys
import time
from util.utils import (
    check_ocr_box,
    get_yolo_model,
    get_caption_model_processor,
    get_som_labeled_img,
)

# Configure logging
logging.basicConfig(level=logging.INFO, handlers=[])  # Reset root logger
logger = logging.getLogger("omniparser")
logger.setLevel(logging.INFO)
logger.handlers = []  # Reset any existing handlers

# Create console handler with color formatting
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

# Create formatter with consistent style including timing
formatter = logging.Formatter(
    '\033[1;36m%(asctime)s\033[0m - \033[1;33m%(name)s\033[0m - \033[1;35m%(levelname)s\033[0m - \033[1;32m%(timing)s\033[0m - %(message)s'
)

# Add timing filter
class TimingFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, 'timing'):
            record.timing = ''
        return True

logger.addFilter(TimingFilter())
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.propagate = False  # Prevent propagation to avoid duplicate logs

# Log startup information
logger.info("Initializing OmniParser API")

# Environment configuration with defaults
CONCURRENCY_LIMIT = int(os.environ.get("CONCURRENCY_LIMIT", "1"))
API_PORT = int(os.environ.get("API_PORT", "7861"))

logger.info(f"Environment configuration loaded: CONCURRENCY_LIMIT={CONCURRENCY_LIMIT}, API_PORT={API_PORT}")

def convert_to_pil_image(image_input) -> Image.Image:
    """Convert various image input formats to PIL Image."""
    logger.debug("Converting input to PIL Image")
    try:
        if isinstance(image_input, np.ndarray):
            logger.debug("Converting numpy array to PIL Image")
            return Image.fromarray(image_input)
        elif isinstance(image_input, dict) and "image" in image_input:
            image_data = image_input["image"]
            if isinstance(image_data, str) and image_data.startswith("data:image"):
                logger.debug("Converting base64 string to PIL Image")
                image_data = image_data.split(",")[1]
                return Image.open(io.BytesIO(base64.b64decode(image_data)))
            elif isinstance(image_data, np.ndarray):
                logger.debug("Converting nested numpy array to PIL Image")
                return Image.fromarray(image_data)
            raise ValueError("Unsupported image data format in dictionary")
        elif isinstance(image_input, Image.Image):
            logger.debug("Input is already a PIL Image")
            return image_input
        raise ValueError(f"Unsupported image input format: {type(image_input)}")
    except Exception as e:
        logger.error(f"Failed to convert image: {str(e)}", exc_info=True)
        raise

def get_bbox_config(image: Image.Image) -> Dict:
    """Calculate bounding box overlay configuration based on image size."""
    logger.debug(f"Calculating bbox config for image size: {image.size}")
    box_overlay_ratio = image.size[0] / 3200
    config = {
        "text_scale": 0.8 * box_overlay_ratio,
        "text_thickness": max(int(2 * box_overlay_ratio), 1),
        "text_padding": max(int(3 * box_overlay_ratio), 1),
        "thickness": max(int(3 * box_overlay_ratio), 1),
    }
    logger.debug(f"Generated bbox config: {config}")
    return config

def create_process_fn(yolo_model, caption_model_processor):
    def process(
        image_input, 
        box_threshold: float, 
        iou_threshold: float, 
        use_paddleocr: bool, 
        imgsz: int
    ) -> Tuple[Optional[Image.Image], str]:
        start_time = time.time()
        request_id = f"req_{id(image_input)}"
        logger.info(f"[{request_id}] Starting image processing with parameters: box_threshold={box_threshold}, iou_threshold={iou_threshold}, use_paddleocr={use_paddleocr}, imgsz={imgsz}")
        
        try:
            # Convert input to PIL Image
            t0 = time.time()
            image = convert_to_pil_image(image_input)
            if not isinstance(image, Image.Image):
                logger.error(f"[{request_id}] Image conversion failed")
                raise ValueError("Failed to convert input to PIL Image")
            logger.info(f"[{request_id}] Image conversion completed", extra={'timing': f'+{(time.time() - t0):.2f}s'})

            logger.debug(f"[{request_id}] Input image size: {image.size}")

            # Get bounding box configuration
            draw_bbox_config = get_bbox_config(image)

            # Perform OCR
            t0 = time.time()
            logger.info(f"[{request_id}] Starting OCR processing with {'PaddleOCR' if use_paddleocr else 'EasyOCR'}")
            ocr_bbox_rslt, _ = check_ocr_box(
                image,
                display_img=False,
                output_bb_format="xyxy",
                goal_filtering=None,
                easyocr_args={"paragraph": False, "text_threshold": 0.9},
                use_paddleocr=use_paddleocr,
            )
            text, ocr_bbox = ocr_bbox_rslt
            logger.info(f"[{request_id}] OCR processing completed. Found {len(text)} text elements", extra={'timing': f'+{(time.time() - t0):.2f}s'})

            # Get labeled image and parsed content
            t0 = time.time()
            logger.info(f"[{request_id}] Starting image labeling and content parsing")
            dino_labled_img, _, parsed_content_list = get_som_labeled_img(
                image,
                yolo_model,
                BOX_TRESHOLD=box_threshold,
                output_coord_in_ratio=True,
                ocr_bbox=ocr_bbox,
                draw_bbox_config=draw_bbox_config,
                caption_model_processor=caption_model_processor,
                ocr_text=text,
                iou_threshold=iou_threshold,
                imgsz=imgsz,
            )
            logger.info(f"[{request_id}] Image labeling completed", extra={'timing': f'+{(time.time() - t0):.2f}s'})

            # Process output
            output_image = Image.open(io.BytesIO(base64.b64decode(dino_labled_img)))
            total_time = time.time() - start_time
            logger.info(f"[{request_id}] Image processing completed successfully. Found {len(parsed_content_list)} elements", 
                       extra={'timing': f'total: {total_time:.2f}s'})
            
            parsed_content = "\n".join(
                f"icon {i}: {str(v)}" for i, v in enumerate(parsed_content_list)
            )
            return output_image, parsed_content

        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"[{request_id}] Error processing image: {str(e)}", 
                        extra={'timing': f'failed after {total_time:.2f}s'}, 
                        exc_info=True)
            return None, str(e)

    return process

# Initialize models
logger.info("Initializing YOLO model...")
try:
    yolo_model = get_yolo_model(model_path="weights/icon_detect/model.pt")
    logger.info("YOLO model initialized successfully")
except Exception as e:
    logger.critical("Failed to initialize YOLO model", exc_info=True)
    raise

logger.info("Initializing caption model and processor...")
try:
    caption_model_processor = get_caption_model_processor(
        model_name="florence2", 
        model_name_or_path="weights/icon_caption_florence"
    )
    logger.info("Caption model and processor initialized successfully")
except Exception as e:
    logger.critical("Failed to initialize caption model", exc_info=True)
    raise

process_fn = create_process_fn(yolo_model, caption_model_processor)
logger.info("Process function created successfully")

# Create FastAPI app
app = FastAPI(title="OmniParser API", description="API for processing GUI images")

class ProcessRequest(BaseModel):
    image_data: str = Field(..., description="Base64 encoded image string including data URI prefix")
    box_threshold: float = Field(default=0.05, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.1, ge=0.0, le=1.0)
    use_paddleocr: bool = Field(default=True)
    imgsz: int = Field(default=640, ge=320, le=1920)

@app.post("/process", response_model=Dict[str, str])
async def process_image(req: ProcessRequest):
    """Process an image and return the labeled result with parsed content."""
    request_id = f"req_{id(req)}"
    logger.info(f"[{request_id}] Received image processing request")
    logger.debug(f"[{request_id}] Request parameters: box_threshold={req.box_threshold}, iou_threshold={req.iou_threshold}, imgsz={req.imgsz}, use_paddleocr={req.use_paddleocr}")
    
    image_input = {"image": req.image_data}
    logger.info(f"[{request_id}] Starting image processing")
    processed_image, parsed_content = process_fn(
        image_input,
        req.box_threshold,
        req.iou_threshold,
        req.use_paddleocr,
        req.imgsz
    )
    
    if processed_image is None:
        logger.error(f"[{request_id}] Image processing failed: {parsed_content}")
        raise HTTPException(status_code=400, detail=f"Processing failed: {parsed_content}")
    
    # Convert the processed image to base64
    buffered = io.BytesIO()
    processed_image.save(buffered, format="PNG")
    encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    logger.info(f"[{request_id}] Successfully processed and encoded image")
    
    return {
        "processed_image": encoded_image,
        "parsed_content": parsed_content
    }

if __name__ == "__main__":
    logger.info("=== Starting OmniParser API ===")
    logger.info(f"Configuration: PORT={API_PORT}, CONCURRENCY_LIMIT={CONCURRENCY_LIMIT}")
    uvicorn.run(app, host="0.0.0.0", port=API_PORT, log_level="info")
