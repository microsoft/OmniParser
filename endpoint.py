import modal
import numpy as np
from PIL import Image
import io
import base64
import os
import logging
import sys
import time
from typing import Dict, List, Optional, Any
import json
import uuid

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field, validator

from util.utils import (
    check_ocr_box,
    get_yolo_model,
    get_caption_model_processor,
    get_som_labeled_img,
)

# Default configuration values
DEFAULT_CONCURRENCY_LIMIT = 1
DEFAULT_CONTAINER_TIMEOUT = 300
DEFAULT_GPU_CONFIG = "A100"
DEFAULT_API_PORT = 7861
DEFAULT_MAX_CONTAINERS = 10

# Default request parameters
DEFAULT_BOX_THRESHOLD = 0.05
DEFAULT_IOU_THRESHOLD = 0.1
DEFAULT_USE_PADDLEOCR = True
DEFAULT_IMGSZ = 640

# Environment configuration with defaults
ENV_CONFIG = {
    "CONCURRENCY_LIMIT": int(
        os.environ.get("CONCURRENCY_LIMIT", str(DEFAULT_CONCURRENCY_LIMIT))
    ),
    "MODAL_CONTAINER_TIMEOUT": int(
        os.environ.get("MODAL_CONTAINER_TIMEOUT", str(DEFAULT_CONTAINER_TIMEOUT))
    ),
    "MODAL_GPU_CONFIG": os.environ.get("MODAL_GPU_CONFIG", DEFAULT_GPU_CONFIG),
    "API_PORT": int(os.environ.get("API_PORT", str(DEFAULT_API_PORT))),
    "MAX_CONTAINERS": int(
        os.environ.get("MAX_CONTAINERS", str(DEFAULT_MAX_CONTAINERS))
    ),
}


def setup_logging():
    """Configure and return a logger with custom formatting and stream handlers."""
    logger = logging.getLogger("omniparser")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Simple formatter for regular logs
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.propagate = False

    return logger


logger = setup_logging()
logger.info(f"Environment configuration loaded: {ENV_CONFIG}")


class RequestLogger:
    """Simplified logging helper for tracking request processing metrics."""

    def __init__(self, request_id, endpoint):
        self.start_time = time.time()
        self.request_id = request_id
        self.endpoint = endpoint
        self.timings = {}
        self.current_step = None
        self.step_start_time = None

    def start_step(self, step_name):
        """Start timing a new processing step."""
        self.current_step = step_name
        self.step_start_time = time.time()
        logger.debug(f"[{self.request_id}] Step '{step_name}' started")
        return self

    def end_step(self):
        """End timing for the current step and record its duration."""
        if self.current_step and self.step_start_time:
            duration = time.time() - self.step_start_time
            self.timings[self.current_step] = round(duration, 3)
            step_name = self.current_step  # Save before clearing
            self.current_step = None
            self.step_start_time = None
            logger.debug(
                f"[{self.request_id}] Step '{step_name}' completed in {duration:.3f}s"
            )
        return self

    def log_completion(self, success=True, error=None, metadata=None):
        """Log completion of request processing with metrics."""
        total_duration = time.time() - self.start_time

        if success:
            status_msg = "successfully"
            log_fn = logger.info
        else:
            status_msg = f"with error: {error}"
            log_fn = logger.error

        # Include any provided metadata in the log
        meta_str = f" | {metadata}" if metadata else ""

        log_fn(
            f"[{self.request_id}] Request to '{self.endpoint}' completed {status_msg} "
            f"in {total_duration:.3f}s | Steps: {json.dumps(self.timings)}{meta_str}"
        )


class ImageProcessor:
    """Utility class for image processing operations."""

    @staticmethod
    def convert_to_pil_image(image_input) -> Image.Image:
        """
        Convert various image input formats to PIL Image.

        Args:
            image_input: Input image in various formats (numpy array, base64 string, PIL Image)

        Returns:
            PIL.Image.Image: Converted PIL image

        Raises:
            ValueError: If the image format is unsupported
        """
        try:
            if isinstance(image_input, np.ndarray):
                return Image.fromarray(image_input)
            elif isinstance(image_input, dict) and "image" in image_input:
                image_data = image_input["image"]
                if isinstance(image_data, str) and image_data.startswith("data:image"):
                    image_data = image_data.split(",")[1]
                    return Image.open(io.BytesIO(base64.b64decode(image_data)))
                elif isinstance(image_data, np.ndarray):
                    return Image.fromarray(image_data)
                raise ValueError("Unsupported image data format in dictionary")
            elif isinstance(image_input, Image.Image):
                return image_input
            elif isinstance(image_input, str) and image_input.startswith("data:image"):
                image_data = image_input.split(",")[1]
                return Image.open(io.BytesIO(base64.b64decode(image_data)))
            elif isinstance(image_input, bytes):
                return Image.open(io.BytesIO(image_input))

            raise ValueError(f"Unsupported image input format: {type(image_input)}")
        except Exception as e:
            logger.error(f"Image conversion error: {str(e)}")
            raise ValueError(f"Image conversion error: {str(e)}") from e

    @staticmethod
    def get_bbox_config(image: Image.Image) -> Dict:
        """
        Calculate bounding box overlay configuration based on image size.

        Args:
            image: PIL Image

        Returns:
            Dict: Configuration for bounding box drawing
        """
        box_overlay_ratio = image.size[0] / 3200
        return {
            "text_scale": 0.8 * box_overlay_ratio,
            "text_thickness": max(int(2 * box_overlay_ratio), 1),
            "text_padding": max(int(3 * box_overlay_ratio), 1),
            "thickness": max(int(3 * box_overlay_ratio), 1),
        }


class ProcessRequest(BaseModel):
    """Request model for processing a single image."""

    image_data: str = Field(
        ..., description="Base64 encoded image string including data URI prefix"
    )
    box_threshold: float = Field(default=DEFAULT_BOX_THRESHOLD, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=DEFAULT_IOU_THRESHOLD, ge=0.0, le=1.0)
    use_paddleocr: bool = Field(default=DEFAULT_USE_PADDLEOCR)
    imgsz: int = Field(default=DEFAULT_IMGSZ, ge=320, le=1920)

    @validator("image_data")
    def validate_image_data(cls, v):
        if not v.startswith("data:image"):
            raise ValueError("Image data must begin with 'data:image' prefix")
        return v


class BatchProcessRequest(BaseModel):
    """Request model for processing multiple images in a batch."""

    images: List[str] = Field(
        ...,
        description="Array of base64 encoded image strings including data URI prefix",
    )
    box_threshold: float = Field(default=DEFAULT_BOX_THRESHOLD, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=DEFAULT_IOU_THRESHOLD, ge=0.0, le=1.0)
    use_paddleocr: bool = Field(default=DEFAULT_USE_PADDLEOCR)
    imgsz: int = Field(default=DEFAULT_IMGSZ, ge=320, le=1920)

    @validator("images")
    def validate_images(cls, v):
        if not v:
            raise ValueError("At least one image must be provided")
        for img in v:
            if not img.startswith("data:image"):
                raise ValueError("All image data must begin with 'data:image' prefix")
        return v


class ProcessResult(BaseModel):
    """Model for the result of image processing."""

    processed_image: str = Field(..., description="Base64 encoded processed image")
    parsed_content: str = Field(
        ..., description="Textual representation of parsed content"
    )
    error: Optional[str] = Field(
        default=None, description="Error message if processing failed"
    )


class OmniParser:
    """Main class for parsing images and extracting information from UI elements."""

    def __init__(self):
        """Initialize the OmniParser with null models (to be loaded later)"""
        self.yolo_model = None
        self.caption_model_processor = None
        self.models_initialized = False

    def init_models(self):
        """Initialize and load the ML models."""
        if self.models_initialized:
            return

        try:
            # Configure PyTorch for better performance
            import torch

            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

            # Load models
            self.yolo_model = get_yolo_model(model_path="weights/icon_detect/model.pt")
            self.caption_model_processor = get_caption_model_processor(
                model_name="florence2",
                model_name_or_path="weights/icon_caption_florence",
            )
            self.models_initialized = True
            logger.info("OmniParser models initialized successfully")
        except Exception as e:
            logger.critical(f"Failed to initialize models: {str(e)}")
            raise RuntimeError(f"Model initialization failed: {str(e)}") from e

    def process_image(
        self,
        image_data: str,
        box_threshold: float = DEFAULT_BOX_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        use_paddleocr: bool = DEFAULT_USE_PADDLEOCR,
        imgsz: int = DEFAULT_IMGSZ,
    ) -> Dict[str, Any]:
        """
        Process an image to detect and parse UI elements.

        Args:
            image_data: Base64 encoded image string
            box_threshold: Confidence threshold for bounding boxes
            iou_threshold: IOU threshold for non-maximum suppression
            use_paddleocr: Whether to use PaddleOCR for text detection
            imgsz: Image size for processing

        Returns:
            Dict containing processed image and parsed content
        """
        request_id = f"req_{str(uuid.uuid4())[:8]}"
        request_log = RequestLogger(request_id=request_id, endpoint="process_image")

        # Ensure models are initialized
        if not self.models_initialized:
            self.init_models()

        try:
            # Convert and process image
            request_log.start_step("image_conversion")
            image = ImageProcessor.convert_to_pil_image({"image": image_data})
            draw_bbox_config = ImageProcessor.get_bbox_config(image)
            request_log.end_step()

            # Perform OCR
            request_log.start_step("ocr_processing")
            ocr_bbox_rslt, is_goal_filtered = check_ocr_box(
                image,
                display_img=False,
                output_bb_format="xyxy",
                goal_filtering=None,
                easyocr_args={"paragraph": False, "text_threshold": 0.9},
                use_paddleocr=use_paddleocr,
            )
            text, ocr_bbox = ocr_bbox_rslt
            request_log.end_step()

            # Process image with ML models
            request_log.start_step("icon_detection")
            dino_labled_img, _, parsed_content_list = get_som_labeled_img(
                image,
                self.yolo_model,
                BOX_TRESHOLD=box_threshold,
                output_coord_in_ratio=True,
                ocr_bbox=ocr_bbox,
                draw_bbox_config=draw_bbox_config,
                caption_model_processor=self.caption_model_processor,
                ocr_text=text,
                iou_threshold=iou_threshold,
                imgsz=imgsz,
            )
            request_log.end_step()

            # Prepare response
            request_log.start_step("response_preparation")
            output_image = Image.open(io.BytesIO(base64.b64decode(dino_labled_img)))
            buffered = io.BytesIO()
            output_image.save(buffered, format="PNG")
            encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
            parsed_content = "\n".join(
                f"icon {i}: {str(v)}" for i, v in enumerate(parsed_content_list)
            )

            result = {
                "processed_image": encoded_image,
                "parsed_content": parsed_content,
            }
            request_log.end_step()

            # Log completion
            request_log.log_completion(
                metadata={
                    "image_width": image.width,
                    "image_height": image.height,
                    "text_elements": len(text),
                    "icons_detected": len(parsed_content_list),
                }
            )

            return result

        except Exception as e:
            # Log error and return error response
            if request_log.current_step:
                request_log.end_step()

            error_msg = f"Processing failed: {str(e)}"
            request_log.log_completion(success=False, error=error_msg)

            return {
                "processed_image": "",
                "parsed_content": "",
                "error": error_msg,
            }


def create_modal_image():
    """Create and configure the Modal image with all dependencies."""
    return (
        modal.Image.debian_slim()
        .apt_install("libgl1-mesa-glx", "libglib2.0-0")
        .pip_install(
            "torch",
            "torchvision",
            "transformers",
            "accelerate",
            "timm",
            "einops==0.8.0",
            "easyocr",
            "supervision==0.18.0",
            "ultralytics==8.3.70",
            "opencv-python",
            "opencv-python-headless",
            "paddlepaddle",
            "paddleocr",
            "gradio",
            "streamlit>=1.38.0",
            "fastapi>=0.109.0",
            "uvicorn>=0.27.0",
            "httpx>=0.24.0",
            "python-multipart",
            "numpy==1.26.4",
            "dill",
            "jsonschema==4.22.0",
        )
        .copy_local_file(
            "weights/icon_detect/model.pt", "/root/weights/icon_detect/model.pt"
        )
        .copy_local_dir(
            "weights/icon_caption_florence", "/root/weights/icon_caption_florence"
        )
        .copy_local_dir("util", "/root/util")
    )


app = modal.App("omniparser", image=create_modal_image())


@app.cls(
    gpu=ENV_CONFIG["MODAL_GPU_CONFIG"],
    container_idle_timeout=ENV_CONFIG["MODAL_CONTAINER_TIMEOUT"],
    allow_concurrent_inputs=ENV_CONFIG["CONCURRENCY_LIMIT"],
    concurrency_limit=ENV_CONFIG["MAX_CONTAINERS"],
)
class ModalContainer:
    """Modal container for deploying OmniParser on Modal platform."""

    def __init__(self):
        self.omniparser = OmniParser()

    @modal.enter()
    def enter(self):
        """Initialize models and configure PyTorch settings on container startup"""
        logger.info("Initializing Modal container...")
        self.omniparser.init_models()

    @modal.web_endpoint(method="POST")
    def process_image(self, req: ProcessRequest) -> ProcessResult:
        """Process a single image"""
        result = self.omniparser.process_image(
            image_data=req.image_data,
            box_threshold=req.box_threshold,
            iou_threshold=req.iou_threshold,
            use_paddleocr=req.use_paddleocr,
            imgsz=req.imgsz,
        )
        return ProcessResult(**result)

    @modal.web_endpoint(method="POST")
    def process_batched(self, req: BatchProcessRequest) -> List[ProcessResult]:
        """Process multiple images in a single request"""
        batch_id = f"batch_{str(uuid.uuid4())[:8]}"
        logger.info(f"[{batch_id}] Processing batch of {len(req.images)} images")

        results = []
        for idx, image_data in enumerate(req.images):
            logger.info(f"[{batch_id}] Processing image {idx+1}/{len(req.images)}")
            result = self.omniparser.process_image(
                image_data=image_data,
                box_threshold=req.box_threshold,
                iou_threshold=req.iou_threshold,
                use_paddleocr=req.use_paddleocr,
                imgsz=req.imgsz,
            )
            results.append(ProcessResult(**result))

            # If an error occurred, log it but continue processing the batch
            if "error" in result and result["error"]:
                logger.warning(
                    f"[{batch_id}] Error processing image {idx+1}: {result['error']}"
                )

        logger.info(f"[{batch_id}] Batch processing complete")
        return results


class FastApiOmniParser:
    """FastAPI implementation for local deployment of OmniParser."""

    def __init__(self):
        """Initialize the FastAPI server with OmniParser instance"""
        self.omniparser = OmniParser()
        
        # Define lifespan context manager
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup: initialize models
            logger.info("Initializing models for FastAPI server...")
            self.omniparser.init_models()
            yield
            # Shutdown: cleanup if needed
            logger.info("Shutting down FastAPI server...")
        
        # Create FastAPI app with lifespan
        self.api = FastAPI(
            title="OmniParser API",
            description="API for parsing UI screens",
            version="1.0.0",
            lifespan=lifespan
        )
        
        self._setup_routes()
        logger.info("FastAPI server initialized")

    def _setup_routes(self):
        """Set up FastAPI routes for the API endpoints"""

        @self.api.post("/process_image", response_model=ProcessResult)
        async def process_image(req: ProcessRequest, background_tasks: BackgroundTasks):
            """Process a single image"""
            result = self.omniparser.process_image(
                image_data=req.image_data,
                box_threshold=req.box_threshold,
                iou_threshold=req.iou_threshold,
                use_paddleocr=req.use_paddleocr,
                imgsz=req.imgsz,
            )

            if "error" in result and result["error"]:
                raise HTTPException(status_code=400, detail=result["error"])

            return ProcessResult(**result)

        @self.api.post("/process_batched", response_model=List[ProcessResult])
        async def process_batched(
            req: BatchProcessRequest, background_tasks: BackgroundTasks
        ):
            """Process multiple images in a single request"""
            batch_id = f"batch_{str(uuid.uuid4())[:8]}"
            logger.info(f"[{batch_id}] Processing batch of {len(req.images)} images")

            results = []
            for idx, image_data in enumerate(req.images):
                logger.info(f"[{batch_id}] Processing image {idx+1}/{len(req.images)}")
                result = self.omniparser.process_image(
                    image_data=image_data,
                    box_threshold=req.box_threshold,
                    iou_threshold=req.iou_threshold,
                    use_paddleocr=req.use_paddleocr,
                    imgsz=req.imgsz,
                )
                results.append(ProcessResult(**result))

                # If an error occurred, log it but continue processing the batch
                if "error" in result and result["error"]:
                    logger.warning(
                        f"[{batch_id}] Error processing image {idx+1}: {result['error']}"
                    )

            logger.info(f"[{batch_id}] Batch processing complete")
            return results

        # Add CORS middleware to allow cross-origin requests
        self.api.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def run(self, host="0.0.0.0", port=None, debug=False):
        """Run the FastAPI server"""
        port = port or ENV_CONFIG["API_PORT"]
        logger.info(f"Starting FastAPI server on {host}:{port}")
        uvicorn.run(self.api, host=host, port=port, log_level="info")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OmniParser API Server")
    parser.add_argument(
        "--port",
        type=int,
        default=ENV_CONFIG["API_PORT"],
        help="Port to run the server on",
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to run the server on"
    )
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")

    args = parser.parse_args()

    logger.info("Starting OmniParser API locally with FastAPI...")
    try:
        server = FastApiOmniParser()
        server.run(host=args.host, port=args.port, debug=args.debug)
    except Exception as e:
        logger.critical(f"Failed to start server: {str(e)}")
        sys.exit(1)
