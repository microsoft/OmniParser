import modal
import numpy as np
from PIL import Image
import io
import base64
import os
import logging
import sys
import time
from typing import Dict, List, Optional, Union, Any
import traceback

try:
    from flask import Flask, request, jsonify
except ImportError:
    logging.warning("Flask not installed. Local server functionality will not be available.")

from pydantic import BaseModel, Field, validator

from util.utils import (
    check_ocr_box,
    get_yolo_model,
    get_caption_model_processor,
    get_som_labeled_img,
)

# Default configuration values
DEFAULT_CONCURRENCY_LIMIT = 100
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

    formatter = logging.Formatter(
        "\033[1;36m%(asctime)s\033[0m - \033[1;33m%(name)s\033[0m - \033[1;35m%(levelname)s\033[0m - \033[1;32m%(timing)s\033[0m - %(message)s"
    )

    class TimingFilter(logging.Filter):
        def filter(self, record):
            if not hasattr(record, "timing"):
                record.timing = ""
            return True

    logger.addFilter(TimingFilter())
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.propagate = False

    return logger


logger = setup_logging()
logger.info(f"Environment configuration loaded: {ENV_CONFIG}")


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
        logger.debug("Converting input to PIL Image")
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
            logger.error(f"Failed to convert image: {str(e)}", exc_info=True)
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
    
    @validator('image_data')
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
    
    @validator('images')
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
    parsed_content: str = Field(..., description="Textual representation of parsed content")
    error: Optional[str] = Field(default=None, description="Error message if processing failed")


class OmniParser:
    """Main class for parsing images and extracting information from UI elements."""
    
    def __init__(self):
        """Initialize models upon OmniParser instantiation"""
        self.yolo_model = None
        self.caption_model_processor = None
        self.init_models()

    def init_models(self):
        """Initialize models for both Modal and local environments."""
        try:
            import torch
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

            self.yolo_model = get_yolo_model(model_path="weights/icon_detect/model.pt")
            self.caption_model_processor = get_caption_model_processor(
                model_name="florence2",
                model_name_or_path="weights/icon_caption_florence",
            )
            logger.info("Models initialized successfully")
        except Exception as e:
            logger.critical(f"Failed to initialize models: {str(e)}", exc_info=True)
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
            
        Raises:
            Exception: If processing fails
        """
        start_time = time.time()
        request_id = f"req_{id(image_data)}"
        logger.info(
            f"[{request_id}] Received image processing request",
            extra={"timing": " 0.00s"},
        )
        try:
            # Convert and process image
            image = ImageProcessor.convert_to_pil_image({"image": image_data})
            draw_bbox_config = ImageProcessor.get_bbox_config(image)

            # Perform OCR
            t0 = time.time()
            ocr_bbox_rslt, _ = check_ocr_box(
                image,
                display_img=False,
                output_bb_format="xyxy",
                goal_filtering=None,
                easyocr_args={"paragraph": False, "text_threshold": 0.9},
                use_paddleocr=use_paddleocr,
            )
            text, ocr_bbox = ocr_bbox_rslt
            logger.info(
                f"[{request_id}] OCR completed. Found {len(text)} text elements",
                extra={"timing": f"+{(time.time() - t0):.2f}s"},
            )

            # Process image with ML models
            t0 = time.time()
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
            logger.info(
                f"[{request_id}] Get labeled image completed",
                extra={"timing": f"+{(time.time() - t0):.2f}s"},
            )

            # Prepare response
            output_image = Image.open(io.BytesIO(base64.b64decode(dino_labled_img)))
            buffered = io.BytesIO()
            output_image.save(buffered, format="PNG")
            encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
            parsed_content = "\n".join(
                f"icon {i}: {str(v)}" for i, v in enumerate(parsed_content_list)
            )

            logger.info(
                f"[{request_id}] Processing completed successfully",
                extra={"timing": f" {(time.time() - start_time):.2f}s"},
            )
            return {
                "processed_image": encoded_image,
                "parsed_content": parsed_content,
            }

        except Exception as e:
            logger.error(
                f"[{request_id}] Error processing image: {str(e)}", 
                extra={"timing": f" {(time.time() - start_time):.2f}s"},
                exc_info=True
            )
            return {
                "processed_image": "",
                "parsed_content": "",
                "error": f"Processing failed: {str(e)}\n{traceback.format_exc()}"
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
            "screeninfo",
            "uiautomation",
            "pyautogui==0.9.54",
            "openai==1.3.5",
            "anthropic[bedrock,vertex]>=0.37.1",
            "dashscope",
            "groq",
            "azure-identity",
            "boto3>=1.28.57",
            "google-auth<3,>=2",
            "httpx>=0.24.0",
            "numpy==1.26.4",
            "dill",
            "jsonschema==4.22.0",
            "ruff==0.6.7",
            "pre-commit==3.8.0",
            "pytest==8.3.3",
            "pytest-asyncio==0.23.6",
            "flask",
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
        self._configure_torch()
        self.omniparser.init_models()

    def _configure_torch(self):
        """Configure PyTorch performance settings"""
        import torch

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    def _process_request(self, req: ProcessRequest) -> Dict[str, Any]:
        """Common request processing logic"""
        return self.omniparser.process_image(
            image_data=req.image_data,
            box_threshold=req.box_threshold,
            iou_threshold=req.iou_threshold,
            use_paddleocr=req.use_paddleocr,
            imgsz=req.imgsz,
        )

    @modal.web_endpoint(method="POST")
    def process(self, req: ProcessRequest) -> Dict[str, Any]:
        """Process a single image"""
        return self._process_request(req)

    @modal.web_endpoint(method="POST")
    def process_batched(self, req: BatchProcessRequest) -> List[Dict[str, Any]]:
        """Process multiple images in a single request"""
        try:
            results = []
            for image_data in req.images:
                result = self.omniparser.process_image(
                    image_data=image_data,
                    box_threshold=req.box_threshold,
                    iou_threshold=req.iou_threshold,
                    use_paddleocr=req.use_paddleocr,
                    imgsz=req.imgsz,
                )
                results.append(result)
            return results
        except Exception as e:
            logger.error(f"Error in batched processing: {str(e)}", exc_info=True)
            return [{"error": str(e), "processed_image": "", "parsed_content": ""}]


class FlaskOmniParserServer:
    """Flask server for local deployment of OmniParser."""
    
    def __init__(self):
        """Initialize the Flask server with OmniParser instance"""
        try:
            from flask import Flask
            self.omniparser = OmniParser()
            self.web_app = Flask(__name__)
            self._setup_routes()
            logger.info("Flask server initialized successfully")
        except ImportError as e:
            logger.error(f"Failed to initialize Flask server: {str(e)}")
            raise RuntimeError("Flask is required for local server. Please install Flask.") from e

    def _setup_routes(self):
        """Set up Flask routes for the API endpoints"""
        from flask import request, jsonify
        
        @self.web_app.route("/health", methods=["GET"])
        def health_check():
            """Health check endpoint to verify server is running"""
            return jsonify({"status": "healthy", "version": "1.0.0"})

        @self.web_app.route("/process", methods=["POST"])
        def process():
            """Process a single image"""
            data = request.get_json()
            try:
                result = self.omniparser.process_image(
                    data["image_data"],
                    data.get("box_threshold", DEFAULT_BOX_THRESHOLD),
                    data.get("iou_threshold", DEFAULT_IOU_THRESHOLD),
                    data.get("use_paddleocr", DEFAULT_USE_PADDLEOCR),
                    data.get("imgsz", DEFAULT_IMGSZ),
                )
                return jsonify(result)
            except Exception as e:
                logger.error(f"Error processing image: {str(e)}", exc_info=True)
                return jsonify({"error": str(e), "processed_image": "", "parsed_content": ""}), 500

        @self.web_app.route("/process_batched", methods=["POST"])
        def process_batched():
            """Process multiple images in a single request"""
            data = request.get_json()
            try:
                results = []
                for image_data in data["images"]:
                    result = self.omniparser.process_image(
                        image_data,
                        data.get("box_threshold", DEFAULT_BOX_THRESHOLD),
                        data.get("iou_threshold", DEFAULT_IOU_THRESHOLD),
                        data.get("use_paddleocr", DEFAULT_USE_PADDLEOCR),
                        data.get("imgsz", DEFAULT_IMGSZ),
                    )
                    results.append(result)
                return jsonify(results)
            except Exception as e:
                logger.error(f"Error in batched processing: {str(e)}", exc_info=True)
                return jsonify([{"error": str(e), "processed_image": "", "parsed_content": ""}]), 500

    def run(self, host="0.0.0.0", port=None, debug=False):
        """Run the Flask server"""
        port = port or ENV_CONFIG["API_PORT"]
        logger.info(f"Starting Flask server on {host}:{port}")
        self.web_app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    logger.info("=== Starting OmniParser API locally with Flask ===")
    try:
        server = FlaskOmniParserServer()
        server.run(debug=True)
    except Exception as e:
        logger.critical(f"Failed to start server: {str(e)}", exc_info=True)
        sys.exit(1)
