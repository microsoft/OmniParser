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
import traceback
import json
import uuid

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


class CanonicalLogger:
    """Helper class for creating canonical log lines."""

    def __init__(self, request_id, endpoint, request_params=None):
        self.start_time = time.time()
        self.request_id = request_id
        self.endpoint = endpoint
        self.request_params = request_params or {}
        self.timings = {}
        self.current_step = None
        self.step_start_time = None
        self.metadata = {}

    def start_step(self, step_name):
        """Start timing a new processing step."""
        self.current_step = step_name
        self.step_start_time = time.time()
        return self

    def end_step(self):
        """End timing for the current step and record its duration."""
        if self.current_step and self.step_start_time:
            duration = time.time() - self.step_start_time
            self.timings[self.current_step] = round(duration, 3)
            self.current_step = None
            self.step_start_time = None
        return self

    def add_metadata(self, key, value):
        """Add additional metadata to be included in the log line."""
        self.metadata[key] = value
        return self

    def log_success(self, logger_instance, additional_data=None):
        """Log a successful request with all collected information."""
        total_duration = time.time() - self.start_time

        log_data = {
            "request_id": self.request_id,
            "endpoint": self.endpoint,
            "total_duration": round(total_duration, 3),
            "status": "success",
            "timings": self.timings,
        }

        # Add request parameters
        if self.request_params:
            log_data["request_params"] = self.request_params

        # Add metadata
        if self.metadata:
            log_data.update(self.metadata)

        # Add any additional data
        if additional_data:
            log_data.update(additional_data)

        # Convert to logfmt style
        logfmt_line = " ".join(
            [f"{k}={self._format_value(v)}" for k, v in log_data.items()]
        )
        logger_instance.info(f"REQUEST_LOG {logfmt_line}")

    def log_error(self, logger_instance, error, traceback_str=None):
        """Log a failed request with error information."""
        total_duration = time.time() - self.start_time

        log_data = {
            "request_id": self.request_id,
            "endpoint": self.endpoint,
            "total_duration": round(total_duration, 3),
            "status": "error",
            "error": str(error),
            "timings": self.timings,
        }

        # Add request parameters
        if self.request_params:
            log_data["request_params"] = self.request_params

        # Add metadata
        if self.metadata:
            log_data.update(self.metadata)

        # Convert to logfmt style
        logfmt_line = " ".join(
            [f"{k}={self._format_value(v)}" for k, v in log_data.items()]
        )
        logger_instance.error(f"CANONICAL_LOG {logfmt_line}")

        # Log traceback separately for debugging if provided
        if traceback_str and logger_instance.isEnabledFor(logging.DEBUG):
            logger_instance.debug(f"Traceback for {self.request_id}: {traceback_str}")

    def _format_value(self, value):
        """Format a value for logfmt style logging."""
        if isinstance(value, dict):
            # Convert dict to a compact string representation
            return json.dumps(value, separators=(",", ":"))
        elif isinstance(value, (list, tuple)):
            # Convert list/tuple to a compact string representation
            return json.dumps(value, separators=(",", ":"))
        elif isinstance(value, bool):
            return str(value).lower()
        elif value is None:
            return "null"
        else:
            # Quote strings that contain spaces or special characters
            value_str = str(value)
            if " " in value_str or "=" in value_str or '"' in value_str:
                return '"' + value_str.replace('"', '\\"') + '"'
            return value_str


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

        Raises:
            Exception: If processing fails
        """
        request_id = f"req_{str(uuid.uuid4())[:8]}"

        # Initialize canonical logger for this request
        canonical_log = CanonicalLogger(
            request_id=request_id,
            endpoint="process_image",
            request_params={
                "box_threshold": box_threshold,
                "iou_threshold": iou_threshold,
                "use_paddleocr": use_paddleocr,
                "imgsz": imgsz,
            },
        )

        try:
            # Convert and process image
            canonical_log.start_step("image_conversion")
            image = ImageProcessor.convert_to_pil_image({"image": image_data})
            draw_bbox_config = ImageProcessor.get_bbox_config(image)
            canonical_log.end_step()

            # Set up utils to avoid logging
            os.environ["OMNIPARSER_SUPPRESS_UTILS_LOGS"] = "1"

            # Perform OCR
            canonical_log.start_step("ocr_processing")
            ocr_bbox_rslt, _ = check_ocr_box(
                image,
                display_img=False,
                output_bb_format="xyxy",
                goal_filtering=None,
                easyocr_args={"paragraph": False, "text_threshold": 0.9},
                use_paddleocr=use_paddleocr,
            )
            text, ocr_bbox = ocr_bbox_rslt
            canonical_log.add_metadata("text_elements_count", len(text))
            canonical_log.end_step()

            # Process image with ML models
            canonical_log.start_step("icon_detection")
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
            canonical_log.add_metadata("icons_detected", len(parsed_content_list))
            canonical_log.end_step()

            # Prepare response
            canonical_log.start_step("response_preparation")
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
            canonical_log.end_step()

            # Log success with canonical log line
            canonical_log.log_success(
                logger,
                {
                    "image_width": image.width,
                    "image_height": image.height,
                },
            )

            return result

        except Exception as e:
            # Stop timing current step if there is one
            if canonical_log.current_step:
                canonical_log.end_step()

            # Log error with canonical log line
            canonical_log.log_error(logger, e, traceback.format_exc())

            return {
                "processed_image": "",
                "parsed_content": "",
                "error": f"Processing failed: {str(e)}",
            }
        finally:
            # Reset utils logging suppression
            os.environ.pop("OMNIPARSER_SUPPRESS_UTILS_LOGS", None)


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
    def process_batched(self, req: BatchProcessRequest) -> List[Dict[str, Any]]:
        """Process multiple images in a single request"""
        batch_canonical_log = CanonicalLogger(
            request_id=f"batch_{str(uuid.uuid4())[:8]}",
            endpoint="process_batched",
            request_params={
                "batch_size": len(req.images),
                "box_threshold": req.box_threshold,
                "iou_threshold": req.iou_threshold,
                "use_paddleocr": req.use_paddleocr,
                "imgsz": req.imgsz,
            },
        )

        try:
            batch_canonical_log.start_step("batch_processing")
            results = []

            for idx, image_data in enumerate(req.images):
                batch_canonical_log.start_step(f"image_{idx}")
                result = self.omniparser.process_image(
                    image_data=image_data,
                    box_threshold=req.box_threshold,
                    iou_threshold=req.iou_threshold,
                    use_paddleocr=req.use_paddleocr,
                    imgsz=req.imgsz,
                )
                
                # Check if the current image processing failed
                if "error" in result and result["error"]:
                    # If there's an error, log it and raise an exception to fail the entire batch
                    error_msg = f"Batch processing failed at image {idx}: {result['error']}"
                    batch_canonical_log.end_step()
                    batch_canonical_log.end_step()  # End both image step and batch step
                    batch_canonical_log.log_error(logger, error_msg)
                    raise ValueError(error_msg)
                    
                results.append(result)
                batch_canonical_log.end_step()

            batch_canonical_log.end_step()

            # Log success with canonical log line
            batch_canonical_log.log_success(
                logger,
                {
                    "successful_images": len(results),
                    "failed_images": 0,  # All must be successful at this point
                },
            )

            return results
        except Exception as e:
            # Stop timing current step if there is one
            if batch_canonical_log.current_step:
                batch_canonical_log.end_step()

            # Log error with canonical log line
            batch_canonical_log.log_error(logger, e, traceback.format_exc())

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
            raise RuntimeError(
                "Flask is required for local server. Please install Flask."
            ) from e

    def _setup_routes(self):
        """Set up Flask routes for the API endpoints"""
        from flask import request, jsonify

        @self.web_app.route("/process_batched", methods=["POST"])
        def process_batched():
            """Process multiple images in a single request"""
            data = request.get_json()
            batch_canonical_log = CanonicalLogger(
                request_id=f"flask_batch_{str(uuid.uuid4())[:8]}",
                endpoint="/process_batched",
                request_params={
                    "batch_size": len(data["images"]),
                    "box_threshold": data.get("box_threshold", DEFAULT_BOX_THRESHOLD),
                    "iou_threshold": data.get("iou_threshold", DEFAULT_IOU_THRESHOLD),
                    "use_paddleocr": data.get("use_paddleocr", DEFAULT_USE_PADDLEOCR),
                    "imgsz": data.get("imgsz", DEFAULT_IMGSZ),
                },
            )

            try:
                batch_canonical_log.start_step("batch_processing")
                results = []

                for idx, image_data in enumerate(data["images"]):
                    batch_canonical_log.start_step(f"image_{idx}")
                    result = self.omniparser.process_image(
                        image_data,
                        data.get("box_threshold", DEFAULT_BOX_THRESHOLD),
                        data.get("iou_threshold", DEFAULT_IOU_THRESHOLD),
                        data.get("use_paddleocr", DEFAULT_USE_PADDLEOCR),
                        data.get("imgsz", DEFAULT_IMGSZ),
                    )
                    
                    # Check if the current image processing failed
                    if "error" in result and result["error"]:
                        # If there's an error, log it and raise an exception to fail the entire batch
                        error_msg = f"Batch processing failed at image {idx}: {result['error']}"
                        batch_canonical_log.end_step()
                        batch_canonical_log.end_step()  # End both image step and batch step
                        batch_canonical_log.log_error(logger, error_msg)
                        return jsonify({"error": error_msg, "processed_image": "", "parsed_content": ""}), 400
                        
                    results.append(result)
                    batch_canonical_log.end_step()

                batch_canonical_log.end_step()

                # Log success with canonical log line
                batch_canonical_log.log_success(
                    logger,
                    {
                        "http_status": 200,
                        "client_ip": request.remote_addr,
                        "successful_images": len(results),
                        "failed_images": 0,  # All must be successful at this point
                    },
                )

                return jsonify(results)
            except Exception as e:
                # Stop timing current step if there is one
                if batch_canonical_log.current_step:
                    batch_canonical_log.end_step()

                # Log error with canonical log line
                batch_canonical_log.log_error(logger, e, traceback.format_exc())

                error_response = {"error": str(e), "processed_image": "", "parsed_content": ""}
                return jsonify(error_response), 500

    def run(self, host="0.0.0.0", port=None, debug=False):
        """Run the Flask server"""
        port = port or ENV_CONFIG["API_PORT"]
        logger.info(f"Starting Flask server on {host}:{port}")
        self.web_app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    logger.info("Starting OmniParser API locally with Flask...")
    try:
        server = FlaskOmniParserServer()
        server.run(debug=True)
    except Exception as e:
        logger.critical(f"Failed to start server: {str(e)}")
        sys.exit(1)
