import modal
import numpy as np
from PIL import Image
import io
import base64
import os
import logging
import sys
import time
import signal
from typing import Dict, List, Optional, Any, Callable, Tuple
import traceback
import json
import uuid
import concurrent.futures
import psutil
import gc
import threading
import torch
import GPUtil  # type: ignore
from pydantic import BaseModel, Field, field_validator
import random

from util.utils import (
    check_ocr_box,
    get_yolo_model,
    get_caption_model_processor,
    get_som_labeled_img,
    paddle_ocr_pool,
)

# Default configuration values
DEFAULT_CONCURRENCY_LIMIT = 1
DEFAULT_CONTAINER_TIMEOUT = 300
DEFAULT_GPU_CONFIG = "H100"
DEFAULT_API_PORT = 7861
DEFAULT_MAX_CONTAINERS = 10
DEFAULT_MAX_BATCH_SIZE = 100
DEFAULT_LOG_LEVEL = "DEBUG"

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
    "MAX_BATCH_SIZE": int(
        os.environ.get("MAX_BATCH_SIZE", str(DEFAULT_MAX_BATCH_SIZE))
    ),
    "LOG_LEVEL": os.environ.get("LOG_LEVEL", DEFAULT_LOG_LEVEL).upper(),
}

def setup_logging():
    """Configure and return a logger with custom formatting and stream handlers."""
    logger = logging.getLogger("omniparser")

    # Set log level based on configuration
    log_level = ENV_CONFIG["LOG_LEVEL"]
    logger.setLevel(
        logging.DEBUG
        if log_level == "DEBUG"
        else logging.INFO if log_level == "INFO" else logging.CRITICAL
    )

    # Force removal of any existing handlers to prevent duplicates
    logger.handlers = []

    # Create console handler and set level explicitly
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(
        logging.DEBUG
        if log_level == "DEBUG"
        else logging.INFO if log_level == "INFO" else logging.CRITICAL
    )

    # Simple formatter for regular logs
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.propagate = False

    # Set the root logger level to match our logger to prevent other libraries from overriding
    logging.basicConfig(level=logger.level)

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

        # Choose log type based on endpoint
        if "batch" in self.endpoint.lower():
            logger_instance.info(f"BATCH_LINE {logfmt_line}")
        else:
            logger_instance.info(f"IMAGE_LINE {logfmt_line}")

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

        # Choose log type based on endpoint, consistent with log_success
        if "batch" in self.endpoint.lower():
            logger_instance.error(f"BATCH_LINE {logfmt_line}")
        else:
            logger_instance.error(f"IMAGE_LINE {logfmt_line}")

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

    @field_validator("image_data")
    @classmethod
    def validate_image_data(cls, v):
        if not v.startswith("data:image"):
            raise ValueError("Image data must begin with 'data:image' prefix")
        return v


class BatchProcessRequest(BaseModel):
    """Request model for processing multiple images in a batch.

    Images are processed in parallel using a thread pool, with the maximum number
    of concurrent threads controlled by the MAX_BATCH_SIZE environment variable.
    Results are returned in the same order as the input images.
    """

    images: List[str] = Field(
        ...,
        description="Array of base64 encoded image strings including data URI prefix",
    )
    box_threshold: float = Field(default=DEFAULT_BOX_THRESHOLD, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=DEFAULT_IOU_THRESHOLD, ge=0.0, le=1.0)
    use_paddleocr: bool = Field(default=DEFAULT_USE_PADDLEOCR)
    imgsz: int = Field(default=DEFAULT_IMGSZ, ge=320, le=1920)

    @field_validator("images")
    @classmethod
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


# New utility classes to reduce code duplication


def collect_resource_metrics(detailed=False) -> Dict[str, float]:
    """
    Collect system and GPU resource metrics.

    Args:
        detailed: Whether to collect detailed metrics (more expensive operations)

    Returns:
        Dict of resource metrics
    """
    # Basic metrics that are always collected when metrics are enabled
    metrics = {
        "cpu_percent": psutil.cpu_percent(),
        "system_memory_percent": psutil.virtual_memory().percent,
        "system_memory_mb": psutil.virtual_memory().used / (1024 * 1024),
    }

    # Add GPU metrics if available and detailed metrics are requested
    if torch.cuda.is_available():
        try:
            # Force synchronization to get accurate metrics
            torch.cuda.synchronize()
            
            # For H100s, we need to be more aggressive about GPU metrics collection
            # Get GPU metrics using GPUtil - more expensive but provides utilization info
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use the first GPU
                metrics["gpu_memory_percent"] = gpu.memoryUtil * 100
                metrics["gpu_memory_mb"] = gpu.memoryUsed
                metrics["gpu_utilization"] = gpu.load * 100
            else:
                # Fallback to PyTorch for basic memory info
                gpu_id = torch.cuda.current_device()
                metrics["gpu_memory_mb"] = torch.cuda.memory_allocated(gpu_id) / (
                    1024 * 1024
                )
                metrics["gpu_memory_percent"] = (
                    metrics["gpu_memory_mb"]
                    / (
                        torch.cuda.get_device_properties(gpu_id).total_memory
                        / (1024 * 1024)
                    )
                ) * 100
                metrics["gpu_utilization"] = 0  # Not available through PyTorch alone
        except Exception as e:
            logger.warning(f"Failed to collect GPU metrics: {e}")
            metrics["gpu_memory_mb"] = 0
            metrics["gpu_memory_percent"] = 0
            metrics["gpu_utilization"] = 0
    else:
        metrics["gpu_memory_mb"] = 0
        metrics["gpu_memory_percent"] = 0
        metrics["gpu_utilization"] = 0

    return metrics


def process_batch_images(
    request_id: str,
    omniparser: OmniParser,
    images: List[str],
    process_params: Dict[str, Any],
    max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
    collect_metrics: bool = True,
    collect_detailed_metrics: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Process multiple images in parallel using a thread pool.

    Args:
        request_id: Unique identifier for this batch request
        omniparser: OmniParser instance to use for processing
        images: List of image data strings
        process_params: Parameters for image processing (box_threshold, iou_threshold, etc.)
        max_batch_size: Maximum number of threads to use
        collect_metrics: Whether to collect performance metrics
        collect_detailed_metrics: Whether to collect detailed performance metrics

    Returns:
        Tuple containing:
        - List of results for each image
        - Dictionary of performance metrics
    """
    # Limit the batch size based on available resources to prevent thread contention
    # Try to import the PaddleOCRPool size to align thread count with OCR resources
    try:
        from util.utils import paddle_ocr_pool
        recommended_batch_size = min(paddle_ocr_pool.pool_size * 6, len(images))
        
        # Cap at the configured max_batch_size
        effective_batch_size = min(recommended_batch_size, max_batch_size)
        logger.debug(f"[{request_id}] Adjusting batch size to {effective_batch_size} (OCR pool size: {paddle_ocr_pool.pool_size}, images: {len(images)})")
    except (ImportError, AttributeError):
        # If we can't import paddle_ocr_pool, just use the configured max_batch_size
        # But still respect the number of images
        effective_batch_size = min(max_batch_size, len(images))
        logger.debug(f"[{request_id}] Using configured batch size: {effective_batch_size}")
    
    batch_canonical_log = CanonicalLogger(
        request_id=request_id,
        endpoint="process_batched",
        request_params={
            "batch_size": len(images),
            **process_params,
            "max_threads": effective_batch_size,
        },
    )

    # Track performance metrics if enabled
    start_time = time.time()
    per_image_times = {}
    active_threads = 0
    max_active_threads = 0

    # Initial resource usage metrics if enabled
    initial_metrics = {}
    if collect_metrics:
        batch_canonical_log.start_step("resource_monitoring")
        initial_metrics = collect_resource_metrics(detailed=collect_detailed_metrics)
        batch_canonical_log.end_step()

    try:
        batch_canonical_log.start_step("batch_processing")
        results = []

        # Create a semaphore to track active threads
        thread_semaphore = threading.Semaphore(0)
        thread_counter_lock = threading.Lock()

        def process_with_metrics(idx, image_data):
            nonlocal active_threads, max_active_threads
            image_start_time = time.time()

            # Increment active thread count if metrics collection is enabled
            if collect_metrics:
                with thread_counter_lock:
                    active_threads += 1
                    max_active_threads = max(max_active_threads, active_threads)

            try:
                thread_id = threading.get_ident()
                logger.debug(
                    f"[{request_id}] Starting processing image {idx} in thread {thread_id}"
                )

                # Collect pre-processing GPU metrics for this thread if detailed metrics are enabled
                pre_gpu_mem = 0
                if collect_detailed_metrics and torch.cuda.is_available():
                    gpu_id = torch.cuda.current_device()
                    pre_gpu_mem = torch.cuda.memory_allocated(gpu_id) / (1024**2)  # MB
                    logger.debug(
                        f"[{request_id}] Image {idx} - Pre-process GPU memory: {pre_gpu_mem:.2f} MB"
                    )

                # Process the image
                # Setting CUDA thread priority higher to ensure GPU access isn't a bottleneck
                if torch.cuda.is_available():
                    torch.cuda.set_device(0)  # Ensure using first GPU
                    # Force device synchronization before processing to avoid thread contention
                    torch.cuda.synchronize()
                    
                    # Process the image with GPU acceleration
                    result = omniparser.process_image(
                        image_data=image_data,
                        box_threshold=process_params.get(
                            "box_threshold", DEFAULT_BOX_THRESHOLD
                        ),
                        iou_threshold=process_params.get(
                            "iou_threshold", DEFAULT_IOU_THRESHOLD
                        ),
                        use_paddleocr=process_params.get(
                            "use_paddleocr", DEFAULT_USE_PADDLEOCR
                        ),
                        imgsz=process_params.get("imgsz", DEFAULT_IMGSZ),
                    )
                    
                    # Force synchronization after processing
                    torch.cuda.synchronize()
                else:
                    result = omniparser.process_image(
                        image_data=image_data,
                        box_threshold=process_params.get(
                            "box_threshold", DEFAULT_BOX_THRESHOLD
                        ),
                        iou_threshold=process_params.get(
                            "iou_threshold", DEFAULT_IOU_THRESHOLD
                        ),
                        use_paddleocr=process_params.get(
                            "use_paddleocr", DEFAULT_USE_PADDLEOCR
                        ),
                        imgsz=process_params.get("imgsz", DEFAULT_IMGSZ),
                    )

                # Collect post-processing GPU metrics if detailed metrics are enabled
                if collect_detailed_metrics and torch.cuda.is_available():
                    gpu_id = torch.cuda.current_device()
                    post_gpu_mem = torch.cuda.memory_allocated(gpu_id) / (1024**2)  # MB
                    logger.debug(
                        f"[{request_id}] Image {idx} - Post-process GPU memory: {post_gpu_mem:.2f} MB"
                    )
                    logger.debug(
                        f"[{request_id}] Image {idx} - GPU memory delta: {post_gpu_mem - pre_gpu_mem:.2f} MB"
                    )

                # Record processing time if metrics collection is enabled
                if collect_metrics:
                    image_time = time.time() - image_start_time
                    per_image_times[idx] = image_time
                    logger.debug(
                        f"[{request_id}] Finished processing image {idx} in {image_time:.3f}s"
                    )

                return result
            except Exception as exc:
                logger.error(f"[{request_id}] Image {idx} processing failed: {exc}")
                return {
                    "processed_image": "",
                    "parsed_content": "",
                    "error": f"Processing failed: {str(exc)}",
                }
            finally:
                # Release resources explicitly if detailed metrics are enabled
                if collect_detailed_metrics and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Reduce aggressive garbage collection - it's causing synchronization issues
                if collect_metrics:
                    # Only do GC on 10% of threads instead of 20% to reduce overhead
                    if collect_detailed_metrics or (random.random() < 0.1):
                        gc.collect(generation=0)  # Only collect youngest generation

                    # Decrement active thread count
                    with thread_counter_lock:
                        active_threads -= 1

                    # Signal thread completion
                    thread_semaphore.release()

        # Process images in parallel using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=effective_batch_size
        ) as executor:            
            # Create a list of future objects
            future_to_idx = {
                executor.submit(process_with_metrics, idx, image_data): idx
                for idx, image_data in enumerate(images)
            }

            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                batch_canonical_log.start_step(f"image_{idx}")
                try:
                    logger.debug(f"[{request_id}] Processing result for image {idx}")
                    result = future.result()
                    # Ensure results are ordered by original index
                    while len(results) <= idx:
                        results.append(None)
                    results[idx] = result
                    logger.debug(f"[{request_id}] Completed processing for image {idx}")
                except Exception as exc:
                    logger.error(f"[{request_id}] Image {idx} processing failed: {exc}")
                    # Add error entry for failed image
                    while len(results) <= idx:
                        results.append(None)
                    results[idx] = {
                        "processed_image": "",
                        "parsed_content": "",
                        "error": f"Processing failed: {str(exc)}",
                    }
                finally:
                    batch_canonical_log.end_step()

        # Fill any missing results (though this shouldn't happen)
        results = [
            (
                r
                if r is not None
                else {
                    "processed_image": "",
                    "parsed_content": "",
                    "error": "Processing failed: Unknown error",
                }
            )
            for r in results
        ]

        batch_canonical_log.end_step()

        # Generate performance metrics
        performance_metrics = {}

        if collect_metrics:
            # Collect final resource metrics
            batch_canonical_log.start_step("final_resource_monitoring")
            final_metrics = collect_resource_metrics(detailed=collect_detailed_metrics)
            batch_canonical_log.end_step()

            # Calculate resource usage deltas
            memory_delta_mb = final_metrics.get(
                "system_memory_mb", 0
            ) - initial_metrics.get("system_memory_mb", 0)
            gpu_memory_delta_mb = final_metrics.get(
                "gpu_memory_mb", 0
            ) - initial_metrics.get("gpu_memory_mb", 0)

            # Calculate timing statistics
            total_time = time.time() - start_time

            # Basic performance metrics
            performance_metrics = {
                "batch_size": len(images),
                "total_time_seconds": f"{total_time:.3f}",
                "successful_images": len(
                    [r for r in results if "error" not in r or not r["error"]]
                ),
                "failed_images": len(
                    [r for r in results if "error" in r and r["error"]]
                ),
            }

            # Add detailed performance metrics if enabled
            if collect_detailed_metrics:
                avg_time = (
                    sum(per_image_times.values()) / len(per_image_times)
                    if per_image_times
                    else 0
                )
                max_time = max(per_image_times.values()) if per_image_times else 0
                min_time = min(per_image_times.values()) if per_image_times else 0

                # Compute actual throughput
                throughput = len(images) / total_time
                
                # Performance analysis focused on GPU utilization
                gpu_utilization = final_metrics.get("gpu_utilization", 0)
                gpu_memory_percent = final_metrics.get("gpu_memory_percent", 0)
                
                # Calculate efficiency based on GPU utilization instead of thread throughput
                gpu_efficiency = gpu_utilization  # Direct measure of GPU usage
                
                # Determine bottleneck with focus on GPU utilization
                bottleneck = "Unknown"
                if gpu_utilization < 30:
                    if max_active_threads < effective_batch_size:
                        bottleneck = "Insufficient Threads"
                    elif final_metrics.get("system_memory_percent", 0) > 90:
                        bottleneck = "System Memory"
                    elif final_metrics.get("cpu_percent", 0) > 90:
                        bottleneck = "CPU"
                    else:
                        bottleneck = "Thread Synchronization or I/O"
                elif gpu_utilization < 70:
                    if gpu_memory_percent > 90:
                        bottleneck = "GPU Memory"
                    else:
                        bottleneck = "Mixed CPU/GPU Processing"
                else:
                    bottleneck = "GPU Compute"

                # Suggestion for optimal thread count based on GPU utilization
                suggested_threads = max_active_threads
                
                # If GPU is underutilized, recommend more threads to increase load
                if gpu_utilization < 40 and max_active_threads <= effective_batch_size:
                    # Recommend more threads when GPU is idle
                    suggested_threads = min(100, int(effective_batch_size * 1.5))
                # If GPU memory is nearly full but utilization is low
                elif gpu_memory_percent > 85 and gpu_utilization < 70:
                    # Reduce thread count to prevent OOM but maintain processing
                    suggested_threads = max(1, int(max_active_threads * 0.8))
                # If GPU is well utilized (70-95%)
                elif 70 <= gpu_utilization <= 95:
                    # Keep current thread count as it's working well
                    suggested_threads = max_active_threads
                # If GPU is maxed out (>95%)
                elif gpu_utilization > 95:
                    # Slight reduction to prevent throttling
                    suggested_threads = max(1, int(max_active_threads * 0.9))

                # Add the detailed metrics
                detailed_metrics = {
                    "max_threads_config": effective_batch_size,
                    "max_active_threads": max_active_threads,
                    "avg_image_time_seconds": f"{avg_time:.3f}",
                    "min_image_time_seconds": f"{min_time:.3f}",
                    "max_image_time_seconds": f"{max_time:.3f}",
                    "images_per_second": f"{throughput:.2f}",
                    "gpu_efficiency_percent": f"{gpu_efficiency:.1f}",
                    "system_memory_used_mb": f"{final_metrics.get('system_memory_mb', 0):.1f}",
                    "system_memory_delta_mb": f"{memory_delta_mb:.1f}",
                    "system_memory_percent": f"{final_metrics.get('system_memory_percent', 0):.1f}",
                    "gpu_memory_used_mb": f"{final_metrics.get('gpu_memory_mb', 0):.1f}",
                    "gpu_memory_delta_mb": f"{gpu_memory_delta_mb:.1f}",
                    "gpu_memory_percent": f"{final_metrics.get('gpu_memory_percent', 0):.1f}",
                    "gpu_utilization_percent": f"{final_metrics.get('gpu_utilization', 0):.1f}",
                    "likely_bottleneck": bottleneck,
                    "suggested_thread_count": suggested_threads,
                }

                # Merge detailed metrics into performance metrics
                performance_metrics.update(detailed_metrics)

            # Log performance metrics in a structured format
            if collect_detailed_metrics:
                logger.debug(
                    f"BATCH_LINE {' '.join([f'{k}={v}' for k, v in performance_metrics.items()])}"
                )

            # Add summarized metrics to canonical log
            batch_canonical_log.log_success(logger, performance_metrics)
        else:
            # If performance metrics are disabled, just log basic success
            batch_canonical_log.log_success(
                logger,
                {
                    "successful_images": len(
                        [r for r in results if "error" not in r or not r["error"]]
                    ),
                    "failed_images": len(
                        [r for r in results if "error" in r and r["error"]]
                    ),
                },
            )

        return results, performance_metrics

    except Exception as e:
        # Stop timing current step if there is one
        if batch_canonical_log.current_step:
            batch_canonical_log.end_step()

        # Log error with canonical log line
        batch_canonical_log.log_error(logger, e, traceback.format_exc())

        return [{"error": str(e), "processed_image": "", "parsed_content": ""}], {}


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
            "gputil",
            "psutil",
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
        request_id = f"batch_{str(uuid.uuid4())[:8]}"

        logger.debug(
            f"[{request_id}] Starting batch processing with {len(req.images)} images"
        )

        log_level = ENV_CONFIG["LOG_LEVEL"]
        collect_metrics = log_level != "OFF"
        collect_detailed_metrics = log_level == "DEBUG"

        process_params = {
            "box_threshold": req.box_threshold,
            "iou_threshold": req.iou_threshold,
            "use_paddleocr": req.use_paddleocr,
            "imgsz": req.imgsz,
        }

        results, _ = process_batch_images(
            request_id=request_id,
            omniparser=self.omniparser,
            images=req.images,
            process_params=process_params,
            max_batch_size=ENV_CONFIG["MAX_BATCH_SIZE"],
            collect_metrics=collect_metrics,
            collect_detailed_metrics=collect_detailed_metrics,
        )

        return results


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
        except Exception as e:
            logger.error(f"Failed to initialize Flask server: {str(e)}")
            raise RuntimeError(
                "Flask initialization failed. See error log for details."
            ) from e

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
            canonical_log = CanonicalLogger(
                request_id=f"flask_{str(uuid.uuid4())[:8]}",
                endpoint="/process",
                request_params={
                    "box_threshold": data.get("box_threshold", DEFAULT_BOX_THRESHOLD),
                    "iou_threshold": data.get("iou_threshold", DEFAULT_IOU_THRESHOLD),
                    "use_paddleocr": data.get("use_paddleocr", DEFAULT_USE_PADDLEOCR),
                    "imgsz": data.get("imgsz", DEFAULT_IMGSZ),
                },
            )

            try:
                canonical_log.start_step("process_request")
                result = self.omniparser.process_image(
                    data["image_data"],
                    data.get("box_threshold", DEFAULT_BOX_THRESHOLD),
                    data.get("iou_threshold", DEFAULT_IOU_THRESHOLD),
                    data.get("use_paddleocr", DEFAULT_USE_PADDLEOCR),
                    data.get("imgsz", DEFAULT_IMGSZ),
                )
                canonical_log.end_step()

                # Log success with canonical log line
                canonical_log.log_success(
                    logger,
                    {
                        "client_ip": request.remote_addr,
                    },
                )

                return jsonify(result)
            except Exception as e:
                # Stop timing current step if there is one
                if canonical_log.current_step:
                    canonical_log.end_step()

                # Log error with canonical log line
                canonical_log.log_error(logger, e, traceback.format_exc())

                error_response = {
                    "error": str(e),
                    "processed_image": "",
                    "parsed_content": "",
                }
                return jsonify(error_response), 500

        @self.web_app.route("/process_batched", methods=["POST"])
        def process_batched():
            """Process multiple images in a single request"""
            data = request.get_json()
            request_id = f"flask_batch_{str(uuid.uuid4())[:8]}"

            logger.debug(
                f"[{request_id}] Starting batch processing with {len(data['images'])} images"
            )

            log_level = ENV_CONFIG["LOG_LEVEL"]
            collect_metrics = log_level != "OFF"
            collect_detailed_metrics = log_level == "DEBUG"

            process_params = {
                "box_threshold": data.get("box_threshold", DEFAULT_BOX_THRESHOLD),
                "iou_threshold": data.get("iou_threshold", DEFAULT_IOU_THRESHOLD),
                "use_paddleocr": data.get("use_paddleocr", DEFAULT_USE_PADDLEOCR),
                "imgsz": data.get("imgsz", DEFAULT_IMGSZ),
            }

            results, _ = process_batch_images(
                request_id=request_id,
                omniparser=self.omniparser,
                images=data["images"],
                process_params=process_params,
                max_batch_size=ENV_CONFIG["MAX_BATCH_SIZE"],
                collect_metrics=collect_metrics,
                collect_detailed_metrics=collect_detailed_metrics,
            )

            return jsonify(results)

    def run(self, host="0.0.0.0", port=None, debug=False):
        """Run the Flask server"""
        port = port or ENV_CONFIG["API_PORT"]
        logger.info(f"Starting Flask server on {host}:{port}")

        # Disable Flask's default logging handler to prevent overriding our settings
        import logging
        from flask.logging import default_handler

        self.web_app.logger.removeHandler(default_handler)

        # Ensure Flask's logger uses the same level as our logger
        self.web_app.logger.setLevel(logger.level)
        for handler in logger.handlers:
            self.web_app.logger.addHandler(handler)

        # Disable Werkzeug's built-in logging if we're not in debug mode
        if not debug:
            log = logging.getLogger("werkzeug")
            log.setLevel(logging.ERROR)

        self.web_app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    logger.info("Starting OmniParser API locally with Flask...")
    try:
        server = FlaskOmniParserServer()
        server.run()
    except Exception as e:
        logger.critical(f"Failed to start server: {str(e)}")
        sys.exit(1)
