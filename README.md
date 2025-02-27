# OmniParser - Production-Ready GUI Screen Parser

[![License: CC-BY-4.0](https://img.shields.io/badge/License-CC--BY--4.0-blue.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.68.0+-00a393.svg)](https://fastapi.tiangolo.com/)
[![Modal](https://img.shields.io/badge/Modal-Serverless-blueviolet)](https://modal.com/)

> A high-performance, production-ready fork of Microsoft's OmniParser for converting GUI screens to structured elements.

## Overview

OmniParser is a screen parsing tool that converts general GUI screens to structured elements. This fork maintains all the core functionality of the [original repository](https://github.com/microsoft/OmniParser) while adding significant performance optimizations and production features. It includes seamless integration with [Modal Labs](https://modal.com/) for serverless deployment, allowing you to scale processing dynamically without managing infrastructure.

## Key Enhancements

- **Thread-safe PaddleOCR Pool**: Implementation of a resource pool to efficiently manage OCR instances
- **Parallel Batch Processing**: Process multiple images concurrently with configurable thread pool size
- **Modal Labs Integration**: Ready-to-deploy configuration for Modal's serverless platform
- **Performance Monitoring**: Detailed metrics for request processing and optimized resource utilization
- **FastAPI Server**: Development API server for local testing and development
- **Self-tuning Suggestions**: Dynamic suggestions for optimizing thread pool and batch size parameters

## Getting Started

### Prerequisites

- Python 3.8+
- Git
- [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/quick-start) (for downloading model weights)
- GPU recommended (but not required)

### Installation

Clone this repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/OmniParser-Production
cd OmniParser-Production
pip install -r requirements.txt
```

### Download Model Weights

Download the necessary model weights:

```bash
# Create weights directory structure
mkdir -p weights/icon_detect weights/icon_caption_florence

# Download model weights
# For V2 weights:
for f in icon_detect/{train_args.yaml,model.pt,model.yaml} icon_caption/{config.json,generation_config.json,model.safetensors}; do 
  huggingface-cli download microsoft/OmniParser-v2.0 "$f" --local-dir weights
done
mv weights/icon_caption weights/icon_caption_florence
```

## Running OmniParser

### Local FastAPI Server

Start the server locally for development or private usage:

```bash
python app.py --port 7861 --host 0.0.0.0
```

Access the interactive API documentation at `http://localhost:7861/docs`

### Deploy to Modal Labs

Deploy to Modal for serverless, scalable execution:

```bash
# Install Modal CLI if not already installed
pip install modal

# Log in to Modal
modal login

# Deploy the application
modal deploy app.py
```

After deployment, Modal will provide a unique endpoint URL for your application.

## API Usage

The API provides two main endpoints for processing GUI screens:

### 1. Process Single Image

```
POST /process_image
```

Request body:
```json
{
  "image_data": "data:image/png;base64,iVBORw0K...",
  "box_threshold": 0.05,
  "iou_threshold": 0.1,
  "use_paddleocr": true,
  "imgsz": 640
}
```

### 2. Process Multiple Images In Parallel

```
POST /process_batched
```

Request body:
```json
{
  "images": [
    "data:image/png;base64,iVBORw0K...",
    "data:image/png;base64,iVBORw0K..."
  ],
  "box_threshold": 0.05,
  "iou_threshold": 0.1,
  "use_paddleocr": true,
  "imgsz": 640
}
```

### Example Response

```json
{
  "elements": [
    {
      "type": "button",
      "text": "Submit",
      "bbox": [100, 200, 150, 230],
      "confidence": 0.98
    },
    {
      "type": "text_field",
      "text": "Username",
      "bbox": [50, 100, 200, 130],
      "confidence": 0.95
    }
  ],
  "metadata": {
    "processing_time_ms": 610,
    "model_version": "OmniParser-v2.0",
    "image_size": [1111, 2405]
  }
}
```

### Practical Examples

#### cURL Example for Single Image Processing

```bash
curl -X 'POST' \
  'http://localhost:7861/process_image' \
  -H 'Content-Type: application/json' \
  -d '{
  "image_data": "data:image/png;base64,iVBORw0K...",
  "box_threshold": 0.05,
  "iou_threshold": 0.1,
  "use_paddleocr": true,
  "imgsz": 640
}'
```

#### Python Client Example

```python
import requests
import base64
from PIL import Image
import io

# Load and encode image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return f"data:image/png;base64,{base64.b64encode(image_file.read()).decode('utf-8')}"

# Process single image
def process_image(image_path, api_url="http://localhost:7861/process_image"):
    payload = {
        "image_data": encode_image(image_path),
        "box_threshold": 0.05,
        "iou_threshold": 0.1,
        "use_paddleocr": True,
        "imgsz": 640
    }
    
    response = requests.post(api_url, json=payload)
    return response.json()

# Process batch of images
def process_batch(image_paths, api_url="http://localhost:7861/process_batched"):
    payload = {
        "images": [encode_image(path) for path in image_paths],
        "box_threshold": 0.05,
        "iou_threshold": 0.1,
        "use_paddleocr": True,
        "imgsz": 640
    }
    
    response = requests.post(api_url, json=payload)
    return response.json()

# Example usage
result = process_image("screenshot.png")
print(f"Detected {len(result['elements'])} UI elements")
```

## Configuration Options

### Environment Configuration

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `CONCURRENCY_LIMIT` | Number of concurrent request handlers per container | 1 |
| `MODAL_CONTAINER_TIMEOUT` | Container idle timeout in seconds | 500 |
| `MODAL_GPU_CONFIG` | GPU type for Modal deployment | A100 |
| `API_PORT` | Port for FastAPI server | 7861 |
| `MAX_CONTAINERS` | Maximum number of containers for Modal | 10 |
| `MAX_BATCH_SIZE` | Maximum images per batch request | 1000 |
| `THREAD_POOL_SIZE` | Thread pool size for batch processing | 40 |

### Request Parameters

| Parameter | Description | Default Value | Valid Range |
|-----------|-------------|---------------|------------|
| `box_threshold` | Confidence threshold for bounding boxes | 0.05 | 0.0 - 1.0 |
| `iou_threshold` | IOU threshold for non-maximum suppression | 0.1 | 0.0 - 1.0 |
| `use_paddleocr` | Whether to use PaddleOCR for text detection | true | boolean |
| `imgsz` | Image size for processing | 640 | 320 - 1920 |

### PaddleOCR Pool Configuration

The `paddle_ocr_pool.py` module implements a thread-safe pool of PaddleOCR instances with the following parameters:

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `pool_size` | Number of PaddleOCR instances | 16 |
| `lang` | Language for OCR | en |
| `use_angle_cls` | Use angle classification | false |
| `use_gpu` | Use GPU for OCR | true |
| `max_batch_size` | Maximum batch size for OCR | 1024 |
| `rec_batch_num` | Recognition batch number | 1024 |

### How to Configure

#### Environment Variables

Set environment variables before starting the server:

```bash
# Set environment variables
export THREAD_POOL_SIZE=20
export MAX_BATCH_SIZE=500
export API_PORT=8000

# Start the server with custom configuration
python app.py
```



## Performance

This optimized fork delivers significant performance improvements, especially for batch processing:

- **Single Image Processing**: ~0.61s per image for 1111 × 2405 pixel images with OCR
- **Batch Processing**: Near-linear scaling with thread pool size for non-GPU bound operations
- **Memory Efficiency**: Controlled resource usage with configurable PaddleOCR pool

Performance metrics are automatically logged for each request, providing insights for further optimization.

## Example Output and Logs

OmniParser provides detailed logging that gives insight into the processing pipeline. Below is an example of batch processing logs:

```
2025-02-27 08:26:36,831 - omniparser - INFO - [batch_0b7beafa] Processing batch of 6 images in parallel
2025-02-27 08:26:36,831 - omniparser - INFO - [batch_0b7beafa] Submitting image 1/6 for processing
2025-02-27 08:26:36,831 - omniparser - INFO - [batch_0b7beafa] Submitting image 2/6 for processing
2025-02-27 08:26:36,831 - omniparser - INFO - [batch_0b7beafa] Submitting image 3/6 for processing
2025-02-27 08:26:36,833 - omniparser - INFO - [batch_0b7beafa] Submitting image 4/6 for processing
2025-02-27 08:26:36,859 - omniparser - INFO - [batch_0b7beafa] Submitting image 5/6 for processing
2025-02-27 08:26:36,861 - omniparser - INFO - [batch_0b7beafa] Submitting image 6/6 for processing

2025-02-27 08:26:37,849 - omniparser - INFO - [req_f61fc3be] Request to 'process_image' completed successfully in 1.018s | Steps: {"image_conversion": 0.0, "ocr_processing": 0.307, "icon_detection": 0.598, "response_preparation": 0.112} | {'image_width': 1179, 'image_height': 2556, 'text_elements': 1, 'icons_detected': 5}
2025-02-27 08:26:37,850 - omniparser - INFO - [batch_0b7beafa] Completed processing image 1/6 in 1.02s

2025-02-27 08:26:40,847 - omniparser - INFO - [batch_0b7beafa] Batch processing complete - Stats: Total: 6 | Successful: 6 | Failed: 0 | Time: 3.98s | Avg: 0.66s per image | Thread pool size: 40 | Parallelism efficiency: 0.11 | Image times - Avg: 2.88s | Min: 1.02s | Max: 4.02s Consider reducing THREAD_POOL_SIZE (current: 40)
```

### Understanding the Logs

The logs provide valuable insights into processing performance:

1. **Batch Information**: Each batch receives a unique ID (e.g., `batch_0b7beafa`) and reports the number of images being processed.

2. **Individual Request Details**: Each image processing request includes:
   - Unique request ID (e.g., `req_f61fc3be`)
   - Total processing time (e.g., `1.018s`)
   - Breakdown of processing steps with timings:
     - `image_conversion`: Time spent converting/preprocessing the image
     - `ocr_processing`: Time spent on optical character recognition
     - `icon_detection`: Time spent detecting UI elements and icons
     - `response_preparation`: Time spent preparing the final response

3. **Detection Results**:
   - Image dimensions (width × height)
   - Number of text elements detected
   - Number of icons detected

4. **Batch Summary Statistics**:
   - Success/failure counts
   - Total and average processing times
   - Thread pool utilization metrics
   - Performance optimization suggestions (e.g., "Consider reducing THREAD_POOL_SIZE")

### Automatic Performance Tuning

The system monitors processing efficiency and provides suggestions for optimal configuration:

- Thread pool size adjustments based on parallelism efficiency
- Batch size recommendations based on processing characteristics
- Resource allocation suggestions for Modal deployment

## License

This project is licensed under the CC-BY-4.0 License, the same as the original Microsoft OmniParser repository.
