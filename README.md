# OmniParser V2 - Production-Ready GUI Screen Parser

[![License: CC-BY-4.0](https://img.shields.io/badge/License-CC--BY--4.0-blue.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.68.0+-00a393.svg)](https://fastapi.tiangolo.com/)
[![Modal](https://img.shields.io/badge/Modal-Serverless-blueviolet)](https://modal.com/)

> A high-performance, production-ready fork of Microsoft's OmniParser for converting GUI screens to structured elements.

## Overview

OmniParser V2 converts GUI screenshots into structured data representing UI elements. This production-ready fork builds on [Microsoft's original OmniParser](https://github.com/microsoft/OmniParser) with significant performance optimizations and deployment features.

The tool identifies buttons, text fields, icons, and other UI components from images, enabling automated testing, accessibility analysis, and UI documentation. With [Modal Labs](https://modal.com/) integration, you can deploy the parser as a scalable, serverless API.

## Key Enhancements

- **🚀 Thread-safe PaddleOCR Pool**: Efficiently manages OCR instances for optimal resource utilization
- **⚡ Parallel Batch Processing**: Process multiple images concurrently with up to 40x throughput
- **☁️ Modal Labs Integration**: One-command deployment to serverless infrastructure
- **📊 Performance Monitoring**: Detailed metrics with actionable optimization suggestions
- **🔄 Parameter Recommendations**: Dynamic suggestions for optimal thread pool and batch size configuration
- **🧪 FastAPI Development Server**: Built-in API for local testing and development

## Getting Started

### Installation

```bash
# Clone repository
git clone https://github.com/bogini/OmniParser-Production
cd OmniParser-Production

# Install dependencies
pip install -r requirements.txt
```

### Download Model Weights

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

```bash
# Start with default settings
python app.py

# Or customize host and port
python app.py --port 7861 --host 0.0.0.0
```

### Deploy to Modal Labs

```bash
# Install Modal CLI
pip install modal

# Log in to Modal
modal login

# Deploy the application
modal deploy app.py
```

After deployment, Modal provides a unique endpoint URL for your serverless API.

## API Usage

### Endpoints

#### 1. Process Single Image

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

#### 2. Process Multiple Images In Parallel

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

## Configuration Options

### Environment Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `CONCURRENCY_LIMIT` | Concurrent requests per container | 1 |
| `MODAL_CONTAINER_TIMEOUT` | Container idle timeout (seconds) | 500 |
| `MODAL_GPU_CONFIG` | GPU type for Modal deployment | A100 |
| `API_PORT` | FastAPI server port | 7861 |
| `MAX_CONTAINERS` | Maximum Modal containers | 10 |
| `MAX_BATCH_SIZE` | Maximum images per batch | 1000 |
| `THREAD_POOL_SIZE` | Thread pool size | 40 |

### PaddleOCR Pool Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `pool_size` | Number of PaddleOCR instances | 16 |
| `lang` | OCR language | en |
| `use_angle_cls` | Use angle classification | false |
| `use_gpu` | Use GPU for OCR | true |
| `max_batch_size` | Maximum OCR batch size | 1024 |
| `rec_batch_num` | Recognition batch number | 1024 |

### How to Configure

Set environment variables before starting the server:

```bash
# Set environment variables
export THREAD_POOL_SIZE=20
export MAX_BATCH_SIZE=500
export API_PORT=8000

# Start with custom configuration
python app.py
```

## Performance

OmniParser V2 delivers significant performance improvements:

- **Single Image**: ~0.55-0.7s processing time per image (1111×2405 pixels with OCR) on an A100 GPU
- **Batch Processing**: Near-linear scaling with thread pool size
- **Resource Efficiency**: Controlled memory usage with configurable OCR pool

Performance metrics are automatically logged with each request, providing insights for optimization.

## Example Output and Logs

The server provides detailed logging that gives insight into the processing pipeline:

```
2025-02-27 08:26:36,831 - omniparser - INFO - [batch_0b7beafa] Processing batch of 6 images in parallel
2025-02-27 08:26:36,831 - omniparser - INFO - [batch_0b7beafa] Submitting image 1/6 for processing
[...]
2025-02-27 08:26:37,849 - omniparser - INFO - [req_f61fc3be] Request to 'process_image' completed successfully in 1.018s | Steps: {"image_conversion": 0.0, "ocr_processing": 0.307, "icon_detection": 0.598, "response_preparation": 0.112} | {'image_width': 1179, 'image_height': 2556, 'text_elements': 1, 'icons_detected': 5}
[...]
2025-02-27 08:26:40,847 - omniparser - INFO - [batch_0b7beafa] Batch processing complete - Stats: Total: 6 | Successful: 6 | Failed: 0 | Time: 3.98s | Avg: 0.66s per image | Thread pool size: 40 | Parallelism efficiency: 0.11 | Image times - Avg: 2.88s | Min: 1.02s | Max: 4.02s Consider reducing THREAD_POOL_SIZE (current: 40)
```

### Understanding the Logs

Key insights from the logs:

1. **Request Tracking**: Each request and batch gets a unique ID
2. **Processing Breakdown**: Timing for each processing step
3. **Detection Results**: Image dimensions and element counts
4. **Performance Metrics**: Efficiency and utilization statistics
5. **Optimization Suggestions**: Automatic recommendations based on usage patterns

## License

This project is licensed under the CC-BY-4.0 License, the same as the original Microsoft OmniParser repository.
