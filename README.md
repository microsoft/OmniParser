# OmniParser for Pure Vision Based General GUI Agent

![arXiv](https://img.shields.io/badge/arXiv-2408.00203-b31b1b.svg)

OmniParser is a screen parsing tool that converts general GUI screens to structured elements.

## Overview

OmniParser processes screenshots of user interfaces and identifies UI elements along with their descriptions, positions, and relationships. It can be used to analyze UIs for automated testing, accessibility evaluation, or as input for AI agents.

The tool uses:
- YOLO for element detection
- OCR for text recognition
- Florence-2 model for icon/element captioning

## Installation

### Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Download Weights

Download the model weights:

```bash
# Create weights directory
mkdir -p weights/icon_detect weights/icon_caption_florence

# Download the model weights (These commands are placeholders, replace with actual download links)
wget -O weights/icon_detect/model.pt [YOUR_MODEL_DOWNLOAD_LINK]
# Download Florence caption model to weights/icon_caption_florence/ directory
```

## Usage

### Running the OmniParser API

You can run OmniParser as either a local FastAPI server or deploy it using Modal:

#### Local FastAPI Server

```bash
python endpoint.py --port 7861 --host 0.0.0.0
```

#### Deploy to Modal

```bash
modal deploy endpoint.py
```

### API Endpoints

The API provides two main endpoints:

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

#### 2. Process Multiple Images (Batch)

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

### Response Format

```json
{
  "processed_image": "base64_encoded_image_with_detections",
  "parsed_content": "icon 0: search button\nicon 1: menu button\n..."
}
```

### Demo UI

To run the Gradio demo UI:

```bash
python gradio_demo.py
```

This will start a web interface on http://localhost:7861 where you can upload images and see the parsed results.

## Troubleshooting

### Common Issues

1. **GPU issues**: If you encounter CUDA-related errors, try setting `use_paddleocr=False` in your request.

2. **Memory errors**: Lower the `imgsz` parameter (e.g., to 320) if you're running out of memory.

3. **Slow performance**: The caption generation can be slow on CPU. For better performance, use a GPU.

## Citation

If you use OmniParser in your research, please cite:

```
@misc{omniparser2024,
    title={OmniParser: Pure Vision Based General GUI Agent},
    author={Authors},
    year={2024},
    eprint={2408.00203},
    archivePrefix={arXiv}
}
```

## License

This project is licensed under the terms of the LICENSE file included in the repository.
