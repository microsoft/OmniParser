# OmniParserV2 Modal

This repository is a fork of [Microsoft's OmniParser](https://github.com/microsoft/OmniParser) that adds support for running OmniParser on [Modal Labs](https://modal.com/). For full details about OmniParser itself, please refer to the [original repository](https://github.com/microsoft/OmniParser).

## What's Different in this Fork?

This fork adapts the original OmniParser's Gradio demo for Modal deployment with several key changes:

### Image Handling Improvements
- Enhanced image input processing in [`gradio_demo.py`](gradio_demo.py) to handle both base64-encoded images and direct file uploads
- Added robust type checking and conversion for various image input formats (numpy arrays, PIL Images, base64 strings)
- Improved error handling for image processing failures

### Modal-Specific Changes
- New [`modal_app.py`](modal_app.py) that configures the Modal deployment environment:
  - Sets up a Debian-based container with required system libraries
  - Installs all Python dependencies via pip
  - Copies model weights and utility files to the container
  - Configures GPU acceleration using H100 instances
  - Handles web endpoint configuration and threading
  - Includes inactivity monitor that closes app after 1 minute without requests

### Architecture Changes
- Split the original monolithic Gradio demo into:
  - Core demo logic in [`gradio_demo.py`](gradio_demo.py) that can run locally or in Modal
  - Modal-specific configuration and deployment code in [`modal_app.py`](modal_app.py)
- Added proper model initialization and resource management for Modal's serverless environment

## Running on Modal

1. First, make sure you have Modal installed and configured:

```bash
pip install modal
modal token new
```

2. Run the application:

```bash
modal run modal_app.py
```

This will:

- Set up a Modal environment with all required dependencies
- Download and configure the necessary model weights
- Launch a Gradio interface accessible via a public URL
- Provide GPU acceleration using an A100 instance

## License

This fork maintains the same licensing as the original OmniParser:

- Icon detection model: AGPL license
- Icon caption models (BLIP2 & Florence): MIT license

For full license details, see the original repository's LICENSE files.
