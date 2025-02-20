# OmniParserV2 Modal

This repository is a fork of [Microsoft's OmniParser](https://github.com/microsoft/OmniParser) that adds support for running OmniParser on [Modal Labs](https://modal.com/). For full details about OmniParser itself, please refer to the [original repository](https://github.com/microsoft/OmniParser).

## What's Different in this Fork?

This fork adds Modal deployment support, allowing you to run OmniParser in Modal's cloud infrastructure with GPU acceleration. The main additions are:

- Modal-specific deployment configuration
- Pre-configured GPU environment setup
- Streamlined model weight handling
- Automatic dependency management
- Support for remote image uploads (base64 and file uploads)

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

## Environment & Dependencies

The Modal configuration automatically handles all dependencies, including:

- Python packages (torch, gradio, etc.)
- System dependencies
- Model weights
- GPU configuration

## License

This fork maintains the same licensing as the original OmniParser:

- Icon detection model: AGPL license
- Icon caption models (BLIP2 & Florence): MIT license

For full license details, see the original repository's LICENSE files.
