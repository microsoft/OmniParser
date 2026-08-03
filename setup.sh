#!/bin/bash

# Create a Python virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Create weights directory if it doesn't exist
mkdir -p weights

# Download model checkpoints
for f in icon_detect/{train_args.yaml,model.pt,model.yaml} icon_caption/{config.json,generation_config.json,model.safetensors}; do
    huggingface-cli download microsoft/OmniParser-v2.0 "$f" --local-dir weights
done

# Rename icon_caption to icon_caption_florence
if [ -d "weights/icon_caption" ]; then
    mv weights/icon_caption weights/icon_caption_florence
fi

echo "Setup completed! To start using OmniParser:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Run the demo: python gradio_demo.py"