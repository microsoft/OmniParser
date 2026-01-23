FROM python:3.12-slim

WORKDIR /app

# Prevent .pyc files and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System deps (minimal, no GUI libs needed for headless OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .

# Pip install requirements
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Download model weights at build time and flatten folders
RUN mkdir -p weights/icon_detect weights/icon_caption_florence \
    \
    # Download icon_detect weights
    && huggingface-cli download microsoft/OmniParser-v2.0 "icon_detect/train_args.yaml" --local-dir weights/icon_detect \
    && huggingface-cli download microsoft/OmniParser-v2.0 "icon_detect/model.pt" --local-dir weights/icon_detect \
    && huggingface-cli download microsoft/OmniParser-v2.0 "icon_detect/model.yaml" --local-dir weights/icon_detect \
    # Flatten icon_detect in case Hugging Face created a nested folder
    && if [ -d weights/icon_detect/icon_detect ]; then mv weights/icon_detect/icon_detect/* weights/icon_detect/ && rmdir weights/icon_detect/icon_detect; fi \
    \
    # Download icon_caption_florence weights
    && huggingface-cli download microsoft/OmniParser-v2.0 "icon_caption/config.json" --local-dir weights/icon_caption_florence \
    && huggingface-cli download microsoft/OmniParser-v2.0 "icon_caption/generation_config.json" --local-dir weights/icon_caption_florence \
    && huggingface-cli download microsoft/OmniParser-v2.0 "icon_caption/model.safetensors" --local-dir weights/icon_caption_florence \
    # Flatten icon_caption folder if Hugging Face created a nested folder
    && if [ -d weights/icon_caption_florence/icon_caption ]; then mv weights/icon_caption_florence/icon_caption/* weights/icon_caption_florence/ && rmdir weights/icon_caption_florence/icon_caption; fi

# Copy app source
COPY . .

# Change working directory to the server folder and run with full arguments
WORKDIR /app/omnitool/omniparserserver

CMD ["python", "-m", "omniparserserver", \
     "--som_model_path", "../../weights/icon_detect/model.pt", \
     "--caption_model_name", "florence2", \
     "--caption_model_path", "../../weights/icon_caption_florence", \
     "--device", "cuda", \
     "--BOX_TRESHOLD", "0.05"]
