# Dockerfile for OmniParser with GPU and OpenGL support.
#
# Base: nvidia/cuda:12.3.1-devel-ubuntu22.04
# Features:
# - Python 3.12 with Miniconda environment.
# - Git LFS for large file support.
# - Required libraries: OpenCV, Hugging Face, Gradio, OpenGL.
# - Gradio server on port 7861.
#
# 1. Build the image with CUDA support.
# Example:
#   ```bash
#   sudo nvidia-docker build -t omniparser .
#   ```
#
# 2. Run the Docker container with GPU access and port mapping for Gradio.
# Example:
#   ```bash
#   sudo docker run -d -p 7861:7861 --gpus all --name omniparser-container omniparser
#   ```

FROM nvidia/cuda:12.3.1-devel-ubuntu22.04

# Install system dependencies with explicit OpenGL libraries
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git \
    git-lfs \
    wget \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libglu1-mesa \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    python3-opencv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# Install Miniconda for Python 3.12
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh
ENV PATH="/opt/conda/bin:$PATH"

# Create and activate Conda environment with Python 3.12, and set it as the default
RUN conda create -n omni python=3.12 && \
    echo "source activate omni" > ~/.bashrc
ENV CONDA_DEFAULT_ENV=omni
ENV PATH="/opt/conda/envs/omni/bin:$PATH"

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy project files and requirements
COPY . .
COPY requirements.txt /usr/src/app/requirements.txt

# Initialize Git LFS and pull LFS files
RUN git lfs install && \
    git lfs pull

# Install dependencies from requirements.txt with specific opencv-python-headless version
RUN . /opt/conda/etc/profile.d/conda.sh && conda activate omni && \
    pip uninstall -y opencv-python opencv-python-headless && \
    pip install --no-cache-dir opencv-python-headless==4.8.1.78 && \
    pip install -r requirements.txt && \
    pip install huggingface_hub

# Run download.py to fetch model weights and convert safetensors to .pt format
RUN . /opt/conda/etc/profile.d/conda.sh && conda activate omni && \
    python download.py && \
    echo "Contents of weights directory:" && \
    ls -lR weights && \
    python weights/convert_safetensor_to_pt.py

# Expose the default Gradio port
EXPOSE 7861

# Configure Gradio to be accessible externally
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Copy and set permissions for entrypoint script
COPY entrypoint.sh /usr/src/app/entrypoint.sh
RUN chmod +x /usr/src/app/entrypoint.sh

# To debug, keep the container running
# CMD ["tail", "-f", "/dev/null"]

# Set the entrypoint
ENTRYPOINT ["/usr/src/app/entrypoint.sh"]
