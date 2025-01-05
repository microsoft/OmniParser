# Dockerfile for OmniParser with GPU support and OpenGL libraries
#
# This Dockerfile is intended to create an environment with NVIDIA CUDA
# support and the necessary dependencies to run the OmniParser project.
# The configuration is designed to support applications that rely on
# Python 3.12, OpenCV, Hugging Face transformers, and Gradio. Additionally,
# it includes steps to pull large files from Git LFS and a script to
# convert model weights from .safetensor to .pt format. The container
# runs a Gradio server by default, exposed on port 7861.
#
# Base image: nvidia/cuda:12.3.1-devel-ubuntu22.04
#
# Key features:
# - System dependencies for OpenGL to support graphical libraries.
# - Miniconda for Python 3.12, allowing for environment management.
# - Git Large File Storage (LFS) setup for handling large model files.
# - Requirement file installation, including specific versions of
#   OpenCV and Hugging Face Hub.
# - Entrypoint script execution with Gradio server configuration for
#   external access.

# If it is gpu enviroment, use nvidia/cuda:12.3.1-devel-ubuntu22.04, otherwise use ubuntu:22.04
# FROM nvidia/cuda:12.3.1-devel-ubuntu22.04
FROM docker.io/ubuntu:22.04

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
    # pip uninstall -y opencv-python opencv-python-headless && \
    # pip install --no-cache-dir opencv-python-headless==4.8.1.78 && \
    pip install -r requirements.txt && \
    pip install huggingface_hub

# Run download.py to fetch model weights and convert safetensors to .pt format
# RUN . /opt/conda/etc/profile.d/conda.sh && conda activate omni && \
#     python download.py && \
#     echo "Contents of weights directory:" && \
#     ls -lR weights && \
#     python weights/convert_safetensor_to_pt.py

# Expose the default Gradio port
EXPOSE 7861

# Configure Gradio to be accessible externally
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Copy and set permissions for entrypoint script
# COPY entrypoint.sh /usr/src/app/entrypoint.sh
# RUN chmod +x /usr/src/app/entrypoint.sh

# To debug, keep the container running
# CMD ["tail", "-f", "/dev/null"]

################################################################################################
# virtual display related setup --> from anthropic-quickstarts/computer-use-demo/Dockerfile

ENV DEBIAN_FRONTEND=noninteractive
ENV DEBIAN_PRIORITY=high

RUN apt-get update && \
    apt-get -y upgrade && \
    apt-get -y install \
    # UI Requirements
    xvfb \
    xterm \
    xdotool \
    scrot \
    imagemagick \
    sudo \
    mutter \
    x11vnc \
    # Python/pyenv reqs
    build-essential \
    libssl-dev  \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    curl \
    git \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    # Network tools
    net-tools \
    netcat \
    # PPA req
    software-properties-common && \
    # Userland apps
    sudo add-apt-repository ppa:mozillateam/ppa && \
    sudo apt-get install -y --no-install-recommends \
    libreoffice \
    firefox-esr \
    x11-apps \
    xpdf \
    gedit \
    xpaint \
    tint2 \
    galculator \
    pcmanfm \
    unzip && \
    apt-get clean

# Install noVNC
RUN git clone --branch v1.5.0 https://github.com/novnc/noVNC.git /opt/noVNC && \
    git clone --branch v0.12.0 https://github.com/novnc/websockify /opt/noVNC/utils/websockify && \
    ln -s /opt/noVNC/vnc.html /opt/noVNC/index.html

# setup user
ENV USERNAME=computeruse
ENV HOME=/home/$USERNAME
RUN useradd -m -s /bin/bash -d $HOME $USERNAME
RUN echo "${USERNAME} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
USER computeruse
WORKDIR $HOME

# setup python
RUN git clone https://github.com/pyenv/pyenv.git ~/.pyenv && \
    cd ~/.pyenv && src/configure && make -C src && cd .. && \
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc && \
    echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc && \
    echo 'eval "$(pyenv init -)"' >> ~/.bashrc
ENV PYENV_ROOT="$HOME/.pyenv"
ENV PATH="$PYENV_ROOT/bin:$PATH"
ENV PYENV_VERSION_MAJOR=3
ENV PYENV_VERSION_MINOR=11
ENV PYENV_VERSION_PATCH=6
ENV PYENV_VERSION=$PYENV_VERSION_MAJOR.$PYENV_VERSION_MINOR.$PYENV_VERSION_PATCH
RUN eval "$(pyenv init -)" && \
    pyenv install $PYENV_VERSION && \
    pyenv global $PYENV_VERSION && \
    pyenv rehash

ENV PATH="$HOME/.pyenv/shims:$HOME/.pyenv/bin:$PATH"

RUN python -m pip install --upgrade pip==23.1.2 setuptools==58.0.4 wheel==0.40.0 && \
    python -m pip config set global.disable-pip-version-check true

# only reinstall if requirements.txt changes
# COPY --chown=$USERNAME:$USERNAME computer_use_demo/requirements.txt $HOME/computer_use_demo/requirements.txt
# RUN python -m pip install -r $HOME/computer_use_demo/requirements.txt

# setup desktop env & app
# COPY --chown=$USERNAME:$USERNAME image/ $HOME
# COPY --chown=$USERNAME:$USERNAME computer_use_demo/ $HOME/computer_use_demo/

ARG DISPLAY_NUM=1
ARG HEIGHT=768
ARG WIDTH=1024
ENV DISPLAY_NUM=$DISPLAY_NUM
ENV HEIGHT=$HEIGHT
ENV WIDTH=$WIDTH

# Set the entrypoint
# ENTRYPOINT ["/usr/src/app/entrypoint.sh"]

#  sudo docker build . -t omniparser-x-demo:local  # manually build the docker image (optional)
