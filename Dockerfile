FROM python:3.13-bookworm

USER root

RUN chmod 1777 /tmp \
    && apt update -q && apt install -y ca-certificates wget libgl1 \
    && wget -qO /tmp/cuda-keyring.deb  https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i /tmp/cuda-keyring.deb && apt update -q \
    && apt install -y --no-install-recommends libcudnn8 libcublas-12-2

RUN pip install -r requirements.txt
RUN pip install fastapi[all] 

RUN chmod +x scripts/download_weights.sh
RUN ./scripts/download_weights.sh
COPY . .

CMD ["fastapi", "run"]