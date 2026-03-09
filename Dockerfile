FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PORT=8000
ENV HF_HOME=/tmp/huggingface
ENV MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

# System libs for OpenCV / PIL / general runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

# Install Torch first, pinned to your current CUDA 12.4 stack
RUN pip install \
    torch==2.4.1 \
    torchvision==0.19.1 \
    torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu124

# Install runtime libs without letting them modify torch
RUN pip install --no-deps \
    fastapi \
    uvicorn[standard] \
    ultralytics==8.4.19 \
    open-clip-torch \
    pillow \
    numpy \
    requests \
    huggingface_hub \
    boto3 \
    opencv-python-headless \
    click \
    polars

# Extra packages commonly needed by ultralytics / open-clip
RUN pip install --no-deps \
    tqdm \
    pyyaml \
    scipy \
    psutil \
    matplotlib \
    ftfy \
    regex \
    safetensors \
    timm

COPY app.py /app/app.py

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
