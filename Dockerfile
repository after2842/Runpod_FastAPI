FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PORT=8000
ENV HF_HOME=/tmp/huggingface
ENV MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel


RUN pip install \
    torch==2.4.1 \
    torchvision==0.19.1 \
    torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu124


RUN pip install \
    fastapi \
    uvicorn

RUN pip install \
    pillow \
    numpy \
    requests \
    huggingface_hub \
    boto3 \
    opencv-python-headless \
    polars \
    tqdm \
    pyyaml \
    scipy \
    psutil \
    matplotlib \
    ftfy \
    regex \
    safetensors \
    timm \
    ultralytics-thop


RUN pip install --no-deps \
    ultralytics==8.4.19 \
    open-clip-torch

COPY app.py /app/app.py

RUN python - <<'PY'
import fastapi, uvicorn, h11, starlette, pydantic, anyio
import boto3, numpy, requests, open_clip
from ultralytics import YOLO
print("import smoke test OK")
PY

EXPOSE 8000

CMD ["sh", "-c", "exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
