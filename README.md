# Fashion Segmentation & Embedding API

A FastAPI service for fashion item detection, segmentation, and CLIP embedding generation. Designed to run on RunPod with GPU support.

## Features

- **Image Segmentation** – Detects and segments fashion items using YOLOv8 (trained on Fashionpedia)
- **CLIP Embeddings** – Generates embeddings for images or text using OpenCLIP (ViT-L-14)
- **S3 Upload** – Optionally uploads cropped images to S3-compatible storage (AWS, Cloudflare R2, MinIO)

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check and config info |
| `POST` | `/embed` | Generate CLIP embedding for an image URL or text |
| `POST` | `/crop-and-embed` | Segment image, crop items, and return embeddings |
| `POST` | `/crop` | Segment image, crop items, and upload to S3 |

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SEG_MODEL_REPO` | HuggingFace repo for YOLO weights | `Samuel77/YOLOv8s-seg-fashionpedia` |
| `SEG_MODEL_FILENAME` | Path to weights in repo | `weights/fp13_from_df2/best.pt` |
| `HF_TOKEN` | HuggingFace token (if repo is private) | – |
| `S3_BUCKET_NAME` | S3 bucket name | – |
| `S3_ENDPOINT_URL` | S3 endpoint (for R2/MinIO) | – |
| `S3_REGION` | S3 region | `us-east-1` |
| `PUBLIC_BUCKET_BASE_URL` | Public URL prefix for uploaded files | – |
| `DEFAULT_IMGSZ` | Default image size for inference | `832` |
| `DEFAULT_CONF` | Default confidence threshold | `0.25` |
| `DEFAULT_IOU` | Default IoU threshold | `0.7` |

## Quick Start

### Local Development

```bash
pip install torch torchvision fastapi uvicorn ultralytics open-clip-torch pillow numpy requests huggingface_hub boto3

uvicorn app:app --reload
```

### Docker

```bash
docker build -t fashion-seg-api .
docker run -p 8000:8000 --gpus all fashion-seg-api
```

## Usage Examples

### Generate Image Embedding

```bash
curl -X POST http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://example.com/outfit.jpg"}'
```

### Segment & Embed

```bash
curl -X POST http://localhost:8000/crop-and-embed \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://example.com/outfit.jpg"}'
```

## License

MIT
