from __future__ import annotations

import io
import os
import re
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3
import numpy as np
import open_clip
import requests
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field
from ultralytics import YOLO


# =========================================================
# Config
# =========================================================
SEG_MODEL_REPO = os.getenv("SEG_MODEL_REPO", "Samuel77/YOLOv8s-seg-fashionpedia")
SEG_MODEL_FILENAME = os.getenv("SEG_MODEL_FILENAME", "weights/fp13_from_df2/best.pt")
HF_TOKEN = os.getenv("HF_TOKEN")

OPENCLIP_WEIGHTS_URL = os.getenv(
    "OPENCLIP_WEIGHTS_URL",
    "https://huggingface.co/laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K/resolve/main/open_clip_pytorch_model.bin",
)
OPENCLIP_MODEL_NAME = os.getenv("OPENCLIP_MODEL_NAME", "ViT-L-14")

LOCAL_MODEL_DIR = Path(os.getenv("LOCAL_MODEL_DIR", "/tmp/models"))
LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL")  # optional for Cloudflare R2 / MinIO / other S3-compatible
S3_REGION = os.getenv("S3_REGION", "us-east-1")
PUBLIC_BUCKET_BASE_URL = os.getenv("PUBLIC_BUCKET_BASE_URL")  # optional, e.g. https://bucket.s3.amazonaws.com

DEFAULT_IMGSZ = int(os.getenv("DEFAULT_IMGSZ", "832"))
DEFAULT_CONF = float(os.getenv("DEFAULT_CONF", "0.25"))
DEFAULT_IOU = float(os.getenv("DEFAULT_IOU", "0.7"))
DEFAULT_MAX_DET = int(os.getenv("DEFAULT_MAX_DET", "50"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEG_MODEL: Optional[YOLO] = None
CLIP_MODEL = None
CLIP_PREPROCESS = None
S3_CLIENT = None


# =========================================================
# Request schemas
# =========================================================
class CropAndEmbedRequest(BaseModel):
    image_url: str
    imgsz: int = DEFAULT_IMGSZ
    conf: float = DEFAULT_CONF
    iou: float = DEFAULT_IOU
    max_det: int = DEFAULT_MAX_DET


class EmbedRequest(BaseModel):
    image_url: Optional[str] = None
    text: Optional[str] = None


class CropRequest(BaseModel):
    username: str = Field(..., min_length=1)
    image_url: str
    imgsz: int = DEFAULT_IMGSZ
    conf: float = DEFAULT_CONF
    iou: float = DEFAULT_IOU
    max_det: int = DEFAULT_MAX_DET


# =========================================================
# Utilities
# =========================================================
def sanitize_name(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "item"


def now_utc_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def clamp_bbox(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(1, min(int(x2), w))
    y2 = max(1, min(int(y2), h))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return x1, y1, x2, y2


def download_file(url: str, dst: Path) -> Path:
    if dst.exists() and dst.stat().st_size > 0:
        return dst

    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    return dst


def download_image_from_url(url: str) -> Image.Image:
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download image: {e}")


def build_s3_client():
    if not S3_BUCKET_NAME:
        return None

    session = boto3.session.Session()
    return session.client(
        "s3",
        endpoint_url=S3_ENDPOINT_URL,
        region_name=S3_REGION,
    )


def upload_bytes_to_bucket(content: bytes, key: str, content_type: str) -> Dict[str, Any]:
    if S3_CLIENT is None or not S3_BUCKET_NAME:
        raise HTTPException(status_code=500, detail="Bucket is not configured")

    S3_CLIENT.put_object(
        Bucket=S3_BUCKET_NAME,
        Key=key,
        Body=content,
        ContentType=content_type,
    )

    payload = {
        "bucket": S3_BUCKET_NAME,
        "key": key,
    }

    if PUBLIC_BUCKET_BASE_URL:
        payload["url"] = f"{PUBLIC_BUCKET_BASE_URL.rstrip('/')}/{key}"

    return payload


def pil_to_jpeg_bytes(img: Image.Image, quality: int = 95) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def draw_boxes_on_image(
    image: Image.Image,
    boxes_xyxy: np.ndarray,
    class_ids: np.ndarray,
    confs: np.ndarray,
    names: Dict[int, str],
) -> Image.Image:
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for i in range(len(boxes_xyxy)):
        x1, y1, x2, y2 = [int(v) for v in boxes_xyxy[i]]
        label = f"{names[int(class_ids[i])]} {float(confs[i]):.2f}"

        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        if font is not None:
            left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
            tw, th = right - left, bottom - top
        else:
            tw, th = len(label) * 6, 12

        tx = x1
        ty = y1 - th - 4
        if ty < 0:
            ty = y1 + 2

        draw.rectangle([tx, ty, tx + tw + 6, ty + th + 4], fill="red")
        draw.text((tx + 3, ty + 2), label, fill="white", font=font)

    return img


def ensure_mask_shape(mask: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    if mask.shape[0] == target_h and mask.shape[1] == target_w:
        return mask

    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    mask_img = mask_img.resize((target_w, target_h), resample=Image.NEAREST)
    return (np.array(mask_img).astype(np.float32) / 255.0)


def segment_image(
    image: Image.Image,
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
) -> Dict[str, Any]:
    assert SEG_MODEL is not None

    image_rgb = np.array(image.convert("RGB"))
    h, w = image_rgb.shape[:2]

    results = SEG_MODEL.predict(
        source=image_rgb,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        max_det=max_det,
        retina_masks=True,
        verbose=False,
    )

    r = results[0]

    if r.boxes is None or len(r.boxes) == 0:
        return {
            "boxes_xyxy": np.zeros((0, 4), dtype=np.float32),
            "class_ids": np.zeros((0,), dtype=np.int64),
            "confs": np.zeros((0,), dtype=np.float32),
            "masks": np.zeros((0, h, w), dtype=np.float32),
            "names": SEG_MODEL.names,
            "image_rgb": image_rgb,
        }

    boxes_xyxy = r.boxes.xyxy.cpu().numpy()
    class_ids = r.boxes.cls.cpu().numpy().astype(int)
    confs = r.boxes.conf.cpu().numpy()

    if r.masks is not None and r.masks.data is not None:
        masks = r.masks.data.cpu().numpy()
        if len(masks) > 0:
            resized = []
            for m in masks:
                resized.append(ensure_mask_shape(m, h, w))
            masks = np.stack(resized, axis=0)
        else:
            masks = np.zeros((0, h, w), dtype=np.float32)
    else:
        masks = np.zeros((0, h, w), dtype=np.float32)

    return {
        "boxes_xyxy": boxes_xyxy,
        "class_ids": class_ids,
        "confs": confs,
        "masks": masks,
        "names": SEG_MODEL.names,
        "image_rgb": image_rgb,
    }


def make_crops(seg: Dict[str, Any]) -> List[Dict[str, Any]]:
    image_rgb = seg["image_rgb"]
    boxes_xyxy = seg["boxes_xyxy"]
    class_ids = seg["class_ids"]
    confs = seg["confs"]
    names = seg["names"]
    masks = seg["masks"]

    h, w = image_rgb.shape[:2]
    outputs = []

    for i in range(len(boxes_xyxy)):
        x1, y1, x2, y2 = clamp_bbox(*boxes_xyxy[i], w, h)

        crop_rgb = image_rgb[y1:y2, x1:x2]
        bbox_pil = Image.fromarray(crop_rgb).convert("RGB")

        if len(masks) > i:
            mask = masks[i] > 0.5
            crop_mask = mask[y1:y2, x1:x2]

            masked_rgb = np.empty_like(crop_rgb)
            masked_rgb[:] = np.array((255, 255, 255), dtype=np.uint8)
            masked_rgb[crop_mask] = crop_rgb[crop_mask]

            rgba = np.zeros((crop_rgb.shape[0], crop_rgb.shape[1], 4), dtype=np.uint8)
            rgba[..., :3] = crop_rgb
            rgba[..., 3] = crop_mask.astype(np.uint8) * 255

            masked_rgb_pil = Image.fromarray(masked_rgb).convert("RGB")
            masked_rgba_pil = Image.fromarray(rgba, mode="RGBA")
        else:
            masked_rgb_pil = bbox_pil
            masked_rgba_pil = bbox_pil.convert("RGBA")

        outputs.append(
            {
                "class_id": int(class_ids[i]),
                "class_name": names[int(class_ids[i])],
                "confidence": float(confs[i]),
                "bbox_xyxy": [float(v) for v in boxes_xyxy[i]],
                "bbox_pil": bbox_pil,
                "masked_rgb_pil": masked_rgb_pil,
                "masked_rgba_pil": masked_rgba_pil,
            }
        )

    return outputs


def embed_pil_images(images: List[Image.Image]) -> List[List[float]]:
    assert CLIP_MODEL is not None
    assert CLIP_PREPROCESS is not None

    if not images:
        return []

    batch = torch.stack([CLIP_PREPROCESS(img.convert("RGB")) for img in images]).to(DEVICE)

    with torch.no_grad():
        feats = CLIP_MODEL.encode_image(batch)
        feats = F.normalize(feats, dim=-1)

    return feats.cpu().tolist()


def embed_text(text: str) -> List[float]:
    assert CLIP_MODEL is not None

    tokens = open_clip.tokenize([text]).to(DEVICE)
    with torch.no_grad():
        feats = CLIP_MODEL.encode_text(tokens)
        feats = F.normalize(feats, dim=-1)

    return feats[0].cpu().tolist()


# =========================================================
# Startup / app lifespan
# =========================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global SEG_MODEL, CLIP_MODEL, CLIP_PREPROCESS, S3_CLIENT

    # YOLO weights from Hugging Face repo
    seg_path = hf_hub_download(
        repo_id=SEG_MODEL_REPO,
        filename=SEG_MODEL_FILENAME,
        token=HF_TOKEN,
    )

    # OpenCLIP weights from direct URL
    clip_weights_path = LOCAL_MODEL_DIR / "open_clip_pytorch_model.bin"
    download_file(OPENCLIP_WEIGHTS_URL, clip_weights_path)

    # Load models once
    SEG_MODEL = YOLO(seg_path)

    CLIP_MODEL, _, CLIP_PREPROCESS = open_clip.create_model_and_transforms(
        OPENCLIP_MODEL_NAME,
        pretrained=str(clip_weights_path),
    )
    CLIP_MODEL = CLIP_MODEL.to(DEVICE)
    CLIP_MODEL.eval()

    S3_CLIENT = build_s3_client()

    yield


app = FastAPI(title="fashion-seg-embed-api", lifespan=lifespan)


# =========================================================
# Endpoints
# =========================================================
@app.get("/health")
def health():
    return {
        "ok": True,
        "device": DEVICE,
        "seg_model_repo": SEG_MODEL_REPO,
        "seg_model_filename": SEG_MODEL_FILENAME,
        "bucket_configured": S3_CLIENT is not None,
    }


@app.post("/embed")
def embed(req: EmbedRequest):
    if not req.image_url and not req.text:
        raise HTTPException(status_code=400, detail="Provide either image_url or text")

    if req.image_url and req.text:
        raise HTTPException(status_code=400, detail="Provide only one of image_url or text")

    if req.image_url:
        image = download_image_from_url(req.image_url)
        embedding = embed_pil_images([image])[0]
        return {
            "type": "image",
            "image_url": req.image_url,
            "embedding": embedding,
        }

    embedding = embed_text(req.text or "")
    return {
        "type": "text",
        "text": req.text,
        "embedding": embedding,
    }


@app.post("/crop-and-embed")
def crop_and_embed(req: CropAndEmbedRequest):
    image = download_image_from_url(req.image_url)

    seg = segment_image(
        image=image,
        imgsz=req.imgsz,
        conf=req.conf,
        iou=req.iou,
        max_det=req.max_det,
    )
    crops = make_crops(seg)

    masked_images = [c["masked_rgb_pil"] for c in crops]
    embeddings = embed_pil_images(masked_images)

    detections = []
    for c, emb in zip(crops, embeddings):
        detections.append(
            {
                "class_name": c["class_name"],
                "confidence": c["confidence"],
                "bbox_xyxy": c["bbox_xyxy"],
                "embedding": emb,
            }
        )

    return {
        "image_url": req.image_url,
        "num_detections": len(detections),
        "detections": detections,
    }


@app.post("/crop")
def crop(req: CropRequest):
    if S3_CLIENT is None or not S3_BUCKET_NAME:
        raise HTTPException(
            status_code=500,
            detail="Bucket is not configured. Set S3_BUCKET_NAME and credentials env vars.",
        )

    image = download_image_from_url(req.image_url)

    seg = segment_image(
        image=image,
        imgsz=req.imgsz,
        conf=req.conf,
        iou=req.iou,
        max_det=req.max_det,
    )
    crops = make_crops(seg)

    # Upload annotated image
    annotated = draw_boxes_on_image(
        image=image,
        boxes_xyxy=seg["boxes_xyxy"],
        class_ids=seg["class_ids"],
        confs=seg["confs"],
        names=seg["names"],
    )

    username = sanitize_name(req.username)
    ts = now_utc_str()
    request_id = uuid.uuid4().hex[:10]

    annotated_key = f"{username}/{ts}_{request_id}_annotated.jpg"
    annotated_upload = upload_bytes_to_bucket(
        pil_to_jpeg_bytes(annotated),
        annotated_key,
        "image/jpeg",
    )

    detections = []
    for idx, c in enumerate(crops):
        cls_name = sanitize_name(c["class_name"])
        conf_str = f"{c['confidence']:.2f}".replace(".", "_")

        masked_key = f"{username}/{ts}_{request_id}_{idx:02d}_{cls_name}_{conf_str}_masked.png"
        bbox_key = f"{username}/{ts}_{request_id}_{idx:02d}_{cls_name}_{conf_str}_bbox.jpg"

        masked_upload = upload_bytes_to_bucket(
            pil_to_png_bytes(c["masked_rgba_pil"]),
            masked_key,
            "image/png",
        )
        bbox_upload = upload_bytes_to_bucket(
            pil_to_jpeg_bytes(c["bbox_pil"]),
            bbox_key,
            "image/jpeg",
        )

        detections.append(
            {
                "class_name": c["class_name"],
                "confidence": c["confidence"],
                "bbox_xyxy": c["bbox_xyxy"],
                "masked_image": masked_upload,
                "bbox_image": bbox_upload,
            }
        )

    return {
        "image_url": req.image_url,
        "annotated_image": annotated_upload,
        "num_detections": len(detections),
        "detections": detections,
    }