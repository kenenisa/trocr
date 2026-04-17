import io
import os
import re
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, ImageOps
from pydantic import BaseModel
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


TX_ID_PATTERN = re.compile(r"\b[DC][A-Z0-9]{9}\b")
MODEL_NAME = os.getenv("TROCR_MODEL_NAME", "microsoft/trocr-small-printed")
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))

app = FastAPI(title="TrOCR Service", version="1.0.0")

processor: Optional[TrOCRProcessor] = None
model: Optional[VisionEncoderDecoderModel] = None
device = "cuda" if torch.cuda.is_available() else "cpu"


class OcrResponse(BaseModel):
  text: str
  tx_id: Optional[str]
  model: str
  device: str


def extract_tx_id(candidate_text: str) -> Optional[str]:
  upper = candidate_text.upper()
  regex_match = TX_ID_PATTERN.search(upper)
  if regex_match:
    return regex_match.group(0)

  # OCR often inserts separators; collapse and scan 10-char windows.
  compact = re.sub(r"[^A-Z0-9]", "", upper)
  for idx in range(0, max(0, len(compact) - 9)):
    window = compact[idx : idx + 10]
    if re.fullmatch(r"[DC][A-Z0-9]{9}", window):
      return window
  return None


def textline_crops(image: Image.Image) -> list[Image.Image]:
  width, height = image.size
  crops: list[Image.Image] = [image]

  # Also try lower sections where receipt IDs usually appear.
  crops.append(image.crop((0, int(height * 0.35), width, height)))
  crops.append(image.crop((0, int(height * 0.45), width, height)))

  gray = ImageOps.grayscale(image)
  arr = np.array(gray)
  dark_mask = arr < 175
  row_density = dark_mask.mean(axis=1)
  active_rows = np.where(row_density > 0.012)[0]

  if active_rows.size == 0:
    return crops

  groups: list[tuple[int, int]] = []
  start = int(active_rows[0])
  prev = int(active_rows[0])
  for row in active_rows[1:]:
    row = int(row)
    if row - prev > 6:
      groups.append((start, prev))
      start = row
    prev = row
  groups.append((start, prev))

  for y0, y1 in groups:
    if y1 - y0 < 16:
      continue
    pad_y = 12
    top = max(0, y0 - pad_y)
    bottom = min(height, y1 + pad_y)

    band = dark_mask[top:bottom, :]
    col_density = band.mean(axis=0)
    active_cols = np.where(col_density > 0.01)[0]
    if active_cols.size == 0:
      continue

    pad_x = 12
    left = max(0, int(active_cols[0]) - pad_x)
    right = min(width, int(active_cols[-1]) + pad_x)
    if right - left < 40:
      continue

    crops.append(image.crop((left, top, right, bottom)))

  return crops


def crop_variants(crop: Image.Image) -> list[Image.Image]:
  variants: list[Image.Image] = []
  gray = ImageOps.grayscale(crop)
  auto = ImageOps.autocontrast(gray)
  binary = auto.point(lambda px: 255 if px > 170 else 0)

  for base in [crop.convert("RGB"), auto.convert("RGB"), binary.convert("RGB")]:
    variants.append(base)
    # Upscale helps TrOCR on UI screenshots with tiny text.
    if base.width < 1400:
      scale = 2
      upscaled = base.resize(
        (base.width * scale, base.height * scale),
        resample=Image.Resampling.LANCZOS,
      )
      variants.append(upscaled)
  return variants


def decode_text(image: Image.Image) -> str:
  if processor is None or model is None:
    return ""
  pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
  with torch.no_grad():
    generated_ids = model.generate(
      pixel_values,
      max_new_tokens=96,
      num_beams=4,
      early_stopping=True,
      no_repeat_ngram_size=2,
    )
  return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


@app.on_event("startup")
def load_model() -> None:
  global processor, model
  processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
  model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
  model.to(device)
  model.eval()


@app.get("/healthz")
def healthz() -> dict:
  return {
    "ok": True,
    "model_loaded": processor is not None and model is not None,
    "model": MODEL_NAME,
    "device": device,
  }


@app.post("/ocr", response_model=OcrResponse)
async def run_ocr(file: UploadFile = File(...)) -> OcrResponse:
  if processor is None or model is None:
    raise HTTPException(status_code=503, detail="Model not loaded")

  raw = await file.read()
  if not raw:
    raise HTTPException(status_code=400, detail="Empty file")
  if len(raw) > MAX_UPLOAD_BYTES:
    raise HTTPException(status_code=413, detail="File too large")

  try:
    image = Image.open(io.BytesIO(raw)).convert("RGB")
  except Exception as exc:
    raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc

  attempts: list[str] = []
  tx_id: Optional[str] = None

  seen: set[str] = set()
  for crop in textline_crops(image):
    for variant in crop_variants(crop):
      text = decode_text(variant)
      if not text:
        continue
      if text in seen:
        continue
      seen.add(text)
      attempts.append(text)
      maybe_tx_id = extract_tx_id(text)
      if maybe_tx_id:
        tx_id = maybe_tx_id
        break
    if tx_id:
      break

  text = "\n".join(attempts) if attempts else ""

  return OcrResponse(
    text=text,
    tx_id=tx_id,
    model=MODEL_NAME,
    device=device,
  )
