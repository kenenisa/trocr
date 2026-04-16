import io
import os
import re
from typing import Optional

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
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

  pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

  with torch.no_grad():
    generated_ids = model.generate(pixel_values, max_new_tokens=128)

  text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
  upper = text.upper()
  match = TX_ID_PATTERN.search(upper)
  tx_id = match.group(0) if match else None

  return OcrResponse(
    text=text,
    tx_id=tx_id,
    model=MODEL_NAME,
    device=device,
  )
