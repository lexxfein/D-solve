from pathlib import Path
import base64
import io
import sys

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cv.alignment.alignment import align_smile
from cv.veneer.veneer import veneer_smile
from cv.whitening.whitening import whiten_smile

app = FastAPI(
    title="Digital Smile Design API",
    version="1.0.0",
    description="Bridge between the TypeScript frontend and the team's computer-vision pipeline.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED = {"alignment", "whitening", "veneers", "combined"}

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "digital-smile-design"}

async def decode_image(file: UploadFile) -> np.ndarray:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image.")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="The image could not be decoded.")
    return image

def encode_jpeg(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode processed image.")
    return encoded.tobytes()

@app.post("/api/simulate")
async def simulate(
    file: UploadFile = File(...),
    treatment: str = Form("whitening"),
    intensity: int = Form(50),
):
    treatment = treatment.lower().strip()
    if treatment not in ALLOWED:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported treatment. Use one of: {', '.join(sorted(ALLOWED))}",
        )
    intensity = max(0, min(100, int(intensity)))
    image = await decode_image(file)

    try:
        if treatment == "alignment":
            result = align_smile(image, intensity)
        elif treatment == "whitening":
            result = whiten_smile(image, intensity)
        elif treatment == "veneers":
            result = veneer_smile(image, intensity)
        else:
            # Combined demo pipeline: whitening -> alignment -> veneers.
            result = whiten_smile(image, intensity)
            result = align_smile(result, intensity)
            result = veneer_smile(result, intensity)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return Response(
        content=encode_jpeg(result),
        media_type="image/jpeg",
        headers={
            "X-Treatment": treatment,
            "X-Intensity": str(intensity),
        },
    )
