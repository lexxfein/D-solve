from pathlib import Path
import json
import sys

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cv.alignment.alignment import align_smile  # kept for the existing 2D CV module/tests
from cv.alignment.alignment3d import auto_align, inspect_model, manual_transform
from cv.veneer.veneer import veneer_smile
from cv.whitening.whitening import whiten_smile

app = FastAPI(
    title="Digital Smile Design API",
    version="2.0.0",
    description="Local backend for the D-Solve 2D smile simulation and 3D dental-model alignment workflows.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_2D = {"whitening", "veneers", "combined"}


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "digital-smile-design", "three_d_alignment": True}


async def read_bytes(file: UploadFile, *, max_mb: int = 80) -> bytes:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(data) > max_mb * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"File must be smaller than {max_mb} MB.")
    return data


async def decode_image(file: UploadFile) -> np.ndarray:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image.")
    data = await read_bytes(file, max_mb=10)
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="The image could not be decoded.")
    return image


def encode_jpeg(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode processed image.")
    return encoded.tobytes()


def model_filename(file: UploadFile) -> str:
    return file.filename or "model.glb"


@app.post("/api/simulate")
async def simulate(
    file: UploadFile = File(...),
    treatment: str = Form("whitening"),
    intensity: int = Form(50),
):
    treatment = treatment.lower().strip()
    if treatment == "alignment":
        raise HTTPException(
            status_code=400,
            detail="2D alignment has been moved to the 3D Alignment workflow. Upload a dental 3D model there.",
        )
    if treatment not in ALLOWED_2D:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported 2D treatment. Use one of: {', '.join(sorted(ALLOWED_2D))}",
        )
    intensity = max(0, min(100, int(intensity)))
    image = await decode_image(file)

    try:
        if treatment == "whitening":
            result = whiten_smile(image, intensity)
        elif treatment == "veneers":
            result = veneer_smile(image, intensity)
        else:
            # Keep the existing 2D workflows while removing alignment from this path.
            result = whiten_smile(image, intensity)
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


@app.post("/api/alignment/info")
async def alignment_info(file: UploadFile = File(...)):
    data = await read_bytes(file)
    try:
        info = inspect_model(data, model_filename(file))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return info.__dict__


@app.post("/api/alignment/auto")
async def alignment_auto(
    file: UploadFile = File(...),
    strength: float = Form(100),
):
    data = await read_bytes(file)
    strength = max(0.0, min(100.0, float(strength)))
    try:
        output, transform = auto_align(data, model_filename(file), strength)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return Response(
        content=output,
        media_type="model/gltf-binary",
        headers={
            "Content-Disposition": 'attachment; filename="dsolve_auto_aligned.glb"',
            "X-Alignment-Transform": json.dumps(transform),
        },
    )


@app.post("/api/alignment/transform")
async def alignment_transform(
    file: UploadFile = File(...),
    tx: float = Form(0),
    ty: float = Form(0),
    tz: float = Form(0),
    rx: float = Form(0),
    ry: float = Form(0),
    rz: float = Form(0),
    scale: float = Form(1),
):
    data = await read_bytes(file)
    try:
        output = manual_transform(
            data,
            model_filename(file),
            tx=float(tx),
            ty=float(ty),
            tz=float(tz),
            rx=float(rx),
            ry=float(ry),
            rz=float(rz),
            scale=float(scale),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return Response(
        content=output,
        media_type="model/gltf-binary",
        headers={"Content-Disposition": 'attachment; filename="dsolve_aligned.glb"'},
    )
