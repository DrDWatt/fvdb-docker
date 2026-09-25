"""
Depth Estimation Service using Depth Anything V2.
Provides GPU-accelerated monocular depth estimation via a REST API.
"""

import io
import logging
import time
from typing import Optional

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import Response
from PIL import Image
from transformers import pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("depth-service")

app = FastAPI(title="Depth Estimation Service", docs_url="/api")

# Global model pipeline
depth_pipe = None
MODEL_NAME = "depth-anything/Depth-Anything-V2-Small-hf"


def get_depth_pipeline():
    """Lazy-load the depth estimation model"""
    global depth_pipe
    if depth_pipe is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading {MODEL_NAME} on {device}...")
        depth_pipe = pipeline("depth-estimation", model=MODEL_NAME, device=device)
        logger.info("Model loaded successfully")
    return depth_pipe


@app.on_event("startup")
async def startup():
    """Pre-load model at startup"""
    get_depth_pipeline()


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_NAME, "gpu": torch.cuda.is_available()}


@app.post("/estimate")
async def estimate_depth(
    file: UploadFile = File(...),
    colormap: str = Query("inferno", description="OpenCV colormap name"),
    raw: bool = Query(False, description="Return raw 16-bit depth instead of colorized PNG"),
):
    """
    Estimate depth from an uploaded image.
    Returns a colorized depth map PNG or raw 16-bit depth.
    """
    t0 = time.time()

    # Read uploaded image
    contents = await file.read()
    img_array = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return Response(content=b"Invalid image", status_code=400)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    # Run depth estimation
    pipe = get_depth_pipeline()
    result = pipe(pil_img)
    depth = np.array(result["depth"])

    t1 = time.time()
    logger.info(f"Depth estimation: {t1-t0:.2f}s, shape={depth.shape}")

    if raw:
        # Return raw 16-bit depth as PNG
        depth_16 = depth.astype(np.uint16)
        _, png_data = cv2.imencode(".png", depth_16)
        return Response(content=png_data.tobytes(), media_type="image/png")

    # Normalize and colorize
    d_min, d_max = depth.min(), depth.max()
    if d_max > d_min:
        depth_norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    else:
        depth_norm = np.zeros_like(depth, dtype=np.uint8)

    # Map colormap name to OpenCV constant
    cmap_map = {
        "inferno": cv2.COLORMAP_INFERNO,
        "jet": cv2.COLORMAP_JET,
        "magma": cv2.COLORMAP_MAGMA,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "turbo": cv2.COLORMAP_TURBO,
    }
    cmap = cmap_map.get(colormap, cv2.COLORMAP_INFERNO)
    depth_colored = cv2.applyColorMap(depth_norm, cmap)

    _, png_data = cv2.imencode(".png", depth_colored)
    return Response(content=png_data.tobytes(), media_type="image/png")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8013)
