"""Which crawled catalogue page is this segmented object? (Object Metadata suggestions)

POST /segment/{index}/rag-matches {model}
  crops SAM3 mask `index` out of the frame it was segmented on (background neutral grey), measures the mask's
  height/width, and asks Splat Blueprints (/rag/match): Cosmos 3 describes the crop, the ranking against the
  model's crawled pages is deterministic (object type, words, published H/W). Returns ranked candidates; the
  user confirms one in Object Metadata (nothing is linked automatically: many catalogue objects look alike).
"""
import io
import logging
from pathlib import Path

import httpx
import numpy as np
from fastapi import APIRouter, Body, HTTPException
from PIL import Image

from . import blueprints_client as bp

logger = logging.getLogger("supersplat-viewer.segment_match")
PAD_PX, MIN_PIXELS, MATCH_TIMEOUT_S = 8, 25, 120.0
GREY = np.array([128, 128, 128], dtype=np.uint8)


def crop(frame: Image.Image, mask: np.ndarray) -> tuple[bytes, float]:
    """(PNG bytes of the masked object on grey, mask height / width)."""
    while mask.ndim > 2:
        mask = mask[0]
    binary = mask > 0.5
    if binary.sum() < MIN_PIXELS:
        raise HTTPException(status_code=422, detail="mask is empty")
    rgb = np.asarray(frame.convert("RGB").resize((binary.shape[1], binary.shape[0])))
    ys, xs = np.nonzero(binary)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    aspect = float((y1 - y0) / max(1, x1 - x0))
    y0, x0 = max(0, y0 - PAD_PX), max(0, x0 - PAD_PX)
    y1, x1 = min(binary.shape[0], y1 + PAD_PX), min(binary.shape[1], x1 + PAD_PX)
    out = np.where(binary[..., None], rgb, GREY)[y0:y1, x0:x1].astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(out).save(buf, format="PNG")
    return buf.getvalue(), aspect


def create_router(cache_dir: Path, current_prompt) -> APIRouter:
    """current_prompt() -> the SAM3 text prompt of the current segmentation ('' if none)."""
    router = APIRouter(prefix="/segment", tags=["segment-match"])

    @router.post("/{index}/rag-matches")
    def rag_matches(index: int, body: dict = Body(default={})):
        model = str(body.get("model", "")).strip()
        frame_path, mask_path = cache_dir / "current_frame.png", cache_dir / f"mask_{index}.npy"
        if not model:
            raise HTTPException(status_code=422, detail="model is required")
        if not frame_path.is_file() or not mask_path.is_file():
            raise HTTPException(status_code=404, detail=f"segment {index} not found (segment the view first)")
        png, aspect = crop(Image.open(frame_path), np.load(str(mask_path)))
        try:
            with bp.client(MATCH_TIMEOUT_S) as c:
                r = c.post("/rag/match", data={"model": model, "label": current_prompt() or "", "aspect": f"{aspect:.4f}"},
                           files={"image": ("segment.png", png, "image/png")})
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"Blueprints unreachable: {e}") from None
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail=f"matching failed: HTTP {r.status_code} {r.text[:200]}")
        return {"index": index, "mask_hw": round(aspect, 3), **r.json()}

    return router
