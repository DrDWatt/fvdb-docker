"""Which crawled catalogue page is this segmented object? (Object Metadata suggestions)

POST /segment/{index}/rag-matches {model}
  crops SAM3 mask `index` out of the frame it was segmented on (background neutral grey), measures the mask's
  height/width, and asks Splat Blueprints (/rag/match): Cosmos 3 describes the crop, the ranking against the
  model's crawled pages is deterministic (object type, words, published H/W). Returns ranked candidates; the
  user confirms one in Object Metadata (nothing is linked automatically: many catalogue objects look alike).

POST /segment/{index}/link-scale {model, candidate}
  measures the segment in 3D (features/object_measure.py) and links it to the candidate's published height;
  Splat Blueprints recalibrates the scene from all linked objects, so measurements come out in m / cm.
"""
import hashlib
import io
import json
import logging
from pathlib import Path

import httpx
import numpy as np
from fastapi import APIRouter, Body, HTTPException
from PIL import Image

from . import blueprints_client as bp
from . import object_measure

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


def create_router(cache_dir: Path, current_prompt, model_dir: Path | None = None, scene_for=None) -> APIRouter:
    """current_prompt() -> the SAM3 text prompt of the current segmentation ('' if none);
    scene_for(model) -> Splat Blueprints scene id (features/measure.py) for "Use for scale"."""
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

    @router.post("/{index}/link-scale")
    def link_scale(index: int, body: dict = Body(...)):
        """Measure segment `index` in 3D and link it to its catalogue size; Blueprints recalibrates the scene."""
        model = Path(str(body.get("model", ""))).name
        cand = body.get("candidate") or {}
        dims = (cand.get("fields") or {}).get("Dimensions", "")
        if not model or not cand.get("url") or not dims:
            raise HTTPException(status_code=422, detail="model and a candidate with Dimensions are required")
        frame_cam, mask_path = cache_dir / "camera.json", cache_dir / f"mask_{index}.npy"
        if not mask_path.is_file() or not frame_cam.is_file():
            raise HTTPException(status_code=404, detail=f"segment {index} or its camera not found (segment the view again)")
        if model_dir is None or scene_for is None:
            raise HTTPException(status_code=503, detail="scale linking not configured")
        try:
            m = object_measure.measure(model_dir / model, np.load(str(mask_path)), json.loads(frame_cam.read_text()))
        except ValueError as e:
            raise HTTPException(status_code=422, detail=f"could not measure this object: {e}") from None
        link_id = hashlib.sha1(f"{model}|{current_prompt()}|{index}|{cand['url']}".encode()).hexdigest()[:16]
        payload = {"link_id": link_id, "name": cand.get("name", ""), "url": cand["url"], "dimensions": dims,
                   "p": m["p"], "q": m["q"], "sigma_mu": m["sigma_mu"], "n_points": m["n_points"]}
        try:
            with bp.client(120) as c:
                r = c.post(f"/projects/{bp.BLUEPRINTS_PROJECT}/scenes/{scene_for(model)}/linked-objects", json=payload)
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"Blueprints unreachable: {e}") from None
        if r.status_code != 200:
            raise HTTPException(status_code=r.status_code if r.status_code < 500 else 502, detail=r.text[:300])
        return {"measured": {k: m[k] for k in ("height_mu", "width_mu", "n_points", "sigma_mu")}, **r.json()}

    return router
