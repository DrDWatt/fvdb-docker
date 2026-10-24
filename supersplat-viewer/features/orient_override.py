"""Saved display orientation per PLY (user flip / 90° rotations), stored in Splat Blueprints' Mongo.

Why manual: for PLYs without trained cameras the up axis cannot be decided reliably. Measured 2026-10-23
on new_peacock.ply: the geometric cues disagreed by 82 deg (viewer default kept -> upside down), and Cosmos 3
called both the upside-down and the upright view "not upside down". Rendered from the room centre, the
default orientation showed the vases hanging downwards; quaternion [0, 0, 0, 1] (no 180 deg turn) is upright.

  GET    /orientation/{model}   {model, source: saved|automatic|trained cameras, quat, can_edit}
  POST   /orientation/{model}   {quat: [x, y, z, w]}  viewer = R @ ply; camera-less PLYs only
  DELETE /orientation/{model}   back to the automatic estimate

Key = "<file>:<size>:<mtime_ns>", so a re-exported file starts fresh. Without Blueprints the viewer keeps
working with the automatic orientation.
"""
import logging
import math
import time
from pathlib import Path

import httpx
from fastapi import APIRouter, Body, HTTPException

from . import blueprints_client as bp

logger = logging.getLogger("supersplat-viewer.orientation")
CACHE_S = 30.0
_cache: dict[str, tuple[float, list | None]] = {}


def key_for(path: Path) -> str:
    st = path.stat()
    return f"{path.name}:{st.st_size}:{st.st_mtime_ns}"


def saved_quat(path: Path) -> list | None:
    """The saved quaternion for this exact file, or None (also when Blueprints is unreachable)."""
    key = key_for(path)
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < CACHE_S:
        return hit[1]
    quat = None
    try:
        with bp.client(5) as c:
            r = c.get(f"/viewer-orientations/{key}")
        if r.status_code == 200:
            quat = r.json()["quat"]
        elif r.status_code != 404:
            logger.warning(f"Saved orientation for {path.name}: HTTP {r.status_code}")
    except (httpx.HTTPError, HTTPException) as e:  # not configured / unreachable: automatic orientation
        logger.info(f"Saved orientation unavailable for {path.name}: {e}")
        return None
    _cache[key] = (time.time(), quat)
    return quat


def matrix(q) -> list:
    x, y, z, w = q
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]


def create_router(model_dir: Path, has_cameras, describe) -> APIRouter:
    """has_cameras(path) -> bool; describe(path) -> current orientation info (flythrough.viewer_orientation)."""
    router = APIRouter(prefix="/orientation", tags=["orientation"])

    def _path(model: str) -> Path:
        path = model_dir / Path(model).name
        if path.suffix.lower() != ".ply" or not path.is_file():
            raise HTTPException(status_code=404, detail=f"Model {model} not found")
        return path

    @router.get("/{model}")
    def get_orientation(model: str):
        path = _path(model)
        cams = has_cameras(path)
        o = describe(path)
        return {"model": path.name, "quat": o["quat"] if o else None, "can_edit": not cams,
                "source": "trained cameras" if cams else (o or {}).get("source", "automatic")}

    @router.post("/{model}")
    def save_orientation(model: str, body: dict = Body(...)):
        path = _path(model)
        if has_cameras(path):
            raise HTTPException(status_code=409, detail="this PLY has trained cameras; its orientation comes from them")
        q = body.get("quat")
        if not (isinstance(q, list) and len(q) == 4 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in q)):
            raise HTTPException(status_code=422, detail="quat must be [x, y, z, w]")
        n = math.sqrt(sum(v * v for v in q))
        if n < 1e-6:
            raise HTTPException(status_code=422, detail="quat must be non-zero")
        q = [v / n for v in q]
        key = key_for(path)
        with bp.client(10) as c:
            r = c.put(f"/viewer-orientations/{key}", json={"model": path.name, "quat": q, "note": str(body.get("note", ""))[:200]})
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail=f"saving orientation failed: HTTP {r.status_code} {r.text[:200]}")
        _cache[key] = (time.time(), r.json()["quat"])
        logger.info(f"Saved orientation for {path.name}: {r.json()['quat']}")
        return get_orientation(model)

    @router.delete("/{model}")
    def reset_orientation(model: str):
        path = _path(model)
        key = key_for(path)
        with bp.client(10) as c:
            r = c.delete(f"/viewer-orientations/{key}")
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail=f"reset failed: HTTP {r.status_code}")
        _cache[key] = (time.time(), None)
        return get_orientation(model)

    return router
