"""Flythrough + MP4 export for the SuperSplat viewer (:8086).

Uses the SuperSplat viewer's own camera-animation API instead of server renders:
  * GET  /flythrough/settings/{model}   - viewer settings.json (v2) whose animTrack
    follows the model's smoothed trained-camera path (or an orbit when the PLY has
    no cameras). The viewer plays it natively on the GPU at display refresh rate.
  * The browser exports by calling window.captureFrame({time}) per frame
    (deterministic, supersampled, offscreen) and streams JPEGs to
    /flythrough/export/{session}/frames; the server encodes H.264 MP4.
"""
import logging
import math
import re
import time
import uuid
from pathlib import Path

import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask

from viewer_common.camera_path import (
    MIN_VIDEO_SECONDS, build_smooth_path, ease_in_out, frames_for_duration, sample_path,
)
from viewer_common.video import H264Writer

logger = logging.getLogger("supersplat-viewer.flythrough")

KEY_INTERVAL_S = 0.1        # animTrack keyframe spacing (viewer splines between keys)
TRACK_FRAME_RATE = 30       # animTrack time unit: keyframe times are in frames
SESSION_TTL_S = 15 * 60     # abandoned export sessions are reaped after this
MAX_DURATION_S = 120

_PLY_TYPES = {
    'float': 'f4', 'float32': 'f4', 'double': 'f8', 'float64': 'f8',
    'uchar': 'u1', 'uint8': 'u1', 'char': 'i1', 'int8': 'i1',
    'short': 'i2', 'int16': 'i2', 'ushort': 'u2', 'uint16': 'u2',
    'int': 'i4', 'int32': 'i4', 'uint': 'u4', 'uint32': 'u4',
}
_TENSOR_COMMENT = re.compile(r'^comment fvdb_ply_af_\d+([A-Za-z_]\w*)\|tensor\|([\d,]+)$')


def _read_ply_layout(path: Path):
    """Return (header_end, elements, tensor_shapes) for a binary little-endian PLY."""
    elements, shapes = [], {}
    with open(path, 'rb') as f:
        while True:
            line = f.readline().decode('ascii', errors='ignore').strip()
            if not line and f.tell() > 1_000_000:
                raise ValueError("PLY header not terminated")
            if line.startswith('element'):
                _, name, count = line.split()
                elements.append({"name": name, "count": int(count), "props": []})
            elif line.startswith('property') and elements:
                parts = line.split()
                if parts[1] == 'list':
                    raise ValueError("list properties are not supported")
                elements[-1]["props"].append((parts[2], _PLY_TYPES[parts[1]]))
            elif line.startswith('comment'):
                m = _TENSOR_COMMENT.match(line)
                if m:
                    dims = [int(x) for x in m.group(2).split(',')]
                    shapes[m.group(1)] = tuple(dims[1:])  # first value is ndim
            elif line == 'end_header':
                return f.tell(), elements, shapes


def _read_elements(path: Path):
    """Memory-map every element of the PLY as a structured numpy array."""
    offset, elements, shapes = _read_ply_layout(path)
    arrays = {}
    for el in elements:
        dtype = np.dtype([(name, '<' + t) for name, t in el["props"]])
        arrays[el["name"]] = np.memmap(path, dtype=dtype, mode='r', offset=offset, shape=(el["count"],))
        offset += dtype.itemsize * el["count"]
    return arrays, shapes


def read_trained_cameras(path: Path):
    """Trained cameras stored by fVDB as extra PLY elements, or None if absent."""
    arrays, shapes = _read_elements(path)

    def tensor(name):
        if name not in arrays:
            return None
        values = np.asarray(arrays[name]['value'], dtype=np.float64)
        return values.reshape(shapes[name]) if name in shapes else values

    c2w, K, sizes = (tensor(n) for n in ('camera_to_world_matrices', 'projection_matrices', 'image_sizes'))
    if c2w is None or K is None or sizes is None or c2w.ndim != 3 or len(c2w) < 2:
        return None
    return {"c2w": c2w, "K": K, "sizes": sizes, "depths": tensor('median_depths')}


def scene_bounds(path: Path, samples: int = 200_000):
    """Robust (center, half-extent) of the gaussian means from a vertex subsample."""
    arrays, _ = _read_elements(path)
    v = arrays['vertex']
    step = max(1, len(v) // samples)
    xyz = np.stack([np.asarray(v[k][::step], dtype=np.float64) for k in ('x', 'y', 'z')], axis=1)
    xyz = xyz[np.isfinite(xyz).all(axis=1)]
    lo, hi = np.percentile(xyz, 5, axis=0), np.percentile(xyz, 95, axis=0)
    return (lo + hi) / 2, (hi - lo) / 2


def _to_viewer(p):
    """The viewer loads splats with Euler (0, 0, 180): (x, y, z) -> (-x, -y, z)."""
    return np.array([-p[0], -p[1], p[2]])


def _path_track(cams, duration):
    """Forward-then-back loop along the trained path: seamless when repeated, and
    the first `duration` seconds are the one-way flythrough that gets exported."""
    path = build_smooth_path(cams["c2w"], cams["K"], cams["sizes"], cams["depths"])
    fallback_depth = float(np.median(np.linalg.norm(path["pos"] - path["pos"].mean(axis=0), axis=1))) or 1.0
    keys = int(round(2 * duration / KEY_INTERVAL_S))
    positions, targets, fovs, times = [], [], [], []
    for k in range(keys):
        t = k * KEY_INTERVAL_S
        u = ease_in_out(t / duration if t <= duration else 2 - t / duration)
        pos, rot, K_norm, depth = sample_path(path, u)
        target = pos + rot[:, 2] * (depth or fallback_depth)  # OpenCV camera looks down +z
        positions += list(_to_viewer(pos))
        targets += list(_to_viewer(target))
        fovs.append(math.degrees(2 * math.atan(0.5 / K_norm[0, 0])))  # horizontal FOV
        times.append(t * TRACK_FRAME_RATE)
    return positions, targets, fovs, times, 2 * duration, path["num_cameras"]


def _orbit_track(model_path, duration, fov=60.0):
    """One full orbit per `duration` for models without trained cameras."""
    center, half = scene_bounds(model_path)
    c = _to_viewer(center)
    radius = float(np.linalg.norm(half)) * 1.6 or 1.0
    elevation = math.radians(20)
    keys = int(round(duration / KEY_INTERVAL_S))
    positions, targets, fovs, times = [], [], [], []
    for k in range(keys):
        a = 2 * math.pi * k / keys
        positions += [c[0] + radius * math.cos(elevation) * math.sin(a),
                      c[1] + radius * math.sin(elevation),
                      c[2] + radius * math.cos(elevation) * math.cos(a)]
        targets += list(c)
        fovs.append(fov)
        times.append(k * KEY_INTERVAL_S * TRACK_FRAME_RATE)
    return positions, targets, fovs, times, duration, 0


def build_settings(model_path: Path, duration: float):
    """SuperSplat v2 settings whose first animTrack is the flythrough."""
    track = None
    try:
        cams = read_trained_cameras(model_path)
        if cams is not None:
            track, mode = _path_track(cams, duration), "camera_path"
    except Exception as e:
        logger.warning(f"Trained-camera path unavailable for {model_path.name}: {e}")
    if track is None:
        try:
            track, mode = _orbit_track(model_path, duration), "orbit"
        except Exception as e:
            logger.warning(f"Orbit path unavailable for {model_path.name}: {e}")
    settings = {
        "version": 2,
        "tonemapping": "none",
        "highPrecisionRendering": False,
        "background": {"color": [0.04, 0.04, 0.06]},
        "postEffectSettings": {
            "sharpness": {"enabled": False, "amount": 0},
            "bloom": {"enabled": False, "intensity": 1, "blurLevel": 2},
            "grading": {"enabled": False, "brightness": 0, "contrast": 1, "saturation": 1, "tint": [1, 1, 1]},
            "vignette": {"enabled": False, "intensity": 0.5, "inner": 0.3, "outer": 0.75, "curvature": 1},
            "fringing": {"enabled": False, "intensity": 0.5},
        },
        "animTracks": [],
        "cameras": [],
        "annotations": [],
        "startMode": "default",
    }
    if track is None:
        # The viewer generates its own orbit / figure-8 track when none is given
        return settings, {"mode": "viewer_default", "num_cameras": 0, "duration": duration, "loop_duration": None}
    positions, targets, fovs, times, track_len, n_cams = track
    settings["animTracks"] = [{
        "name": "flythrough",
        "duration": track_len,
        "frameRate": TRACK_FRAME_RATE,
        "loopMode": "repeat",
        "interpolation": "spline",
        "smoothness": 1,
        "keyframes": {"times": times, "values": {"position": positions, "target": targets, "fov": fovs}},
    }]
    settings["cameras"] = [{"initial": {"position": positions[:3], "target": targets[:3], "fov": fovs[0]}}]
    settings["startMode"] = "animTrack"
    info = {"mode": mode, "num_cameras": n_cams, "duration": duration, "loop_duration": track_len}
    return settings, info


def create_router(model_dir: Path, output_dir: Path) -> APIRouter:
    router = APIRouter(prefix="/flythrough", tags=["flythrough"])
    settings_cache = {}
    sessions = {}

    def _model_path(model: str) -> Path:
        path = (model_dir / Path(model).name)
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Model {model} not found")
        return path

    def _settings(model: str, duration: float):
        path = _model_path(model)
        key = (path.name, path.stat().st_mtime, duration)
        if key not in settings_cache:
            started = time.time()
            settings_cache.clear()  # keep only the current model in memory
            settings_cache[key] = build_settings(path, duration)
            info = settings_cache[key][1]
            logger.info(f"Flythrough track for {path.name}: {info['mode']}, {info['num_cameras']} cameras, "
                        f"{duration}s ({time.time() - started:.2f}s to build)")
        return settings_cache[key]

    def _duration(value: float) -> float:
        return float(min(max(value, MIN_VIDEO_SECONDS), MAX_DURATION_S))

    @router.get("/settings/{model}")
    async def flythrough_settings(model: str, duration: float = Query(MIN_VIDEO_SECONDS)):
        """Viewer settings.json with the flythrough animTrack (load via ?settings=)."""
        return _settings(model, _duration(duration))[0]

    @router.get("/info/{model}")
    async def flythrough_info(model: str, duration: float = Query(MIN_VIDEO_SECONDS)):
        """Path mode, camera count and durations for the UI."""
        return {**_settings(model, _duration(duration))[1], "min_duration": MIN_VIDEO_SECONDS,
                "max_duration": MAX_DURATION_S}

    def _reap_sessions():
        now = time.time()
        for sid in [s for s, v in sessions.items() if now - v["updated"] > SESSION_TTL_S]:
            sessions.pop(sid)["writer"].abort()
            logger.info(f"Reaped abandoned export session {sid}")

    @router.post("/export/start")
    async def export_start(body: dict):
        """Open an H.264 encode session; the browser then streams captured frames."""
        _reap_sessions()
        model_path = _model_path(body.get("model", ""))
        fps = int(min(max(int(body.get("fps", 30)), 1), 60))
        duration = _duration(float(body.get("duration", MIN_VIDEO_SECONDS)))
        width, height = int(body.get("width", 1280)), int(body.get("height", 720))
        if not (100 <= width <= 3840 and 100 <= height <= 2160):
            raise HTTPException(status_code=400, detail="width/height out of range")
        sid = uuid.uuid4().hex[:12]
        out = output_dir / f"{model_path.stem}_flythrough_{sid}.mp4"
        writer = H264Writer(out, width, height, fps, input_format="jpeg")
        sessions[sid] = {"writer": writer, "path": out, "model": model_path.stem,
                         "expected": frames_for_duration(duration, fps), "updated": time.time()}
        return {"session_id": sid, "num_frames": sessions[sid]["expected"], "fps": fps,
                "duration": duration, "width": writer.width, "height": writer.height}

    def _session(sid):
        if sid not in sessions:
            raise HTTPException(status_code=404, detail="Unknown or expired export session")
        return sessions[sid]

    @router.post("/export/{sid}/frames")
    async def export_frames(sid: str, frames: list[UploadFile] = File(...)):
        """Append JPEG frames (in order) to the encode session."""
        session = _session(sid)
        for frame in frames:
            session["writer"].write(await frame.read())
        session["updated"] = time.time()
        return {"received": session["writer"].frames, "expected": session["expected"]}

    @router.post("/export/{sid}/finish")
    async def export_finish(sid: str):
        """Finalize encoding and return the MP4."""
        session = sessions.pop(sid, None)
        if session is None:
            raise HTTPException(status_code=404, detail="Unknown or expired export session")
        writer = session["writer"]
        if writer.frames < session["expected"]:
            writer.abort()
            return JSONResponse(status_code=400, content={
                "message": f"Incomplete export: {writer.frames}/{session['expected']} frames"})
        try:
            writer.close()
        except RuntimeError as e:
            session["path"].unlink(missing_ok=True)
            return JSONResponse(status_code=500, content={"message": str(e)})
        # The browser downloads the file; remove the server copy afterwards so
        # exports don't accumulate in the outputs volume.
        return FileResponse(session["path"], media_type="video/mp4",
                            filename=f"{session['model']}_flythrough.mp4",
                            background=BackgroundTask(session["path"].unlink, missing_ok=True))

    @router.delete("/export/{sid}")
    async def export_abort(sid: str):
        session = sessions.pop(sid, None)
        if session:
            session["writer"].abort()
            session["path"].unlink(missing_ok=True)
        return {"status": "ok"}

    return router
