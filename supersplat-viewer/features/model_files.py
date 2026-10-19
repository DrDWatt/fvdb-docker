"""Serve model PLYs to the viewer with progress, timing and browser caching (files stay .ply).

  * GET /models/{name}           streams the PLY in 4 MB chunks, records bytes sent per download and logs
                                 the achieved throughput (MB/s) per client when it finishes. Range requests
                                 (resume) are served by FileResponse unchanged.
                                 ETag = size-mtime; If-None-Match -> 304, and Cache-Control lets the browser
                                 reuse its cached copy, so a model is downloaded once per change.
  * GET /models-progress/{name}  the latest download of that model: bytes_sent, total, MB/s, done.

Measured 2026-10-18: Frankfurt (711 MB, 10.5 M gaussians) and Graz (1.3 GB, 19.2 M) load in 10-11 s
in Chromium on the Spark itself; on a remote browser the page previously showed a black canvas for the
whole download because the loading overlay closed on iframe load and the viewer UI is hidden (noui).
"""
import logging
import re
import time
from email.utils import formatdate
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse

logger = logging.getLogger("supersplat-viewer.models")
CHUNK = 4 * 1024 * 1024
CACHE_CONTROL = "private, max-age=86400, must-revalidate"


def create_router(model_dir: Path) -> APIRouter:
    router = APIRouter(tags=["model-files"])
    progress: dict[str, dict] = {}

    def _path(name: str) -> Path:
        path = model_dir / Path(name).name
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Model file not found")
        return path

    @router.get("/models/{filename}")
    async def serve_model_file(filename: str, request: Request):
        """Serve .ply model files to the SuperSplat viewer iframe."""
        path = _path(filename)
        st = path.stat()
        etag = f'"{st.st_size:x}-{st.st_mtime_ns:x}"'
        headers = {"ETag": etag, "Cache-Control": CACHE_CONTROL, "Last-Modified": formatdate(st.st_mtime, usegmt=True),
                   "Accept-Ranges": "bytes"}
        if request.headers.get("if-none-match") == etag:
            return Response(status_code=304, headers=headers)
        start, end, status = 0, st.st_size - 1, 200
        rng = request.headers.get("range", "")
        m = re.fullmatch(r"bytes=(\d*)-(\d*)", rng.strip()) if rng else None
        if rng and not m:  # multi-range or malformed: let Starlette handle it unchanged
            return FileResponse(path, media_type="application/octet-stream", filename=filename, headers={"Cache-Control": CACHE_CONTROL})
        if m:  # Chrome resumes large downloads with Range requests (measured 2026-10-18: 206s for Graz)
            a, b = m.groups()
            start, end = (st.st_size - int(b), st.st_size - 1) if not a else (int(a), min(int(b), st.st_size - 1) if b else st.st_size - 1)
            if start > end or start >= st.st_size:
                return Response(status_code=416, headers={"Content-Range": f"bytes */{st.st_size}"})
            status = 206
        client = request.client.host if request.client else "?"
        rec = {"model": path.name, "total": st.st_size, "bytes_sent": start, "range_start": start, "started_at": time.time(),
               "done": False, "seconds": 0.0, "mb_per_s": 0.0, "client": client}
        progress[path.name] = rec

        def stream():
            try:
                with open(path, "rb") as f:
                    f.seek(start)
                    left = end - start + 1
                    while left > 0 and (chunk := f.read(min(CHUNK, left))):
                        yield chunk
                        left -= len(chunk)
                        rec["bytes_sent"] += len(chunk)
                        rec["seconds"] = time.time() - rec["started_at"]
                        rec["mb_per_s"] = (rec["bytes_sent"] - start) / 1e6 / max(rec["seconds"], 1e-6)
                rec["done"] = rec["bytes_sent"] >= end + 1
            finally:
                logger.info(f"Model download {path.name}{f' (range from {start / 1e6:.0f} MB)' if start else ''}: "
                            f"{rec['bytes_sent'] / 1e6:.0f}/{rec['total'] / 1e6:.0f} MB in {rec['seconds']:.1f}s "
                            f"({rec['mb_per_s']:.1f} MB/s) to {client}{'' if rec['done'] else ' [aborted]'}")

        out_headers = {**headers, "Content-Length": str(end - start + 1), "Content-Disposition": f'attachment; filename="{path.name}"'}
        if status == 206:
            out_headers["Content-Range"] = f"bytes {start}-{end}/{st.st_size}"
        return StreamingResponse(stream(), status_code=status, media_type="application/octet-stream", headers=out_headers)

    @router.get("/models-progress/{filename}")
    async def model_progress(filename: str):
        name = Path(filename).name
        rec = progress.get(name)
        if rec is None:
            return {"model": name, "started": False}
        return {**rec, "started": True, "server_now": time.time(), "percent": round(100 * rec["bytes_sent"] / max(rec["total"], 1), 1)}

    return router
