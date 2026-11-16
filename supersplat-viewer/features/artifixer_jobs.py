"""ArtiFixer3D+ jobs from the :8086 viewer (Nov 16 2026): run ArtiFixer3D+ (or the quick floater cleanup) on the loaded
PLY and follow the job, without leaving the viewer. The ArtiFixer controller (~/artifixer-service, :8092) does the work;
this router is a thin server-side proxy to its REST API (same validation as its Swagger page /docs and its /mcp tools).
"""
import os

import httpx
from fastapi import APIRouter, HTTPException

ARTIFIXER_URL = os.environ.get("ARTIFIXER_CONTROLLER_URL", "http://100.105.59.33:8092").rstrip("/")
PUBLIC_URL = os.environ.get("ARTIFIXER_CONTROLLER_PUBLIC_URL", "http://100.105.59.33:8092").rstrip("/")


def create_router() -> APIRouter:
    router = APIRouter(prefix="/artifixer", tags=["artifixer"])

    def _call(method: str, path: str, **kw):
        try:
            r = httpx.request(method, f"{ARTIFIXER_URL}{path}", timeout=60, **kw)
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"ArtiFixer controller unreachable ({ARTIFIXER_URL}): {exc}") from exc
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except ValueError:
                detail = r.text
            raise HTTPException(status_code=r.status_code if r.status_code < 500 else 502, detail=str(detail)[:500])
        return r.json()

    @router.get("/status")
    def status(model: str = ""):
        """Brev login, endpoints, recent jobs and whether `model` can run ArtiFixer3D+ (needs its COLMAP capture)."""
        plys = {p["name"]: p for p in _call("GET", "/api/plys")}
        return {"brev": _call("GET", "/api/brev"), "endpoints": _call("GET", "/api/endpoints"),
                "jobs": _call("GET", "/api/jobs")[:8], "model": plys.get(model),
                "links": {"controller": PUBLIC_URL, "swagger": f"{PUBLIC_URL}/docs", "mcp": f"{PUBLIC_URL}/mcp"}}

    @router.post("/jobs")
    def submit(body: dict):
        """{kind: artifixer3d_plus|cleanup, ply, endpoint_id?, camera_id?} -> the queued job."""
        allowed = {k: body[k] for k in ("kind", "ply", "endpoint_id", "camera_id", "width", "frames", "variant") if k in body}
        return _call("POST", "/api/jobs", json=allowed)

    @router.get("/jobs/{job_id}/log")
    def log(job_id: str, tail: int = 60):
        return _call("GET", f"/api/jobs/{job_id}/log", params={"tail": max(1, min(tail, 500))})

    @router.post("/jobs/{job_id}/cancel")
    def cancel(job_id: str):
        return _call("POST", f"/api/jobs/{job_id}/cancel")

    return router
