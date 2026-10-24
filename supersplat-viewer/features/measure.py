"""Cosmos 3 equation-based measurement for the SuperSplat viewer (:8086).

"How wide is this doorway?" -> the Splat Blueprints API (splat-blueprints repo) renders the splat
with this viewer's camera, Cosmos 3 locates the feature (2D box only), and the deterministic engine
computes the dimension with a traceable equation + shown work. This router is a thin proxy:

  * POST /measure/ask        {model, question, camera} -> {job_id} at once (camera = window.getCameraMatrices())
  * GET  /measure/jobs/{id}  {state: running|completed|failed, stage, elapsed_s, result|error}
  * GET  /measure/overlay/{scene_id}/{render_id}.png?revisions=...   overlay with equation + work
  * GET  /measure/status     configured? + blueprints health

The work runs as a job (import the PLY into Blueprints once, then Blueprints' async ask-view job) because
large splats take minutes to render on CPU; a single silent browser request over Wi-Fi died as
"Failed to fetch" after ~2 minutes (measured 2026-10-18, Graz: 398 s end to end).

The blueprints bearer token is read server-side from BLUEPRINTS_TOKEN_FILE (a compose secret), so it
is never sent to the browser. The viewer's PLY is imported into Blueprints by name from the shared
PLY service (same /models directory); the scene id is cached per (model, size, mtime).
"""
import logging
import threading
import time
import uuid
from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from . import blueprints_client as bp
from . import flythrough
from .blueprints_client import BLUEPRINTS_PROJECT, BLUEPRINTS_URL

logger = logging.getLogger("supersplat-viewer.measure")

ASK_TIMEOUT_S = 960.0  # Blueprints view job budget (900 s) + margin
POLL_S = 2.0
JOB_TTL_S = 3600
IMPORT_TIMEOUT_S = 300.0  # first import downloads + parses the PLY


def create_router(model_dir: Path, rag_sources=lambda model, query=None: []) -> APIRouter:
    """rag_sources(model, question) -> [{source, subject?, text}]: the RAG material checked for real sizes."""
    router = APIRouter(prefix="/measure", tags=["measure"])
    scenes: dict[tuple, str] = {}  # (model, size, mtime_ns) -> blueprints scene_id

    _client = bp.client

    def _raise(resp: httpx.Response, what: str):
        if resp.status_code != 200:
            try:
                detail = resp.json().get("detail", resp.text)
            except ValueError:
                detail = resp.text
            raise HTTPException(status_code=resp.status_code if resp.status_code < 500 else 502, detail=f"{what}: {detail}")

    def _scene_for(model: str) -> str:
        path = model_dir / Path(model).name
        if path.suffix.lower() != ".ply" or not path.is_file():
            raise HTTPException(status_code=404, detail=f"Model {model} not found")
        st = path.stat()
        key = (path.name, st.st_size, st.st_mtime_ns)
        if key not in scenes:
            with _client(IMPORT_TIMEOUT_S) as c:
                r = c.post(f"/projects/{BLUEPRINTS_PROJECT}/scenes/import/ply-service", json={"model": path.name})
            _raise(r, "Blueprints import")
            scenes[key] = r.json()["scene_id"]
            logger.info(f"Blueprints scene for {path.name}: {scenes[key]} (existing={r.json().get('existing')})")
        return scenes[key]

    @router.get("/status")
    def status():
        try:
            tok_ok = bool(bp.token())
        except HTTPException:
            return {"configured": False, "blueprints_url": BLUEPRINTS_URL}
        try:
            ok = httpx.get(f"{BLUEPRINTS_URL}/health", timeout=5).status_code == 200
        except httpx.HTTPError:
            ok = False
        return {"configured": tok_ok, "blueprints_url": BLUEPRINTS_URL, "blueprints_healthy": ok, "project": BLUEPRINTS_PROJECT}

    jobs: dict[str, dict] = {}

    def _run_job(job: dict, model: str, question: str, camera: dict) -> None:
        try:
            job["stage"] = "importing the splat into Blueprints (first time only)"
            scene_id = _scene_for(model)
            job["stage"] = "gathering RAG references (Upload Info, object notes, crawled pages)"
            try:
                sources = rag_sources(model, question)
            except Exception as exc:  # RAG is optional for measuring
                logger.warning(f"RAG sources unavailable for {model}: {exc}")
                sources = []
            orient = flythrough.viewer_orientation(model_dir / model)  # up axis for PLYs without cameras
            job["stage"] = "starting the Cosmos 3 measurement"
            with _client(60) as c:
                r = c.post(f"/projects/{BLUEPRINTS_PROJECT}/scenes/{scene_id}/ask-view",
                           json={"question": question[:300], "viewer_camera": camera, "async": True, "rag_sources": sources,
                                 "up_ply": orient["up_ply"] if orient else None})
                _raise(r, "Cosmos 3 measurement")
                bp_job = r.json()
                deadline = time.time() + ASK_TIMEOUT_S
                while bp_job["state"] in ("queued", "running") and time.time() < deadline:
                    time.sleep(POLL_S)
                    r = c.get(f"/jobs/{bp_job['job_id']}")
                    _raise(r, "Cosmos 3 measurement")
                    bp_job = r.json()
                    job["stage"] = bp_job.get("stage") or job["stage"]
            if bp_job["state"] != "completed":
                raise HTTPException(status_code=502, detail=bp_job.get("error") or f"measurement {bp_job['state']}")
            out = bp_job["result"]
            if out.get("measured", True) and out["results"]:
                revs = ",".join(x["revision_id"] for x in out["results"])
                out["overlay_url"] = f"/measure/overlay/{scene_id}/{out['render_id']}.png?revisions={revs}"
            else:
                out["overlay_url"] = f"/measure/overlay/{scene_id}/{out['render_id']}.png"
            out["scene_id"], out["project"] = scene_id, BLUEPRINTS_PROJECT
            job.update(state="completed", result=out, stage="done")
        except HTTPException as exc:
            job.update(state="failed", error=str(exc.detail))
        except Exception as exc:  # the job boundary reports every failure to the browser
            logger.exception("measure job failed")
            job.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        finally:
            job["elapsed_s"] = round(time.time() - job["started_at"], 1)

    @router.post("/ask")
    def ask(body: dict):
        model, question, camera = body.get("model", ""), str(body.get("question", "")).strip(), body.get("camera")
        if not question:
            raise HTTPException(status_code=422, detail="Type a question, e.g. 'How wide is this doorway?'")
        if not camera or not camera.get("view") or not camera.get("proj"):
            raise HTTPException(status_code=422, detail="Viewer camera unavailable (viewer still loading?)")
        path = model_dir / Path(model).name
        if path.suffix.lower() != ".ply" or not path.is_file():
            raise HTTPException(status_code=404, detail=f"Model {model} not found")
        bp.token()  # fail fast (503) when not configured
        now = time.time()
        for k in [k for k, j in jobs.items() if now - j["started_at"] > JOB_TTL_S]:
            jobs.pop(k)
        job = {"job_id": uuid.uuid4().hex[:12], "state": "running", "stage": "queued", "started_at": now, "model": path.name,
               "question": question}
        jobs[job["job_id"]] = job
        threading.Thread(target=_run_job, args=(job, path.name, question, camera), daemon=True).start()
        return {"job_id": job["job_id"], "state": "running"}

    @router.get("/jobs/{job_id}")
    def job_status(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown or expired measurement job")
        return {**job, "elapsed_s": job.get("elapsed_s") if job["state"] != "running" else round(time.time() - job["started_at"], 1)}

    @router.get("/overlay/{scene_id}/{render_id}.png")
    def overlay(scene_id: str, render_id: str, revisions: str = Query("")):
        with _client(60) as c:  # no revisions (unmeasured answer): the plain render the answer was based on
            r = (c.get(f"/projects/{BLUEPRINTS_PROJECT}/scenes/{scene_id}/renders/{render_id}/overlay.png", params={"revisions": revisions})
                 if revisions else c.get(f"/projects/{BLUEPRINTS_PROJECT}/scenes/{scene_id}/renders/{render_id}.png"))
        _raise(r, "overlay")
        return Response(r.content, media_type="image/png")

    return router
