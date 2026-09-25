"""Metadata linking + RAG query for the SuperSplat viewer (:8086).

Mirrors the fVDB viewer (:8085):
  * Model-level documents ("Upload Info") are stored by the rendering service, so
    they are shared by both viewers.
  * Label / notes / documents can be linked to SAM3 objects and GARField
    extractions (kept in memory, like :8085).
  * "Ask AI" streams answers from Ollama grounded on the aggregated scene context.
"""
import logging
import os
import time
from typing import Callable, List

import httpx
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse

from viewer_common.rag import build_system_prompt, extract_file_text, ollama_status, ollama_stream_sse

logger = logging.getLogger("supersplat-viewer.rag")

MODEL_SERVICE_URL = os.environ.get("MODEL_SERVICE_URL", "http://rendering-service:8001")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://ollama:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "nemotron-mini")
KINDS = ("object", "extraction")


def create_router(get_scene_state: Callable[[], dict]) -> APIRouter:
    """get_scene_state() -> {"model", "num_gaussians", "detections": {"prompt", "masks"},
    "extractions": [{"job_id", "model", "status", "num_gaussians"}]}"""
    router = APIRouter(tags=["metadata-rag"])
    # {(kind, item_id): {"model", "label", "text", "files": [...], "training_data": {...}}}
    items = {}

    def _item(kind: str, item_id: str, model: str = "") -> dict:
        if kind not in KINDS:
            raise HTTPException(status_code=400, detail=f"kind must be one of {KINDS}")
        entry = items.setdefault((kind, item_id), {"model": model, "label": "", "text": "",
                                                   "files": [], "training_data": {}})
        if model:
            entry["model"] = model
        return entry

    def _public(entry: dict) -> dict:
        return {
            "model": entry["model"], "label": entry["label"], "text": entry["text"],
            "training_data": entry["training_data"],
            "files": [{"idx": i, "name": f["name"], "size": f["size"], "content_type": f["content_type"]}
                      for i, f in enumerate(entry["files"])],
        }

    # ----- Model-level documents (shared with :8085 via the rendering service) -----
    def _model_summary(model: str, timeout: float = 5):
        try:
            resp = httpx.get(f"{MODEL_SERVICE_URL}/summary/{model}", timeout=timeout)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.warning(f"Model summary unavailable for {model}: {e}")
        return {"model": model, "has_summary": False, "summary": None}

    @router.get("/model_summary")
    def get_model_summary(model: str = Query(...)):
        return _model_summary(model)

    @router.post("/upload_summary")
    async def upload_model_summary(model: str = Query(...), file: UploadFile = File(...)):
        content = await file.read()
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(f"{MODEL_SERVICE_URL}/summary/{model}",
                                         files={"file": (file.filename, content, file.content_type)})
            if resp.status_code == 200:
                return resp.json()
            return JSONResponse(status_code=resp.status_code,
                                content={"error": f"Upload failed: {resp.status_code}"})
        except Exception as e:
            return JSONResponse(status_code=502, content={"error": str(e)})

    # ----- Per-object / per-extraction metadata links -----
    @router.get("/metadata/{kind}/{item_id}")
    async def get_metadata(kind: str, item_id: str):
        entry = items.get((kind, item_id))
        return {"kind": kind, "id": item_id, "summary": _public(entry) if entry else None}

    @router.post("/metadata/{kind}/{item_id}")
    async def save_metadata(kind: str, item_id: str, label: str = Form(""), text: str = Form(""),
                            model: str = Form("")):
        entry = _item(kind, item_id, model)
        entry["label"], entry["text"] = label, text
        logger.info(f"Saved {kind} metadata {item_id}: label={label!r}")
        return {"status": "ok", "kind": kind, "id": item_id}

    @router.post("/metadata/{kind}/{item_id}/upload")
    async def upload_metadata_files(kind: str, item_id: str, files: List[UploadFile] = File(...),
                                    model: str = Form("")):
        entry = _item(kind, item_id, model)
        uploaded = []
        for f in files:
            data = await f.read()
            entry["files"].append({"name": f.filename, "content_type": f.content_type or "",
                                   "size": f"{len(data) / 1024:.1f} KB", "data": data})
            uploaded.append({"name": f.filename, "size": entry["files"][-1]["size"]})
        entry["training_data"] = {"files_count": len(entry["files"]),
                                  "last_upload": time.strftime("%Y-%m-%d %H:%M:%S"),
                                  "status": "documents_uploaded"}
        return {"status": "ok", "uploaded": uploaded, "summary": _public(entry)}

    @router.get("/metadata/{kind}/{item_id}/file/{file_idx}")
    async def get_metadata_file(kind: str, item_id: str, file_idx: int):
        entry = items.get((kind, item_id))
        if not entry or not 0 <= file_idx < len(entry["files"]):
            return Response(content="File not found", status_code=404)
        f = entry["files"][file_idx]
        return Response(content=f["data"], media_type=f["content_type"] or "application/octet-stream",
                        headers={"Content-Disposition": f'inline; filename="{f["name"]}"'})

    @router.get("/metadata")
    async def list_metadata(model: str = Query(None)):
        return {"items": [{"kind": k, "id": i, **_public(e)} for (k, i), e in items.items()
                          if not model or e["model"] in ("", model)]}

    # ----- RAG -----
    def _build_context(model: str) -> str:
        scene = get_scene_state()
        model = model or scene.get("model") or ""
        parts = []
        if model:
            parts.append(f"Model: {model}, {scene.get('num_gaussians', 'unknown')} gaussians")
        det = scene.get("detections") or {}
        if det.get("masks"):
            scores = ", ".join(f"{m['score']:.2f}" for m in det["masks"])
            parts.append(f"SAM3 segmentation for '{det.get('prompt', '')}': "
                         f"{len(det['masks'])} object(s) detected (scores: {scores})")
        for ext in scene.get("extractions", []):
            if ext.get("model") and model and ext["model"] != model:
                continue
            parts.append(f"GARField 3D extraction {ext['job_id']}: {ext.get('num_gaussians', 0)} gaussians "
                         f"({ext.get('status', 'unknown')})")
        for (kind, item_id), e in items.items():
            if e["model"] and model and e["model"] != model:
                continue
            name = "SAM3 object" if kind == "object" else "3D extraction"
            entry = f"{name} '{e['label'] or item_id}'"
            if e["text"]:
                entry += f": {e['text']}"
            for f in e["files"]:
                doc = extract_file_text(f)
                if doc:
                    entry += f"\n  Document '{f['name']}': {doc}"
            parts.append(entry)
        if model:
            summary = _model_summary(model, timeout=3)
            if summary.get("has_summary") and summary.get("summary"):
                parts.append(f"Model document summary: {summary['summary'][:3000]}")
        return "\n".join(parts)

    @router.get("/rag/context")
    def rag_context(model: str = Query(None)):
        scene = get_scene_state()
        model = model or scene.get("model")
        det = scene.get("detections") or {}
        summary = _model_summary(model, timeout=3) if model else {}
        linked = [e for e in items.values() if not e["model"] or e["model"] == model]
        return {
            "model_name": model,
            "model_summary": (summary.get("summary") or "")[:500] if summary.get("has_summary") else None,
            "segments_count": len(det.get("masks", [])),
            "segment_labels": [e["label"] for (k, _), e in items.items()
                               if k == "object" and e["label"] and e in linked],
            "extractions_count": len([x for x in scene.get("extractions", []) if x.get("model") in (None, model)]),
            "documents_count": sum(len(e["files"]) for e in linked),
            "context_length": len(_build_context(model)),
        }

    @router.get("/rag/status")
    def rag_status():
        return ollama_status(OLLAMA_URL, OLLAMA_MODEL)

    @router.post("/rag/query")
    def rag_query(body: dict):
        query = (body.get("query") or "").strip()
        if not query:
            return JSONResponse({"error": "Empty query"}, status_code=400)
        system_msg = build_system_prompt(
            _build_context(body.get("model")),
            sources="SAM3 segmentation results, GARField 3D extractions",
            empty_hint="No context data available yet. Try: (1) Segment objects with SAM3, "
                       "(2) Extract 3D objects with GARField, (3) Upload documents via Upload Info.",
        )
        messages = [{"role": "system", "content": system_msg}]
        messages += [{"role": m["role"], "content": m["content"]} for m in body.get("history", [])[-20:]]
        messages.append({"role": "user", "content": query})
        return StreamingResponse(ollama_stream_sse(OLLAMA_URL, OLLAMA_MODEL, messages),
                                 media_type="text/event-stream")

    return router
