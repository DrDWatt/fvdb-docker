"""Crawl a website about the current model into the RAG store (Splat Blueprints /rag/*), and read it back.

  POST   /rag/crawl                {model, url, max_depth?, max_pages?, scope?}  -> {job_id}
  GET    /rag/crawl/{job_id}       crawl progress / result (pages, fields, size statements, images)
  GET    /rag/web-documents?model=&q=&limit=   crawled documents, best matches for q first
  DELETE /rag/web-documents?model=  forget the crawled documents of a model

web_sources(model, query) turns the best-matching crawled pages into RAG sources for Ask AI and for Cosmos 3
measurements (labelled fields such as Object Name / Dimensions / Wall / Shelf Number, the description and
the page's size statements, each cited by its URL).
"""
import logging
import time

import httpx
from fastapi import APIRouter, Body, HTTPException, Query

from . import blueprints_client as bp

logger = logging.getLogger("supersplat-viewer.rag_crawl")
CACHE_S = 30.0
_cache: dict[tuple, tuple[float, dict]] = {}


def _get_documents(model: str, query: str = "", limit: int = 8) -> dict:
    key = (model, query, limit)
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < CACHE_S:
        return hit[1]
    try:
        with bp.client(10) as c:
            r = c.get("/rag/documents", params={"model": model, "q": query, "limit": limit})
        data = r.json() if r.status_code == 200 else {"total": 0, "documents": []}
    except (httpx.HTTPError, HTTPException) as e:  # Blueprints down / not configured: no web documents
        logger.info(f"Crawled documents unavailable for {model}: {e}")
        data = {"total": 0, "documents": []}
    _cache[key] = (time.time(), data)
    return data


def web_sources(model: str, query: str | None = None, limit: int = 8) -> list[dict]:
    """Crawled pages as RAG sources: {source, subject, text, url}."""
    if not model:
        return []
    out = []
    for d in _get_documents(model, query or "", limit)["documents"]:
        fields = d.get("fields") or {}
        lines = [f"Title: {d.get('title', '')}"] + [f"{k}: {v}" for k, v in fields.items()]
        if d.get("description"):
            lines.append(d["description"])
        lines += [s["sentence"] for s in d.get("statements") or []]
        out.append({"source": f"web: {d.get('title', '')[:80]} ({d['url']})", "url": d["url"],
                    "subject": fields.get("Object Name") or fields.get("Title") or None, "text": "\n".join(lines)[:6000]})
    return out


def web_context(model: str, query: str | None, max_chars: int = 6000) -> str:
    """Compact text block of the best-matching crawled pages for the Ask AI prompt."""
    parts, used = [], 0
    for s in web_sources(model, query, limit=12):
        block = f"[{s['source']}]\n{s['text']}"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)


def create_router() -> APIRouter:
    router = APIRouter(prefix="/rag", tags=["rag-crawl"])

    def _call(method: str, path: str, **kw):
        try:
            with bp.client(30) as c:
                r = c.request(method, path, **kw)
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"Blueprints unreachable: {e}") from None
        if r.status_code != 200:
            try:
                detail = r.json().get("detail", r.text)
            except ValueError:
                detail = r.text
            raise HTTPException(status_code=r.status_code if r.status_code < 500 else 502, detail=str(detail)[:300])
        return r.json()

    @router.post("/crawl")
    def start(body: dict = Body(...)):
        url, model = str(body.get("url", "")).strip(), str(body.get("model", "")).strip()
        if not url or not model:
            raise HTTPException(status_code=422, detail="url and model are required")
        _cache.clear()
        return _call("POST", "/rag/crawl", json={"url": url, "model": model, "project": bp.BLUEPRINTS_PROJECT,
                                                  "max_depth": int(body.get("max_depth", 3)), "max_pages": int(body.get("max_pages", 200)),
                                                  "scope": body.get("scope", "path"), "download_images": bool(body.get("download_images", True))})

    @router.get("/crawl/{job_id}")
    def status(job_id: str):
        job = _call("GET", f"/jobs/{job_id}")
        if job.get("state") == "completed":
            _cache.clear()
        return {k: job.get(k) for k in ("job_id", "state", "stage", "progress", "result", "error")}

    @router.get("/web-documents")
    def documents(model: str = Query(...), q: str = Query(""), limit: int = Query(20, ge=1, le=200)):
        return _call("GET", "/rag/documents", params={"model": model, "q": q, "limit": limit})

    @router.delete("/web-documents")
    def forget(model: str = Query(...)):
        _cache.clear()
        return _call("DELETE", "/rag/documents", params={"model": model})

    return router
