"""Website crawl -> RAG (user request 2026-10-23: crawl peacockroom.wayne.edu collection 3 for the splat)."""
import json

import httpx
import pytest

DOC = {"url": "https://museum.example/items/show/1", "title": "The Peacock Room | Baluster vase", "description": "A vase",
       "fields": {"Object Name": "Vase", "Dimensions": "HxWxD: 40.3 x 13.0 x 13.0 cm", "Wall": "East"},
       "statements": [{"sentence": "HxWxD: 40.3 x 13.0 x 13.0 cm", "subject": "Vase", "dims_m": {"height": 0.403}}], "images": []}


@pytest.fixture
def blueprints(monkeypatch, tmp_path):
    from features import blueprints_client as bp, rag_crawl
    tok = tmp_path / "tok"
    tok.write_text("tok-viewer\n")
    monkeypatch.setattr(bp, "BLUEPRINTS_TOKEN_FILE", str(tok))
    rag_crawl._cache.clear()
    calls = []

    def handler(req: httpx.Request):
        calls.append((req.method, req.url.path, dict(req.url.params), req.content and json.loads(req.content), req.headers.get("authorization")))
        if req.url.path == "/rag/crawl":
            return httpx.Response(200, json={"job_id": "j1", "state": "queued"})
        if req.url.path == "/jobs/j1":
            return httpx.Response(200, json={"job_id": "j1", "state": "completed", "progress": 1.0,
                                             "result": {"pages": 33, "fields": 400, "statements": 32, "images_found": 66,
                                                        "images_saved": 0, "images_robots_blocked": 33}})
        if req.url.path == "/rag/documents" and req.method == "GET":
            return httpx.Response(200, json={"total": 1, "documents": [DOC]})
        if req.url.path == "/rag/documents" and req.method == "DELETE":
            return httpx.Response(200, json={"deleted_documents": 1, "deleted_crawls": 1})
        return httpx.Response(404, json={"detail": "nope"})

    real = httpx.Client
    monkeypatch.setattr(bp.httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    return calls


def test_crawl_proxy_progress_and_forget(client, blueprints):
    r = client.post("/rag/crawl", json={"url": "https://museum.example/items/browse?collection=3", "model": "room.ply", "max_pages": 50})
    assert r.status_code == 200 and r.json()["job_id"] == "j1"
    sent = blueprints[0][3]
    assert sent["url"].startswith("https://museum.example") and sent["model"] == "room.ply" and sent["project"] == "peacock"
    assert sent["max_pages"] == 50 and sent["max_depth"] == 3 and blueprints[0][4] == "Bearer tok-viewer"
    done = client.get("/rag/crawl/j1").json()
    assert done["state"] == "completed" and done["result"]["images_robots_blocked"] == 33
    assert client.get("/rag/web-documents", params={"model": "room.ply", "q": "vase"}).json()["total"] == 1
    assert client.delete("/rag/web-documents", params={"model": "room.ply"}).json()["deleted_documents"] == 1
    assert client.post("/rag/crawl", json={"url": "", "model": "room.ply"}).status_code == 422


def test_crawled_pages_feed_ask_ai_and_measurements(blueprints):
    from features import rag_crawl
    src = rag_crawl.web_sources("room.ply", "How tall is the vase?")
    assert src[0]["subject"] == "Vase" and src[0]["url"] == DOC["url"] and "Dimensions: HxWxD: 40.3 x 13.0 x 13.0 cm" in src[0]["text"]
    assert any(c[1] == "/rag/documents" and c[2]["q"] == "How tall is the vase?" for c in blueprints)  # question-ranked
    ctx = rag_crawl.web_context("room.ply", "vase")
    assert "Wall: East" in ctx and DOC["url"] in ctx


def test_without_blueprints_there_are_no_web_sources(monkeypatch):
    from features import blueprints_client as bp, rag_crawl
    monkeypatch.setattr(bp, "BLUEPRINTS_TOKEN_FILE", "/nonexistent")
    rag_crawl._cache.clear()
    assert rag_crawl.web_sources("room.ply", "x") == [] and rag_crawl.web_context("room.ply", "x") == ""


def test_upload_info_dialog_has_the_crawl_box(client):
    html = client.get("/").text
    assert "/web/crawl.js" in html
