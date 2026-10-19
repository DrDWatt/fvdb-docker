"""/measure router: proxies a question + viewer camera to Splat Blueprints with a server-side token.

Blueprints is replaced by an in-memory test double (httpx.MockTransport) that records every call.
"""
import json

import httpx
import pytest

CAMERA = {"view": [1.0] * 16, "proj": [0.5] * 16, "model": [1.0] * 16, "width": 1280, "height": 720}


class FakeBlueprints:
    def __init__(self):
        self.calls = []
        self.polls = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b"{}") if request.method == "POST" else None
        self.calls.append((request.method, request.url.path, body, request.headers.get("authorization")))
        p = request.url.path
        if p.endswith("/scenes/import/ply-service"):
            return httpx.Response(200, json={"scene_id": "splat-abc", "revision": 1, "existing": False})
        if p.endswith("/ask-view"):
            assert body["async"] is True
            if "window" in body["question"]:
                return httpx.Response(200, json={"job_id": "bp-fail", "state": "queued"})
            if "stairs" in body["question"]:
                return httpx.Response(200, json={"job_id": "bp-visual", "state": "queued"})
            return httpx.Response(200, json={"job_id": "bp-1", "state": "queued"})
        if p == "/jobs/bp-1":
            self.polls += 1
            if self.polls < 2:
                return httpx.Response(200, json={"job_id": "bp-1", "state": "running", "stage": "rendering the view"})
            return httpx.Response(200, json={"job_id": "bp-1", "state": "completed", "result": {
                "run_id": "run-1234567890", "render_id": "r1", "measured": True, "answer": ["Doorway width: 1.47 mu"], "unscaled": True,
                "results": [{"revision_id": "rev-a", "label": "Doorway width (top)", "display": "1.42 mu", "work": []},
                            {"revision_id": "rev-b", "label": "Doorway width (bottom)", "display": "1.53 mu", "work": []}],
                "edges": {}, "grounding": {"label": "doorway"}, "plane": {"rms": 0.05, "support": 1, "samples": 1}, "seconds": 9.1}})
        if p == "/jobs/bp-visual":
            return httpx.Response(200, json={"job_id": "bp-visual", "state": "completed", "result": {
                "run_id": "run-visual-12", "render_id": "r2", "measured": False, "results": [], "unscaled": True, "seconds": 3.0,
                "answer": ["Not measured with equations: grounding failed", "Cosmos 3 visual answer (not a measurement): about two storeys"]}})
        if p == "/jobs/bp-fail":
            return httpx.Response(200, json={"job_id": "bp-fail", "state": "failed", "error": "HTTPException: 502: controller unavailable"})
        if p.endswith("/overlay.png"):
            return httpx.Response(200, content=b"\x89PNG-overlay")
        if p.endswith("/renders/r2.png"):
            return httpx.Response(200, content=b"\x89PNG-render")
        if p == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(404, json={"detail": "not found"})


@pytest.fixture
def bp(ss, monkeypatch, tmp_path):
    from features import measure
    fake = FakeBlueprints()
    tok = tmp_path / "token"
    tok.write_text("tok-viewer\n")
    monkeypatch.setattr(measure, "BLUEPRINTS_TOKEN_FILE", str(tok))
    real_client = httpx.Client
    monkeypatch.setattr(measure.httpx, "Client", lambda **kw: real_client(transport=httpx.MockTransport(fake), **kw))
    monkeypatch.setattr(measure.httpx, "get", lambda url, **kw: fake(httpx.Request("GET", url)))
    monkeypatch.setattr(measure, "POLL_S", 0.01)
    return fake


def _wait(client, job_id):
    import time
    for _ in range(500):
        job = client.get(f"/measure/jobs/{job_id}").json()
        if job["state"] != "running":
            return job
        time.sleep(0.01)
    raise AssertionError("measurement job did not finish")


def test_ask_runs_as_a_job_imports_once_and_forwards_camera_with_server_token(client, scene, bp):
    for _ in range(2):
        r = client.post("/measure/ask", json={"model": scene, "question": "How wide is this doorway?", "camera": CAMERA})
        assert r.status_code == 200 and r.json()["state"] == "running", r.text
        job = _wait(client, r.json()["job_id"])
        bp.polls = 0
    assert job["state"] == "completed", job
    out = job["result"]
    assert out["overlay_url"] == "/measure/overlay/splat-abc/r1.png?revisions=rev-a,rev-b" and out["scene_id"] == "splat-abc"
    imports = [c for c in bp.calls if c[1].endswith("import/ply-service")]
    asks = [c for c in bp.calls if c[1].endswith("/ask-view")]
    assert len(imports) == 1 and imports[0][2] == {"model": scene}  # scene id cached per (model, size, mtime)
    assert asks[0][1] == "/projects/peacock/scenes/splat-abc/ask-view"
    assert {k: asks[0][2][k] for k in ("question", "viewer_camera", "async")} == {
        "question": "How wide is this doorway?", "viewer_camera": CAMERA, "async": True}
    assert asks[0][2]["rag_sources"] == [] and asks[0][2]["up_ply"] is None  # no notes yet; scene has no orientation fix
    assert all(c[3] == "Bearer tok-viewer" for c in bp.calls if c[1].startswith(("/projects/", "/jobs/")))  # Blueprints calls only
    assert "tok-viewer" not in str(job)  # the token never reaches the browser


def test_visual_answer_failures_overlay_and_validation(client, scene, bp):
    vis = _wait(client, client.post("/measure/ask", json={"model": scene, "question": "How tall are the stairs?", "camera": CAMERA}).json()["job_id"])
    assert vis["state"] == "completed" and vis["result"]["measured"] is False
    assert vis["result"]["answer"][1].startswith("Cosmos 3 visual answer") and vis["result"]["overlay_url"] == "/measure/overlay/splat-abc/r2.png"
    assert client.get("/measure/overlay/splat-abc/r2.png").content == b"\x89PNG-render"  # plain render when nothing was measured
    png = client.get("/measure/overlay/splat-abc/r1.png?revisions=rev-a")
    assert png.status_code == 200 and png.content == b"\x89PNG-overlay"
    assert client.get("/measure/status").json()["blueprints_healthy"] is True
    bad = _wait(client, client.post("/measure/ask", json={"model": scene, "question": "How tall is this window?", "camera": CAMERA}).json()["job_id"])
    assert bad["state"] == "failed" and "controller unavailable" in bad["error"]
    assert client.post("/measure/ask", json={"model": scene, "question": "", "camera": CAMERA}).status_code == 422
    assert client.post("/measure/ask", json={"model": scene, "question": "How wide?", "camera": None}).status_code == 422
    assert client.post("/measure/ask", json={"model": "missing.ply", "question": "How wide?", "camera": CAMERA}).status_code == 404
    assert client.get("/measure/jobs/nope").status_code == 404


def test_not_configured_without_token(client, scene, monkeypatch):
    from features import measure
    monkeypatch.setattr(measure, "BLUEPRINTS_TOKEN_FILE", "/nonexistent/token")
    assert client.get("/measure/status").json()["configured"] is False
    r = client.post("/measure/ask", json={"model": scene, "question": "How wide is this doorway?", "camera": CAMERA})
    assert r.status_code == 503 and "not configured" in r.json()["detail"]


def test_page_has_measure_panel(client):
    html = client.get("/").text
    assert 'id="measure-section"' in html and "/web/measure.js" in html and "Measure.render(" in html


def test_rag_notes_are_sent_as_measurement_references(client, scene, bp):
    client.post("/metadata/object/obj1", data={"label": "front door", "text": "The front door is 36 inches wide.", "model": scene})
    job = _wait(client, client.post("/measure/ask", json={"model": scene, "question": "How wide is this doorway?", "camera": CAMERA}).json()["job_id"])
    assert job["state"] == "completed"
    sent = [c for c in bp.calls if c[1].endswith("/ask-view")][-1][2]["rag_sources"]
    assert {"source": "SAM3 object note 'front door'", "subject": "front door", "text": "The front door is 36 inches wide."} in sent
