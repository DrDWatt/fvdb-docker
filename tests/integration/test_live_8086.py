"""Live :8086 (SuperSplat viewer build, SAM3, Ollama, H.264 encoder) on the isolated test stack.

Browser-side behaviour (interactive navigation, native flythrough playback and the
in-browser captureFrame export) is covered by tests/browser.
"""
import base64

import pytest

from support.fakes import sse_tokens
from support.live import UPLOAD_NAME, reconstruct_with_trellis
from support.media import assert_universal_mp4, jpeg_bytes, probe_mp4
from support.ply import read_vertex_count
from support.wait import wait_for

PROMPT = "table"          # reliably present in the seed scene (orange exhibit table)


@pytest.fixture(scope="module")
def frame(v8085):
    """A real rendered view of the seed model, standing in for the browser canvas capture."""
    png = v8085.get("/render?width=1024&height=768&azimuth=0&elevation=0&zoom=1.0").content
    return "data:image/png;base64," + base64.b64encode(png).decode()


@pytest.fixture(scope="module")
def segmented(v8086, frame):
    data = v8086.post("/segment/text", json={"prompt": PROMPT, "image": frame}).json()
    assert data["status"] == "ok" and data["num_masks"] >= 1, data
    return data


@pytest.fixture(scope="module")
def extraction(v8086, segmented, seed_model):
    job = v8086.post("/garfield/extract", json={"model": seed_model, "mask_index": 0}).json()
    assert job["status"] == "ok", job
    status = wait_for(lambda: (s := v8086.get(f"/garfield/status/{job['job_id']}").json())["status"]
                      in ("done", "error") and s, timeout=120, interval=0.5)
    assert status["status"] == "done" and status["num_gaussians"] > 0, status
    return status


# ----- viewer / navigation wiring -----
def test_viewer_build_exposes_the_camera_apis_the_features_use(v8086):
    bundle = v8086.get("/viewer/index.js").text
    for api in ("window.scrubTo", "window.captureFrame", "window.getCameraMatrices", "animTracks"):
        assert api in bundle, api
    html = v8086.get("/").text
    assert 'id="viewer-iframe"' in html and "/web/flythrough.js" in html and "/web/rag.js" in html


# ----- PLY upload / download -----
def test_ply_upload_download_and_delete(v8086, upload_ply):
    assert v8086.post("/upload_model", files={"file": (UPLOAD_NAME, upload_ply)}).json()["status"] == "ok"
    try:
        assert v8086.get(f"/models/{UPLOAD_NAME}").content == upload_ply
        assert UPLOAD_NAME in v8086.get("/info").json()["models_available"]
    finally:
        assert v8086.delete(f"/delete_model?model={UPLOAD_NAME}").json()["status"] == "ok"
    assert v8086.get(f"/models/{UPLOAD_NAME}").status_code == 404


def test_seed_model_downloads_for_the_viewer(v8086, seed_model):
    with v8086.stream("GET", f"/models/{seed_model}") as resp:
        assert resp.status_code == 200 and int(resp.headers["content-length"]) > 1_000_000


# ----- segmentation -----
def test_sam3_segments_objects_from_a_text_prompt(v8086, segmented):
    assert all(0 < m["score"] <= 1 for m in segmented["masks"])
    assert base64.b64decode(segmented["overlay"])[:8] == b"\x89PNG\r\n\x1a\n"
    assert v8086.get("/segment/mask_overlay/0").json()["status"] == "ok"


# ----- 3D extraction / reconstruction -----
def test_extraction_of_the_segmented_object_downloads(v8086, extraction, tmp_path):
    ply = tmp_path / "extraction.ply"
    ply.write_bytes(v8086.get(f"/garfield/download/{extraction['job_id']}").content)
    assert read_vertex_count(ply) == extraction["num_gaussians"]


@pytest.mark.slow
def test_trellis_reconstructs_a_mesh_via_the_viewer_proxy(v8086, extraction, trellis):
    data = v8086.post("/trellis/reconstruct", json={"job_id": extraction["job_id"]}).json()
    assert data["status"] == "ok" and data["viewer_url"].endswith(data["job_id"]), data
    reconstruct_with_trellis(trellis, data["job_id"])


# ----- metadata -----
def test_object_and_extraction_metadata_typed_and_uploaded(v8086, segmented, extraction, seed_model):
    obj = f"/metadata/object/{seed_model}%7C{PROMPT}%7C0"
    v8086.post(obj, data={"label": "pytest-table", "text": "Orange exhibit table", "model": seed_model})
    v8086.post(f"{obj}/upload", data={"model": seed_model},
               files=[("files", ("spec.txt", b"Table load rating 250 kg", "text/plain"))])
    summary = v8086.get(obj).json()["summary"]
    assert summary["label"] == "pytest-table" and summary["training_data"]["files_count"] >= 1
    assert v8086.get(f"{obj}/file/0").content == b"Table load rating 250 kg"

    ext = f"/metadata/extraction/{extraction['job_id']}"
    v8086.post(ext, data={"label": "pytest-extraction", "text": "Isolated table", "model": seed_model})
    assert v8086.get(ext).json()["summary"]["text"] == "Isolated table"


# ----- RAG data upload + chat -----
def test_rag_document_upload(v8086, seed_model):
    resp = v8086.post(f"/upload_summary?model={seed_model}",
                      files={"file": ("site.md", b"APL exhibit booth with a drone on an orange table.", "text/markdown")})
    assert resp.status_code == 200
    assert "orange table" in v8086.get(f"/model_summary?model={seed_model}").json()["summary"]


def test_rag_chat_answers_from_linked_metadata(v8086, seed_model, segmented, extraction):
    v8086.post(f"/metadata/object/{seed_model}%7C{PROMPT}%7C0",
               data={"label": "pytest-table", "text": "Orange exhibit table", "model": seed_model})
    assert v8086.get("/rag/status").json()["available"] is True
    ctx = v8086.get(f"/rag/context?model={seed_model}").json()
    assert ctx["segments_count"] >= 1 and ctx["extractions_count"] >= 1 and "pytest-table" in ctx["segment_labels"]
    resp = v8086.post("/rag/query", json={"query": "What is the table used for?", "model": seed_model,
                                          "history": []}, timeout=180)
    text, done, errors = sse_tokens(resp.text)
    assert not errors and done and len(text.strip()) > 10


# ----- flythrough + MP4 export -----
def test_flythrough_track_follows_the_trained_cameras(v8086, seed_model):
    info = v8086.get(f"/flythrough/info/{seed_model}").json()
    assert info["mode"] == "camera_path" and info["num_cameras"] > 10 and info["duration"] == 30
    track = v8086.get(f"/flythrough/settings/{seed_model}").json()["animTracks"][0]
    assert track["duration"] == 60 and len(track["keyframes"]["times"]) == 600


def test_mp4_export_session_encodes_a_universal_thirty_second_video(v8086, seed_model):
    session = v8086.post("/flythrough/export/start", json={"model": seed_model, "duration": 10, "fps": 30,
                                                           "width": 640, "height": 360}).json()
    assert session["num_frames"] == 900
    for first in range(0, 900, 100):
        v8086.post(f"/flythrough/export/{session['session_id']}/frames",
                   files=[("frames", (f"f{i}.jpg", jpeg_bytes(640, 360, shade=i), "image/jpeg"))
                          for i in range(first, first + 100)])
    mp4 = v8086.post(f"/flythrough/export/{session['session_id']}/finish")
    assert mp4.status_code == 200
    assert_universal_mp4(probe_mp4(mp4.content), min_seconds=30, fps=30, size=(640, 360))
