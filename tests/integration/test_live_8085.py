"""Live :8085 (fVDB GPU rendering, SAM-2, CLIP, Ollama) on the isolated test stack."""
import numpy as np
import pytest

from support.fakes import sse_tokens
from support.live import UPLOAD_NAME, load_8085, png_array, reconstruct_with_trellis
from support.media import assert_universal_mp4, probe_mp4
from support.ply import read_vertex_count

VIEW = {"azimuth": 0, "elevation": 0, "zoom": 1.0, "cam_idx": 0}


@pytest.fixture(scope="module")
def segmented(v8085):
    data = v8085.post("/segment", data=VIEW).json()
    assert data["status"] == "ok", data
    return data


def largest_segment_center(v8085):
    masks = [np.array(m) for m in v8085.get("/segment/masks").json()["masks"]]
    i = int(np.argmax([m.sum() for m in masks]))
    ys, xs = np.nonzero(masks[i])
    return i, int(np.median(xs)) * 4, int(np.median(ys)) * 4


@pytest.fixture(scope="module")
def extraction(v8085, segmented, seed_model):
    _, x, y = largest_segment_center(v8085)
    data = v8085.post("/garfield/extract", data={"x": x, "y": y, "model_name": seed_model, **VIEW}).json()
    assert data["status"] == "completed" and data["num_gaussians"] > 0, data
    return data


# ----- navigation -----
def test_navigation_renders_the_scene_from_new_viewpoints(v8085):
    base = png_array(v8085.get("/render?width=640&height=480"))
    assert base.shape == (480, 640, 3) and base.std() > 10                        # not a blank frame
    for query in ("azimuth=60", "elevation=30", "zoom=2.5", "pan_x=0.5", "cam_idx=5"):
        view = png_array(v8085.get(f"/render?width=640&height=480&{query}"))
        assert np.abs(view.astype(int) - base).mean() > 3, query


# ----- segmentation -----
def test_segmentation_detects_and_labels_objects(v8085, segmented):
    assert segmented["num_segments"] >= 3
    assert all(isinstance(label, str) and label for label in segmented["labels"].values())
    overlay = png_array(v8085.get("/render_with_segments?width=1024&height=768&zoom=1.0"))
    plain = png_array(v8085.get("/render?width=1024&height=768&zoom=1.0"))
    assert np.abs(overlay.astype(int) - plain).mean() > 1
    assert v8085.post("/segment/label", data={"segment_idx": 0, "label": "pytest-object"}).json()["status"] == "ok"
    assert v8085.get("/segment/labels").json()["labels"]["0"] == "pytest-object"


def test_click_segmentation_adds_an_object(v8085, segmented):
    before = v8085.get("/segment/labels").json()["num_segments"]
    _, x, y = largest_segment_center(v8085)
    data = v8085.post("/segment/point", data={"x": x, "y": y, **VIEW}).json()
    assert data["status"] == "ok" and data["num_segments"] == before + 1


# ----- 3D extraction / reconstruction -----
def test_extraction_downloads_and_renders_the_isolated_object(v8085, extraction, tmp_path):
    ply = tmp_path / "extraction.ply"
    ply.write_bytes(v8085.get(f"/garfield/download/{extraction['job_id']}").content)
    assert read_vertex_count(ply) == extraction["num_gaussians"]
    render = png_array(v8085.get(f"/garfield/render_extraction?job_id={extraction['job_id']}&width=512&height=512"))
    assert render.shape == (512, 512, 3) and render.std() > 5


@pytest.mark.slow
def test_trellis_reconstructs_a_mesh_from_the_extraction(v8085, extraction, trellis):
    image = v8085.get(f"/garfield/render_extraction?job_id={extraction['job_id']}&width=512&height=512").content
    job = trellis.post("/reconstruct", files={"image": ("extraction.png", image, "image/png")},
                       data={"source_job_id": extraction["job_id"], "label": "pytest"}).json()
    reconstruct_with_trellis(trellis, job["job_id"])


# ----- metadata -----
def test_segmented_object_metadata_typed_and_uploaded(v8085, segmented):
    v8085.post("/object_summary", data={"segment_idx": 1, "label": "pytest-table", "text": "Orange exhibit table"})
    v8085.post("/object_summary/upload", data={"segment_idx": 1},
               files=[("files", ("spec.txt", b"Table load rating 250 kg", "text/plain"))])
    summary = v8085.get("/object_summary/1").json()["summary"]
    assert summary["label"] == "pytest-table" and summary["text"] == "Orange exhibit table"
    assert v8085.get(f"/object_summary/1/file/{summary['files'][-1]['idx']}").content == b"Table load rating 250 kg"


def test_extraction_metadata_typed_and_uploaded(v8085, extraction):
    job = extraction["job_id"]
    v8085.post("/extraction_summary", data={"job_id": job, "label": "pytest-extraction", "text": "Isolated object"})
    v8085.post("/extraction_summary/upload", data={"job_id": job},
               files=[("files", ("notes.md", b"# Extraction notes", "text/markdown"))])
    summary = v8085.get(f"/extraction_summary/{job}").json()["summary"]
    assert summary["label"] == "pytest-extraction" and summary["files"][0]["name"] == "notes.md"


# ----- RAG data upload + chat -----
def test_rag_document_upload(v8085, seed_model):
    resp = v8085.post(f"/upload_summary?model={seed_model}",
                      files={"file": ("site.md", b"APL exhibit booth with a drone on an orange table.", "text/markdown")})
    assert resp.json()["message"] == "Summary uploaded successfully"
    assert "orange table" in v8085.get(f"/model_summary?model={seed_model}").json()["summary"]


def test_rag_chat_answers_from_scene_context(v8085, seed_model, segmented):
    assert v8085.get("/rag/status").json()["available"] is True
    ctx = v8085.get(f"/rag/context?model={seed_model}").json()
    assert ctx["segments_count"] >= 3 and ctx["context_length"] > 0
    resp = v8085.post("/rag/query", json={"query": "What objects are in the scene?", "model": seed_model,
                                          "history": []}, timeout=180)
    text, done, errors = sse_tokens(resp.text)
    assert not errors and done and len(text.strip()) > 10


# ----- PLY upload / download -----
def test_ply_upload_list_and_delete(v8085, upload_ply, seed_model):
    assert v8085.post("/upload_model", files={"file": (UPLOAD_NAME, upload_ply)}).json()["filename"] == UPLOAD_NAME
    assert UPLOAD_NAME in v8085.get("/info").json()["models_available"]
    try:
        assert v8085.delete(f"/delete_model?model={UPLOAD_NAME}").json()["status"] == "ok"
        assert UPLOAD_NAME not in v8085.get("/info").json()["models_available"]
    finally:
        load_8085(v8085, seed_model)                          # deleting reloads the first model


# ----- flythrough + MP4 export -----
def test_flythrough_frames_follow_a_smooth_path(v8085):
    assert v8085.get("/flythrough/config").json()["num_cameras"] > 10
    frame = lambda n: png_array(v8085.get(f"/flythrough/frame/{n}?num_frames=900&width=320&height=240"), jpeg=True)
    a, b, far = frame(450), frame(451), frame(0)
    assert a.std() > 10
    step = np.abs(a.astype(int) - b).mean()
    jump = np.abs(a.astype(int) - far).mean()
    assert step < jump / 3                                    # neighbouring frames are close; path moves


def test_flythrough_exports_a_universal_thirty_second_mp4(v8085):
    resp = v8085.post("/flythrough/export?duration=10&fps=30&width=640&height=480")
    assert resp.status_code == 200 and resp.headers["content-type"] == "video/mp4"
    assert_universal_mp4(probe_mp4(resp.content), min_seconds=30, fps=30, size=(640, 480))
