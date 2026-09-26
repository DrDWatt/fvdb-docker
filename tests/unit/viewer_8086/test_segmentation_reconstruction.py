""":8086 SAM3 segmentation, GARField 3D extraction and TRELLIS.2 reconstruction."""
import base64
import io
import json

import numpy as np
from PIL import Image

from support.fakes import FakeSam3Processor
from support.ply import read_positions, read_vertex_count
from support.wait import wait_for

W, H = 640, 360


def frame_data_url():
    buf = io.BytesIO()
    Image.new("RGB", (W, H), (40, 40, 60)).save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def look_at_origin_camera(distance=3.0, fov_deg=60.0):
    """PlayCanvas-style column-major view/proj: camera on +z looking at the origin."""
    view = np.eye(4)
    view[2, 3] = -distance
    f = 1 / np.tan(np.radians(fov_deg) / 2)
    near, far = 0.1, 100.0
    proj = np.array([[f * H / W, 0, 0, 0], [0, f, 0, 0],
                     [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)], [0, 0, -1, 0]])
    return {"view": view.T.ravel().tolist(), "proj": proj.T.ravel().tolist(), "model": None}


def segment(client, prompt="car", camera=None):
    return client.post("/segment/text", json={"prompt": prompt, "image": frame_data_url(), "camera": camera}).json()


def extract(client, model, mask_index=0):
    job = client.post("/garfield/extract", json={"model": model, "mask_index": mask_index}).json()
    assert job["status"] == "ok", job
    status = wait_for(lambda: (s := client.get(f"/garfield/status/{job['job_id']}").json())["status"]
                      in ("done", "error") and s)
    assert status["status"] == "done", status
    return job["job_id"], status["num_gaussians"]


# ----- segmentation -----
def test_text_prompt_is_required(client, sam3):
    assert client.post("/segment/text", json={"prompt": ""}).json()["status"] == "error"


def test_segmentation_returns_masks_scores_and_overlay(client, ss, sam3):
    data = segment(client)
    assert data["status"] == "ok" and data["num_masks"] == 2 and sam3.prompts == ["car"]
    assert [round(m["score"], 2) for m in data["masks"]] == [0.97, 0.81]
    overlay = np.array(Image.open(io.BytesIO(base64.b64decode(data["overlay"]))))
    assert overlay.shape == (H, W, 4)
    assert overlay[H // 2, int(0.3 * W), 3] > 0          # inside mask 0
    assert overlay[5, 5, 3] == 0                          # background untouched
    assert sorted(p.name for p in ss.CACHE_DIR.glob("mask_*.npy")) == ["mask_0.npy", "mask_1.npy"]
    assert ss.last_segmentation["prompt"] == "car"      # feeds the RAG context


def test_camera_matrices_are_saved_for_accurate_extraction(client, ss, sam3):
    segment(client, camera=look_at_origin_camera())
    assert json.loads((ss.CACHE_DIR / "camera.json").read_text())["view"][14] == -3.0
    segment(client)                                       # new frame without camera: stale file removed
    assert not (ss.CACHE_DIR / "camera.json").exists()


def test_single_object_highlight_overlay(client, sam3):
    segment(client)
    overlay = client.get("/segment/mask_overlay/1").json()
    assert overlay["status"] == "ok" and overlay["mask_index"] == 1
    assert client.get("/segment/mask_overlay/7").json()["status"] == "error"


def test_clear_removes_masks_and_rag_detections(client, ss, sam3):
    segment(client)
    assert client.post("/segment/clear").json()["status"] == "ok"
    assert not list(ss.CACHE_DIR.glob("mask_*.npy")) and ss.last_segmentation == {}


# ----- 3D extraction (GARField-style) -----
def test_extraction_requires_a_segmented_mask(client, scene):
    resp = client.post("/garfield/extract", json={"model": scene, "mask_index": 0}).json()
    assert resp["status"] == "error" and "Mask 0 not found" in resp["error"]


def test_orthographic_extraction_selects_gaussians_under_the_mask(client, ss, sam3, scene, tmp_path):
    segment(client)                                  # mask 0 covers x in [10%, 50%], y in [25%, 75%]
    job_id, count = extract(client, scene)
    ply = tmp_path / "ext.ply"
    ply.write_bytes(client.get(f"/garfield/download/{job_id}").content)
    pts = read_positions(ply)
    assert read_vertex_count(ply) == count == len(pts) and 40 < count < 120
    assert pts[:, 0].min() >= -0.85 and pts[:, 0].max() <= 0.05
    assert pts[:, 1].min() >= -0.55 and pts[:, 1].max() <= 0.55


def test_perspective_extraction_matches_the_segmented_screen_region(client, ss, monkeypatch, scene, tmp_path):
    from support.fakes import install_fake_torch
    install_fake_torch(monkeypatch)
    monkeypatch.setattr(ss, "load_sam3", lambda: True)
    monkeypatch.setattr(ss, "sam3_processor", FakeSam3Processor(boxes=((0.0, 0.0, 1.0, 0.5),), scores=(0.9,)))
    segment(client, camera=look_at_origin_camera())  # mask = left half of the screen
    job_id, count = extract(client, scene)
    ply = tmp_path / "ext.ply"
    ply.write_bytes(client.get(f"/garfield/download/{job_id}").content)
    assert count == 200                              # exactly the 10 of 20 grid columns with x < 0
    assert (read_positions(ply)[:, 0] < 0).all()


def test_extraction_status_download_and_clear(client, ss, sam3, scene):
    assert client.get("/garfield/status/nope").json()["status"] == "error"
    assert client.get("/garfield/download/nope").status_code == 404
    segment(client)
    job_id, _ = extract(client, scene)
    assert client.post("/garfield/clear").json()["status"] == "ok"
    assert not list(ss.OUTPUT_DIR.glob("extraction_*.ply"))
    assert client.get(f"/garfield/download/{job_id}").status_code == 404


# ----- TRELLIS.2 mesh reconstruction -----
def test_trellis_receives_the_cropped_object_and_returns_viewer_link(client, sam3, scene, services):
    segment(client)
    job_id, _ = extract(client, scene)
    data = client.post("/trellis/reconstruct", json={"job_id": job_id}).json()
    assert data["status"] == "ok" and data["job_id"] == "trellis1"
    assert data["viewer_url"] == "http://localhost:8013/viewer/trellis1"
    assert services.trellis_jobs[0]["name"] == f"extraction_{job_id}.png"
    assert services.trellis_jobs[0]["bytes"] > 100


def test_trellis_errors_are_reported(client, sam3, scene, services):
    assert client.post("/trellis/reconstruct", json={"job_id": "missing"}).json()["error"] == "Extraction job not found"
    segment(client)
    job_id, _ = extract(client, scene)
    services.trellis_up = False
    error = client.post("/trellis/reconstruct", json={"job_id": job_id}).json()
    assert error["status"] == "error" and "Cannot connect to TRELLIS" in error["error"]
