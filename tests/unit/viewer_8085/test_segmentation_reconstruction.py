""":8085 SAM-2 segmentation and GARField 3D extraction / reconstruction endpoints."""
import io

import numpy as np
from PIL import Image

from support.ply import write_splat_ply


def segment(client, **form):
    return client.post("/segment", data={"azimuth": 10, "elevation": 5, "zoom": 1.5, "cam_idx": 2, **form})


# ----- segmentation -----
def test_auto_segmentation_filters_and_orders_masks(client, renderer, sam2, loaded_model):
    data = segment(client).json()
    # background blob (>50% of the frame) and the 5x5 speck (<0.1%) are dropped
    assert data["status"] == "ok" and data["num_segments"] == 2
    assert data["labels"] == {"0": "object_120000", "1": "object_75000"}         # largest first
    assert data["scores"] == [0.89, 0.88]
    assert renderer.calls[-1] | {"pan_x": 0} == dict(width=1024, height=768, azimuth=10, elevation=5, zoom=1.5,
                                                     cam_idx=2, pan_x=0, pan_y=0.0, pan_z=0.0)


def test_segment_masks_are_downsampled_for_hover(client, renderer, sam2, loaded_model):
    segment(client)
    masks = client.get("/segment/masks").json()
    assert masks["scale"] == 4 and len(masks["masks"]) == 2
    assert np.array(masks["masks"][0]).shape == (192, 256)


def test_click_segmentation_adds_the_best_mask(client, renderer, sam2, loaded_model):
    segment(client)
    data = client.post("/segment/point", data={"x": 800, "y": 200, "zoom": 1.5}).json()
    assert data == {"status": "ok", "num_segments": 3, "new_segment_idx": 2}
    labels = client.get("/segment/labels").json()
    assert labels["num_segments"] == 3 and labels["labels"]["2"] == "object_6400"    # 80x80 best-score box


def test_segments_can_be_labelled(client, renderer, sam2, loaded_model):
    segment(client)
    assert client.post("/segment/label", data={"segment_idx": 1, "label": "Charger"}).json()["label"] == "Charger"
    assert client.get("/segment/labels").json()["labels"]["1"] == "Charger"
    assert client.post("/segment/label", data={"segment_idx": 9, "label": "x"}).status_code == 400


def test_overlay_render_highlights_segments(client, renderer, sam2, loaded_model):
    segment(client)
    plain = np.array(Image.open(io.BytesIO(client.get("/render?width=1024&height=768").content)))
    overlay = np.array(Image.open(io.BytesIO(client.get("/render_with_segments").content)))
    assert overlay.shape == plain.shape
    assert not np.array_equal(overlay[250, 300], plain[250, 300])                   # inside segment 0
    assert np.array_equal(overlay[20, 1010], plain[20, 1010])                         # outside all segments


def test_clear_and_failure_paths(client, ivs, renderer, sam2, loaded_model, monkeypatch):
    segment(client)
    assert client.post("/segment/clear").json()["status"] == "ok"
    assert client.get("/segment/labels").json() == {"labels": {}, "num_segments": 0}
    monkeypatch.setattr(ivs, "sam2_loaded", False)
    monkeypatch.setattr(ivs, "load_sam2", lambda: False)
    assert segment(client).status_code == 500


# ----- GARField 3D extraction / reconstruction -----
def test_extraction_requires_a_loaded_model(client, ivs, sam2):
    ivs.gsplat = None
    data = client.post("/garfield/extract", data={"x": 10, "y": 10, "model_name": "scene.ply"}).json()
    assert data["status"] == "error" and data["error"] == "No model loaded"


def cached_extraction(ivs, tmp_path, job_id="ab12cd34"):
    ply = write_splat_ply(ivs.MODEL_DIR / f"_extraction_{job_id}.ply")
    ivs.extraction_cache[job_id] = {"output_path": str(ply), "indices": list(range(100)), "model_name": "scene"}
    return job_id, ply


def test_extracted_ply_downloads(client, ivs, tmp_path):
    job_id, ply = cached_extraction(ivs, tmp_path)
    resp = client.get(f"/garfield/download/{job_id}")
    assert resp.status_code == 200 and resp.content == ply.read_bytes()
    assert f'filename="extracted_{job_id}.ply"' in resp.headers["content-disposition"]
    assert client.get("/garfield/download/missing").status_code == 404


def test_extraction_view_render_feeds_reconstruction(client, ivs, tmp_path, monkeypatch):
    """The browser sends this render of the isolated object to TRELLIS.2 for meshing."""
    job_id, _ = cached_extraction(ivs, tmp_path)
    calls = []

    def render_extracted(job, width, height, azimuth, elevation, zoom):
        calls.append((job, width, height, azimuth))
        return np.full((height, width, 3), 200, dtype=np.uint8)
    monkeypatch.setattr(ivs, "render_extracted_gaussians", render_extracted)
    resp = client.get(f"/garfield/render_extraction?job_id={job_id}&azimuth=30&width=512&height=512")
    assert resp.headers["content-type"] == "image/png"
    assert Image.open(io.BytesIO(resp.content)).size == (512, 512)
    assert calls == [(job_id, 512, 512, 30)]
    assert client.get("/garfield/render_extraction?job_id=missing").status_code == 404


def test_clear_extractions_deletes_temp_files(client, ivs, tmp_path):
    job_id, ply = cached_extraction(ivs, tmp_path)
    orphan = write_splat_ply(ivs.MODEL_DIR / "_extraction_orphan.ply")
    assert client.post("/garfield/clear").json() == {"status": "ok", "deleted": 2}
    assert not ply.exists() and not orphan.exists()
    assert client.get(f"/garfield/download/{job_id}").status_code == 404
