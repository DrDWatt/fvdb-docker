""":8085 navigation (server-rendered orbit/pan/zoom/camera views) and PLY upload / delete."""
import io

import numpy as np
from PIL import Image

from support.ply import write_splat_ply
from support.wait import wait_for


def png(resp):
    assert resp.status_code == 200 and resp.headers["content-type"] == "image/png"
    return np.array(Image.open(io.BytesIO(resp.content)))


# ----- navigation -----
def test_render_forwards_every_navigation_parameter(client, renderer, loaded_model):
    img = png(client.get("/render?width=640&height=480&azimuth=45&elevation=-20&zoom=2.5&cam_idx=3"
                         "&pan_x=0.5&pan_y=-0.25&pan_z=1"))
    assert img.shape == (480, 640, 3)
    assert renderer.calls[-1] == dict(width=640, height=480, azimuth=45, elevation=-20, zoom=2.5, cam_idx=3,
                                      pan_x=0.5, pan_y=-0.25, pan_z=1)


def test_orbit_zoom_and_pan_change_the_view(client, renderer, loaded_model):
    base = png(client.get("/render?azimuth=0"))
    for query in ("azimuth=90", "elevation=30", "zoom=3", "pan_x=2"):
        assert not np.array_equal(base, png(client.get(f"/render?{query}"))), query


def test_navigation_limits_are_validated(client, renderer, loaded_model):
    for query in ("elevation=95", "azimuth=400", "zoom=0", "zoom=25", "width=50", "height=2000", "cam_idx=-1"):
        assert client.get(f"/render?{query}").status_code == 422, query
    assert renderer.calls == []


def test_render_failure_is_a_server_error(client, ivs, monkeypatch):
    monkeypatch.setattr(ivs, "render_view", lambda *a, **k: None)
    assert client.get("/render").status_code == 500


def test_page_has_navigation_controls(client, loaded_model):
    html = client.get("/").text
    for marker in ('id="azimuth"', 'id="elevation"', 'id="zoom"', 'id="camera"', "WASD: pan", "Arrows: orbit",
                   "document.addEventListener('keydown'"):
        assert marker in html, marker


# ----- PLY upload / delete / model switching -----
def test_ply_upload_is_stored_and_listed(client, ivs, loaded_model, tmp_path):
    src = write_splat_ply(tmp_path / "src.ply")
    resp = client.post("/upload_model", files={"file": ("garage.ply", src.read_bytes(), "application/octet-stream")})
    assert resp.status_code == 200 and resp.json()["filename"] == "garage.ply"
    assert (ivs.MODEL_DIR / "garage.ply").read_bytes() == src.read_bytes()
    assert "garage.ply" in client.get("/info").json()["models_available"]


def test_non_ply_upload_is_rejected(client, ivs):
    assert client.post("/upload_model", files={"file": ("a.obj", b"v 0 0 0", "text/plain")}).status_code == 400
    assert not list(ivs.MODEL_DIR.glob("a.*"))


def test_delete_removes_model_and_metadata_then_loads_another(client, ivs, loaded_model, tmp_path):
    write_splat_ply(ivs.MODEL_DIR / "other.ply")
    (ivs.MODEL_DIR / "scene_metadata.json").write_text("{}")
    assert client.delete("/delete_model?model=scene.ply").json()["status"] == "ok"
    assert not (ivs.MODEL_DIR / "scene.ply").exists() and not (ivs.MODEL_DIR / "scene_metadata.json").exists()
    assert client.get("/info").json()["model_name"] == "other"
    assert client.delete("/delete_model?model=scene.ply").status_code == 404


def test_model_switch_runs_in_background(client, ivs, loaded_model):
    write_splat_ply(ivs.MODEL_DIR / "other.ply")
    assert client.get("/load_model?model=other.ply").json()["status"] == "loading"
    status = wait_for(lambda: (s := client.get("/load_status").json())["state"] == "done" and s)
    assert status["model"] == "other" and status["num_gaussians"] == 100
    assert client.get("/info").json()["model_name"] == "other"
