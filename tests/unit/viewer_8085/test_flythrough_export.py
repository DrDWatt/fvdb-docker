""":8085 flythrough (server-rendered frames along the smoothed camera path) and MP4 export."""
import io

import numpy as np
from PIL import Image

from support.media import assert_universal_mp4, probe_mp4


class FrameRenderer:
    """Stands in for the GPU frame renderer; frame index is encoded in the image."""

    def __init__(self, fail_every=0):
        self.calls, self.fail_every = [], fail_every

    def __call__(self, frame_num, num_frames, width, height, cam_idx=0, return_matrices=False):
        self.calls.append((frame_num, num_frames, width, height))
        if self.fail_every and frame_num % self.fail_every == 5:
            return None
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:, : int(width * frame_num / max(num_frames - 1, 1)) + 1] = (200, 80, 40)
        return img


def test_flythrough_config_reports_trained_cameras(client, loaded_model):
    config = client.get("/flythrough/config").json()
    assert config["num_cameras"] == 24 and config["num_frames"] == 900


def test_smoothed_path_is_built_once_per_model(ivs, loaded_model):
    path = ivs._get_smooth_camera_path()
    assert path["num_cameras"] == 24 and path["arc"][0] == 0 and path["arc"][-1] == 1
    assert ivs._get_smooth_camera_path() is path


def test_flythrough_frames_stream_as_jpeg(client, ivs, loaded_model, monkeypatch):
    renderer = FrameRenderer()
    monkeypatch.setattr(ivs, "render_flythrough_frame", renderer)
    resp = client.get("/flythrough/frame/450?num_frames=900&width=640&height=480")
    assert resp.headers["content-type"] == "image/jpeg"
    assert Image.open(io.BytesIO(resp.content)).size == (640, 480)
    assert renderer.calls == [(450, 900, 640, 480)]
    assert client.get("/flythrough/frame/900?num_frames=900").status_code == 400
    assert client.get("/flythrough/frame/0?num_frames=99999").status_code == 422


def test_page_defaults_to_thirty_second_flythrough(client, loaded_model):
    html = client.get("/").text
    assert 'id="flyDuration" value="30" min="30"' in html
    assert "const FLY_MIN_SECONDS = 30;" in html
    assert 'id="flyProgress" min="0" max="899"' in html


def test_export_is_a_universal_mp4_of_at_least_thirty_seconds(client, ivs, loaded_model, monkeypatch):
    renderer = FrameRenderer()
    monkeypatch.setattr(ivs, "render_flythrough_frame", renderer)
    resp = client.post("/flythrough/export?duration=10&fps=30&width=320&height=240")   # below the minimum
    assert resp.status_code == 200 and resp.headers["content-type"] == "video/mp4"
    assert_universal_mp4(probe_mp4(resp.content), min_seconds=30, fps=30, size=(320, 240))
    assert [c[0] for c in renderer.calls] == list(range(900)) and {c[1] for c in renderer.calls} == {900}


def test_export_honours_fps_and_longer_durations(client, ivs, loaded_model, monkeypatch):
    monkeypatch.setattr(ivs, "render_flythrough_frame", FrameRenderer())
    info = probe_mp4(client.post("/flythrough/export?duration=40&fps=24&width=256&height=144").content)
    assert_universal_mp4(info, min_seconds=40, fps=24, size=(256, 144))
    assert info["frames"] == 960


def test_failed_frames_are_repeated_so_timing_stays_constant(client, ivs, loaded_model, monkeypatch):
    monkeypatch.setattr(ivs, "render_flythrough_frame", FrameRenderer(fail_every=50))
    info = probe_mp4(client.post("/flythrough/export?fps=30&width=160&height=120").content)
    assert info["frames"] == 900 and info["duration"] >= 30


def test_export_fails_cleanly_when_nothing_renders(client, ivs, loaded_model, monkeypatch):
    monkeypatch.setattr(ivs, "render_flythrough_frame", lambda *a, **k: None)
    resp = client.post("/flythrough/export?fps=30&width=160&height=120")
    assert resp.status_code == 500
