"""Fixtures for the fVDB image viewer (:8085) service, run in-process.

The GPU pieces (fVDB gaussian rendering, SAM-2, CLIP labelling) are replaced by
deterministic fakes so the viewer's own logic - request handling, navigation
parameters, segment bookkeeping, metadata, RAG, flythrough/export - is tested
without a GPU. Real rendering is covered by tests/integration.
"""
import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from support.fakes import FakeSam2Predictor, FakeServices, install_fake_sam2, install_fake_torch
from support.ply import orbit_cameras, write_splat_ply


class FakeTensor:
    """numpy-backed stand-in for the torch tensors fVDB returns as PLY metadata."""

    def __init__(self, array):
        self.array = np.asarray(array)
        self.shape = self.array.shape

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class FakeGsplat:
    def __init__(self, n):
        self.num_gaussians = n


class Renderer:
    """Deterministic render_view(): image colour encodes the camera parameters."""

    def __init__(self):
        self.calls = []

    def __call__(self, width=800, height=600, azimuth=0, elevation=0, zoom=1.0, cam_idx=0,
                 pan_x=0.0, pan_y=0.0, pan_z=0.0):
        self.calls.append(dict(width=width, height=height, azimuth=azimuth, elevation=elevation, zoom=zoom,
                               cam_idx=cam_idx, pan_x=pan_x, pan_y=pan_y, pan_z=pan_z))
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[..., 0] = int(azimuth) % 256
        img[..., 1] = int(elevation + 90) % 256
        img[..., 2] = int(zoom * 50 + pan_x * 10) % 256
        return img


@pytest.fixture
def ivs(tmp_path, monkeypatch):
    (tmp_path / "models").mkdir()
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "models"))
    sys.modules.pop("image_viewer_service", None)
    module = importlib.import_module("image_viewer_service")
    module.app.router.on_startup.clear()        # skip CLIP/SAM preloading
    yield module
    sys.modules.pop("image_viewer_service", None)


@pytest.fixture
def client(ivs):
    with TestClient(ivs.app) as c:
        yield c


@pytest.fixture
def services(monkeypatch):
    return FakeServices().install(monkeypatch)


@pytest.fixture
def renderer(ivs, monkeypatch):
    r = Renderer()
    monkeypatch.setattr(ivs, "render_view", r)
    return r


@pytest.fixture
def loaded_model(ivs, monkeypatch):
    """A PLY on disk plus a fake fVDB load that exposes its trained cameras."""
    write_splat_ply(ivs.MODEL_DIR / "scene.ply", cameras=orbit_cameras(24))
    c2w, K, sizes = orbit_cameras(24)

    def fake_load(model_file=None):
        ivs.available_models = ivs.get_available_models()
        if not ivs.available_models:
            return False
        name = model_file if model_file in ivs.available_models else ivs.available_models[0]
        ivs.model_name = Path(name).stem
        ivs.gsplat = FakeGsplat(100)
        ivs.model_metadata = {"camera_to_world_matrices": FakeTensor(c2w), "projection_matrices": FakeTensor(K),
                              "image_sizes": FakeTensor(sizes)}
        return True

    monkeypatch.setattr(ivs, "load_model", fake_load)
    fake_load("scene.ply")
    return "scene.ply"


@pytest.fixture
def sam2(ivs, monkeypatch):
    """SAM-2 automatic masks: a big background blob (filtered), two objects and a speck."""
    install_fake_torch(monkeypatch)
    h, w = 768, 1024
    masks = []
    for top, left, bottom, right in ((0, 0, 700, 1000), (100, 100, 400, 500), (450, 600, 700, 900), (5, 5, 10, 10)):
        m = np.zeros((h, w), dtype=bool)
        m[top:bottom, left:right] = True
        masks.append(m)
    install_fake_sam2(monkeypatch, masks)
    monkeypatch.setattr(ivs, "sam2_loaded", True)
    monkeypatch.setattr(ivs, "sam2_predictor", FakeSam2Predictor())
    monkeypatch.setattr(ivs, "auto_label_segment", lambda image, mask: f"object_{int(mask.sum())}")
    monkeypatch.setattr(ivs, "assign_gaussians_to_segments", lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, "fvdb", types.ModuleType("fvdb"))
    return masks
