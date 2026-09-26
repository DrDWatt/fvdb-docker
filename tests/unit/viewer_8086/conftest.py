"""Fixtures for the SuperSplat viewer (:8086) service, run in-process.

Each test gets a freshly imported supersplat_service whose model/output/cache
directories are temporary, so tests never touch real models or state. SAM3 and
the external HTTP services (rendering service, Ollama, TRELLIS.2) are faked.
"""
import importlib
import sys

import pytest
from fastapi.testclient import TestClient

from support.fakes import FakeSam3Processor, FakeServices, install_fake_torch
from support.ply import grid_points, orbit_cameras, write_splat_ply


@pytest.fixture
def ss(tmp_path, monkeypatch):
    for name in ("models", "outputs", "cache"):
        (tmp_path / name).mkdir()
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    sys.modules.pop("supersplat_service", None)
    module = importlib.import_module("supersplat_service")
    yield module
    sys.modules.pop("supersplat_service", None)


@pytest.fixture
def client(ss):
    # context manager keeps one event loop alive so background jobs can finish
    with TestClient(ss.app) as c:
        yield c


@pytest.fixture
def services(monkeypatch):
    return FakeServices().install(monkeypatch)


@pytest.fixture
def sam3(ss, monkeypatch):
    install_fake_torch(monkeypatch)
    processor = FakeSam3Processor()
    monkeypatch.setattr(ss, "load_sam3", lambda: True)
    monkeypatch.setattr(ss, "sam3_processor", processor)
    return processor


@pytest.fixture
def scene(ss):
    """A 20x20 grid of gaussians in the z=0 plane plus 24 trained orbit cameras."""
    write_splat_ply(ss.MODEL_DIR / "scene.ply", points=grid_points(20), cameras=orbit_cameras(24))
    ss.current_model = "scene.ply"
    return "scene.ply"
