"""Integration fixtures: real viewers (GPU rendering, SAM-2/SAM3, fVDB, Ollama) in the
isolated test stack from docker-compose.test.yml (profile "integration").

The stack has its own model directory (./test-models), rendering service and ports,
so these tests never change the dev viewers' loaded model, segments or metadata.
"""
import os
import shutil
from pathlib import Path

import httpx
import pytest

from support.live import UPLOAD_NAME, load_8085
from support.ply import orbit_cameras, write_splat_ply
from support.wait import wait_for

URL_8085 = os.environ.get("VIEWER_8085_URL", "http://localhost:18085")
URL_8086 = os.environ.get("VIEWER_8086_URL", "http://localhost:18086")
TRELLIS_URL = os.environ.get("TRELLIS_URL", "http://localhost:8013")
TEST_MODELS_DIR = Path(os.environ.get("TEST_MODELS_DIR", "test-models"))
SEED_MODELS_DIR = Path(os.environ.get("SEED_MODELS_DIR", "models"))
SEED_MODEL = os.environ.get("SEED_MODEL", "APL-copter-ultra_model.ply")


def pytest_collection_modifyitems(items):
    for item in items:
        if "/integration/" in str(item.fspath):
            item.add_marker(pytest.mark.integration)


def _reachable(url):
    try:
        return httpx.get(f"{url}/health", timeout=5).status_code == 200
    except httpx.HTTPError:
        return False


@pytest.fixture(scope="session")
def seed_model():
    """Copy one real trained model (with cameras) into the test stack's model dir."""
    src, dst = SEED_MODELS_DIR / SEED_MODEL, TEST_MODELS_DIR / SEED_MODEL
    if not src.exists():
        pytest.skip(f"seed model {src} not available")
    TEST_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if not dst.exists() or dst.stat().st_size != src.stat().st_size:
        shutil.copyfile(src, dst)
    return SEED_MODEL


def _client(url):
    if not _reachable(url):
        pytest.skip(f"test viewer not reachable at {url} "
                    "(docker compose -f docker-compose.test.yml --profile integration up -d)")
    return httpx.Client(base_url=url, timeout=httpx.Timeout(600, connect=10))


@pytest.fixture(scope="session")
def v8085(seed_model):
    with _client(URL_8085) as client:
        load_8085(client, seed_model)
        yield client


@pytest.fixture(scope="session")
def v8086(seed_model):
    with _client(URL_8086) as client:
        client.get(f"/load_model?model={seed_model}")
        wait_for(lambda: client.get("/load_status").json()["state"] == "done", timeout=30, interval=0.5)
        yield client


@pytest.fixture(scope="session")
def trellis():
    if not _reachable(TRELLIS_URL):
        pytest.skip(f"TRELLIS.2 not reachable at {TRELLIS_URL}")
    health = httpx.get(f"{TRELLIS_URL}/health", timeout=5).json()
    # TRELLIS.2 needs ~40 GB of the GB10's unified memory; skip rather than risk an OOM
    if not health.get("pipeline_loaded") and health.get("memory_available_gb", 0) < health.get("memory_required_gb", 0):
        pytest.skip(f"TRELLIS.2 needs {health['memory_required_gb']} GB, {health['memory_available_gb']} GB free")
    with httpx.Client(base_url=TRELLIS_URL, timeout=httpx.Timeout(120, connect=10)) as client:
        yield client


@pytest.fixture
def upload_ply(tmp_path):
    """A small synthetic splat (with trained cameras) to upload."""
    return write_splat_ply(tmp_path / UPLOAD_NAME, cameras=orbit_cameras(12)).read_bytes()
