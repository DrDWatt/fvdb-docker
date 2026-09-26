"""Helpers for integration tests against running viewers."""
import io
from pathlib import Path

import numpy as np
from PIL import Image

from support.wait import wait_for

UPLOAD_NAME = "_pytest_upload.ply"


def png_array(resp, jpeg=False):
    """Decode an image response after checking status and content type."""
    assert resp.status_code == 200, resp.text[:200]
    assert resp.headers["content-type"] == ("image/jpeg" if jpeg else "image/png")
    return np.array(Image.open(io.BytesIO(resp.content)).convert("RGB"))


def load_8085(client, model):
    """Load (or re-load) a model on :8085 and wait until it is on the GPU."""
    if client.get("/info").json().get("model_name") == Path(model).stem:
        return
    client.get(f"/load_model?model={model}")
    wait_for(lambda: client.get("/load_status").json()["state"] in ("done", "error"), timeout=300, interval=1)
    assert client.get("/info").json()["model_name"] == Path(model).stem


def reconstruct_with_trellis(trellis, job_id, timeout=1800):
    """Wait for a TRELLIS.2 job, check the textured mesh downloads, then delete the job
    (TRELLIS.2 is shared with the dev stack, so tests must not leave jobs behind)."""
    try:
        job = wait_for(lambda: (j := trellis.get(f"/jobs/{job_id}").json())["status"] in ("completed", "failed") and j,
                       timeout=timeout, interval=10, message=f"TRELLIS job {job_id} did not finish")
        assert job["status"] == "completed", job
        mesh = trellis.get(f"/download/{job_id}")
        assert mesh.status_code == 200 and len(mesh.content) > 10_000
        return job
    finally:
        trellis.delete(f"/jobs/{job_id}")
