"""Saved display orientation (new_peacock.ply loaded upside down, 2026-10-23): save / reset / applied on load."""
import httpx
import numpy as np
import pytest

from support.ply import grid_points, orbit_cameras, write_splat_ply


@pytest.fixture
def blueprints(monkeypatch, tmp_path):
    """In-memory Blueprints /viewer-orientations double behind the real bearer-token client."""
    from features import blueprints_client as bp, orient_override
    store, calls = {}, []
    tok = tmp_path / "tok"
    tok.write_text("tok-viewer\n")
    monkeypatch.setattr(bp, "BLUEPRINTS_TOKEN_FILE", str(tok))
    orient_override._cache.clear()

    def handler(req: httpx.Request):
        calls.append((req.method, req.url.path, req.headers.get("authorization")))
        key = req.url.path.split("/viewer-orientations/")[1]
        if req.method == "GET":
            return httpx.Response(200, json=store[key]) if key in store else httpx.Response(404, json={"detail": "none"})
        if req.method == "PUT":
            import json
            store[key] = {**json.loads(req.content), "key": key}
            return httpx.Response(200, json=store[key])
        store.pop(key, None)
        return httpx.Response(200, json={"deleted": 1})

    real = httpx.Client
    monkeypatch.setattr(bp.httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    return store, calls


def test_save_applies_on_load_and_reset_returns_to_automatic(client, ss, blueprints):
    store, calls = blueprints
    write_splat_ply(ss.MODEL_DIR / "room.ply", points=grid_points(12, extent=2.0))  # no cameras, no usable cue
    assert client.get("/orientation/room.ply").json() == {"model": "room.ply", "quat": None, "can_edit": True, "source": "automatic"}
    assert "&orient=" not in ss.viewer_src("room.ply")
    saved = client.post("/orientation/room.ply", json={"quat": [0, 0, 0, 2.0], "note": "flip"}).json()
    assert saved["source"] == "saved" and saved["quat"] == [0.0, 0.0, 0.0, 1.0]  # normalised before saving
    assert ss.viewer_src("room.ply").endswith("&orient=0.0,0.0,0.0,1.0")
    info = client.get("/flythrough/info/room.ply").json()
    assert info["orient"] == [0.0, 0.0, 0.0, 1.0] and info["orientation"]["source"] == "saved" and info["mode"] == "arc"
    key = next(iter(store))
    assert key.startswith("room.ply:") and key.count(":") == 2  # per exact file (size + mtime)
    assert all(c[2] == "Bearer tok-viewer" for c in calls)
    assert client.delete("/orientation/room.ply").json()["source"] == "automatic" and not store
    assert "&orient=" not in ss.viewer_src("room.ply")


def test_models_with_trained_cameras_and_bad_input_are_refused(client, ss, blueprints):
    write_splat_ply(ss.MODEL_DIR / "cams.ply", points=grid_points(12), cameras=orbit_cameras(8))
    assert client.get("/orientation/cams.ply").json()["can_edit"] is False
    assert client.post("/orientation/cams.ply", json={"quat": [0, 0, 0, 1]}).status_code == 409
    write_splat_ply(ss.MODEL_DIR / "room2.ply", points=grid_points(12))
    for bad in ({"quat": [0, 0, 1]}, {"quat": [0, 0, 0, 0]}, {"quat": "up"}):
        assert client.post("/orientation/room2.ply", json=bad).status_code == 422, bad
    assert client.get("/orientation/missing.ply").status_code == 404


def test_without_blueprints_the_automatic_orientation_is_used(client, ss, monkeypatch):
    from features import blueprints_client as bp, orient_override
    monkeypatch.setattr(bp, "BLUEPRINTS_TOKEN_FILE", "/nonexistent")
    orient_override._cache.clear()
    write_splat_ply(ss.MODEL_DIR / "room3.ply", points=grid_points(12))
    assert client.get("/orientation/room3.ply").json()["source"] == "automatic"
    assert client.post("/orientation/room3.ply", json={"quat": [0, 0, 0, 1]}).status_code == 503
    assert np.allclose(orient_override.matrix([0, 0, 1, 0]), np.diag([-1.0, -1, 1]))  # the viewer default (0, 0, 180)
