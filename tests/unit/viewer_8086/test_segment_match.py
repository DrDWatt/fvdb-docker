"""SAM3 segment -> RAG catalogue suggestions (Object Metadata, 2026-10-23)."""
import httpx
import numpy as np
from PIL import Image


def test_crop_masks_background_and_measures_aspect():
    from features.segment_match import crop
    frame = Image.new("RGB", (100, 80), (200, 20, 20))
    mask = np.zeros((80, 100), np.float32)
    mask[10:70, 40:60] = 1.0  # 60 tall x 20 wide
    png, aspect = crop(frame, mask[None])
    img = np.asarray(Image.open(__import__("io").BytesIO(png)))
    assert aspect == 3.0 and img.shape[:2] == (76, 36)                  # bbox + 8 px padding
    assert tuple(img[0, 0]) == (128, 128, 128) and tuple(img[38, 18]) == (200, 20, 20)  # background grey, object kept


def test_rag_matches_sends_crop_label_and_aspect(client, ss, monkeypatch, tmp_path):
    from features import blueprints_client as bp
    tok = tmp_path / "tok"
    tok.write_text("tok-viewer\n")
    monkeypatch.setattr(bp, "BLUEPRINTS_TOKEN_FILE", str(tok))
    Image.new("RGB", (100, 80), (10, 10, 200)).save(ss.CACHE_DIR / "current_frame.png")
    m = np.zeros((80, 100), np.float32)
    m[10:70, 40:60] = 1.0
    np.save(ss.CACHE_DIR / "mask_0.npy", m)
    ss.last_segmentation.update(prompt="blue vase", model="room.ply", masks=[{"index": 0, "score": 0.9}])
    seen = []

    def handler(req: httpx.Request):
        seen.append((req.url.path, req.headers.get("authorization"), req.content))
        return httpx.Response(200, json={"description": "baluster vase", "candidates": [{"name": "Baluster vase", "url": "u", "score": 6.1}]})

    real = httpx.Client
    monkeypatch.setattr(bp.httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    out = client.post("/segment/0/rag-matches", json={"model": "room.ply"}).json()
    assert out["candidates"][0]["name"] == "Baluster vase" and out["mask_hw"] == 3.0
    path, auth, body = seen[0]
    assert path == "/rag/match" and auth == "Bearer tok-viewer"
    assert b'name="label"\r\n\r\nblue vase' in body and b'name="aspect"\r\n\r\n3.0000' in body and b"\x89PNG" in body
    assert client.post("/segment/7/rag-matches", json={"model": "room.ply"}).status_code == 404
    assert client.post("/segment/0/rag-matches", json={}).status_code == 422


def _cam_looking_down_minus_z(W=200, H=200, f=200.0, z=5.0):
    """PlayCanvas-style column-major view/proj/model: camera at (0, 0, z) looking down -Z, identity model."""
    view = np.eye(4); view[2, 3] = -z
    n, fa = 0.1, 100.0
    proj = np.zeros((4, 4)); proj[0, 0] = 2 * f / W; proj[1, 1] = 2 * f / H
    proj[2, 2] = -(fa + n) / (fa - n); proj[2, 3] = -2 * fa * n / (fa - n); proj[3, 2] = -1
    cm = lambda m: m.T.reshape(-1).tolist()
    return {"view": cm(view), "proj": cm(proj), "model": cm(np.eye(4)), "width": W, "height": H}


def test_object_measure_takes_the_front_object_not_the_wall_behind(ss):
    """SYNTHETIC: a 1.2 x 0.4 'vase' at z=0 in front of a big wall at z=-2; both project into the same mask."""
    from features.object_measure import measure
    from support.ply import write_splat_ply
    g = np.linspace(-1, 1, 40)
    X, Y = np.meshgrid(g * 0.2, g * 0.6)
    vase = np.c_[X.ravel(), Y.ravel(), np.zeros(X.size)]
    WX, WY = np.meshgrid(g * 3, g * 3)
    wall = np.c_[WX.ravel(), WY.ravel(), np.full(WX.size, -2.0)]
    write_splat_ply(ss.MODEL_DIR / "vase.ply", points=np.r_[vase, wall])
    mask = np.zeros((200, 200), np.float32)
    mask[100 - 26:100 + 26, 100 - 9:100 + 9] = 1  # the vase's silhouette: 1.2 x 0.4 at distance 5, f = 200
    r = measure(ss.MODEL_DIR / "vase.ply", mask, _cam_looking_down_minus_z())
    assert abs(r["height_mu"] - 1.2) < 0.1 and abs(r["width_mu"] - 0.4) < 0.06  # not the wall's 6 x 6
    assert np.allclose(np.subtract(r["q"], r["p"]) / np.linalg.norm(np.subtract(r["q"], r["p"])), [0, 1, 0], atol=1e-6)
    assert r["n_points"] < r["n_in_mask"]  # wall splats behind the vase were dropped


def test_link_scale_measures_and_posts_to_the_scene(client, ss, monkeypatch, tmp_path):
    import json as _json
    from features import blueprints_client as bp
    from support.ply import write_splat_ply
    tok = tmp_path / "tok"
    tok.write_text("tok-viewer\n")
    monkeypatch.setattr(bp, "BLUEPRINTS_TOKEN_FILE", str(tok))
    g = np.linspace(-1, 1, 40)
    X, Y = np.meshgrid(g * 0.2, g * 0.6)
    write_splat_ply(ss.MODEL_DIR / "room.ply", points=np.c_[X.ravel(), Y.ravel(), np.zeros(X.size)])
    mask = np.zeros((200, 200), np.float32)
    mask[74:126, 91:109] = 1
    np.save(ss.CACHE_DIR / "mask_0.npy", mask)
    (ss.CACHE_DIR / "camera.json").write_text(_json.dumps(_cam_looking_down_minus_z()))
    ss.last_segmentation.update(prompt="blue vase", model="room.ply", masks=[{"index": 0, "score": 0.9}])
    sent = []

    def handler(req: httpx.Request):
        if req.url.path.endswith("/import/ply-service"):
            return httpx.Response(200, json={"scene_id": "splat-room", "existing": True})
        sent.append((req.url.path, _json.loads(req.content)))
        return httpx.Response(200, json={"n": 1, "scale": "0.336 m/mu", "verified": False, "leave_one_out": []})

    real = httpx.Client
    monkeypatch.setattr(bp.httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    cand = {"name": "Baluster vase", "url": "https://m/items/show/3397", "height_m": 0.403,
            "dims_text": "HxWxD: 40.3 x 13.0 x 13.0 cm", "fields": {"Dimensions": "HxWxD: 40.3 x 13.0 x 13.0 cm"}}
    out = client.post("/segment/0/link-scale", json={"model": "room.ply", "candidate": cand}).json()
    assert out["n"] == 1 and abs(out["measured"]["height_mu"] - 1.2) < 0.1
    path, body = sent[0]
    assert path == "/projects/peacock/scenes/splat-room/linked-objects"
    assert body["dimensions"] == "HxWxD: 40.3 x 13.0 x 13.0 cm" and body["height_m"] == 0.403 and len(body["p"]) == 3
    assert client.post("/segment/0/link-scale", json={"model": "room.ply", "candidate": {"name": "x", "url": "u", "fields": {}}}).status_code == 422


def test_unsupported_ply_layouts_give_a_clear_error(ss):
    import pytest
    from features.object_measure import measure
    (ss.MODEL_DIR / "packed.ply").write_bytes(b"ply\nformat binary_little_endian 1.0\nelement chunk 1\n"
                                              b"property list uchar float packed\nend_header\n\x00")
    with pytest.raises(ValueError, match="unsupported PLY layout"):
        measure(ss.MODEL_DIR / "packed.ply", np.ones((10, 10), np.float32), _cam_looking_down_minus_z())
