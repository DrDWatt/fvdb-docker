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
