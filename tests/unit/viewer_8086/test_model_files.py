"""/models/<ply>: streamed download with progress, ETag/Cache-Control, 304 revalidation, Range passthrough."""


def test_download_progress_and_caching(client, scene, ss):
    data = (ss.MODEL_DIR / scene).read_bytes()
    r = client.get(f"/models/{scene}")
    assert r.status_code == 200 and r.content == data
    assert r.headers["content-length"] == str(len(data)) and "max-age" in r.headers["cache-control"]
    etag = r.headers["etag"]
    p = client.get(f"/models-progress/{scene}").json()
    assert p["done"] and p["bytes_sent"] == p["total"] == len(data) and p["percent"] == 100.0 and p["server_now"] >= p["started_at"]
    assert client.get(f"/models/{scene}", headers={"If-None-Match": etag}).status_code == 304  # browser cache revalidation
    part = client.get(f"/models/{scene}", headers={"Range": "bytes=0-99"})
    assert part.status_code == 206 and part.content == data[:100]
    tail = client.get(f"/models/{scene}", headers={"Range": f"bytes={len(data) - 50}-"})  # Chrome resuming a download
    assert tail.status_code == 206 and tail.content == data[-50:] and tail.headers["content-range"] == f"bytes {len(data) - 50}-{len(data) - 1}/{len(data)}"
    p2 = client.get(f"/models-progress/{scene}").json()
    assert p2["done"] and p2["range_start"] == len(data) - 50 and p2["bytes_sent"] == len(data)  # resumes are tracked too
    assert client.get(f"/models/{scene}", headers={"Range": f"bytes={len(data) + 5}-"}).status_code == 416


def test_progress_before_any_download_and_missing_model(client, scene):
    assert client.get(f"/models-progress/{scene}").json() == {"model": scene, "started": False}
    assert client.get("/models/missing.ply").status_code == 404
    html = client.get("/").text
    assert "/web/loading.js" in html and "ModelLoading.track(" in html
