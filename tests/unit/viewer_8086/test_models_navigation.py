""":8086 PLY upload / download / delete, model switching and viewer (navigation) wiring.

Navigation itself (orbit/fly/zoom) runs client-side in the SuperSplat viewer; the
server's job is to serve the viewer iframe with the model content and the
flythrough camera track. Interactive navigation is covered by tests/browser.
"""
from urllib.parse import parse_qs, urlparse

from support.ply import read_vertex_count, write_splat_ply
from support.wait import wait_for


def test_ply_upload_then_download_is_byte_identical(client, ss, tmp_path):
    src = write_splat_ply(tmp_path / "upload.ply")
    resp = client.post("/upload_model", files={"file": ("house.ply", src.read_bytes(), "application/octet-stream")})
    assert resp.status_code == 200 and resp.json()["filename"] == "house.ply"
    assert "house.ply" in client.get("/info").json()["models_available"]

    download = client.get("/models/house.ply")
    assert download.status_code == 200
    assert download.content == src.read_bytes()
    assert download.headers["content-type"] == "application/octet-stream"


def test_non_ply_upload_is_rejected(client, ss):
    resp = client.post("/upload_model", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert resp.status_code == 400
    assert not list(ss.MODEL_DIR.iterdir())


def test_delete_model_removes_file_and_listing(client, ss, scene):
    assert client.delete(f"/delete_model?model={scene}").json()["status"] == "ok"
    assert not (ss.MODEL_DIR / scene).exists()
    assert scene not in client.get("/info").json()["models_available"]
    assert client.delete(f"/delete_model?model={scene}").status_code == 404
    assert client.get(f"/models/{scene}").status_code == 404


def test_info_reports_gaussian_count(client, scene):
    info = client.get("/info").json()
    assert info["current_model"] == scene
    assert info["num_gaussians"] == "400"


def test_load_model_switches_current_model(client, ss, scene, tmp_path):
    write_splat_ply(ss.MODEL_DIR / "other.ply")
    assert client.get("/load_model?model=other.ply").json()["status"] == "loading"
    wait_for(lambda: client.get("/load_status").json()["state"] == "done")
    assert client.get("/info").json()["current_model"] == "other.ply"
    assert client.get("/load_model?model=missing.ply").status_code == 404


def viewer_iframe_url(html):
    start = html.index('id="viewer-iframe" src="') + len('id="viewer-iframe" src="')
    return html[start:html.index('"', start)]


def test_viewer_page_loads_model_with_flythrough_track(client, scene):
    html = client.get("/").text
    url = urlparse(viewer_iframe_url(html))
    params = parse_qs(url.query, keep_blank_values=True)
    assert url.path == "/viewer/index.html"
    assert params["content"] == [f"/models/{scene}"]
    assert params["settings"] == [f"/flythrough/settings/{scene}?duration=30"]
    assert "noanim" in params and "webgl" in params and "noui" in params
    # the settings URL the viewer will fetch is served and valid
    assert client.get(params["settings"][0]).json()["startMode"] == "animTrack"


def test_viewer_page_includes_navigation_and_feature_controls(client, scene):
    html = client.get("/").text
    for marker in ('id="model-select"', 'id="flythrough-section"', "/web/flythrough.js", "/web/rag.js",
                   "/web/features.css", "window.FLY_MIN_SECONDS = 30", 'id="seg-prompt"', 'id="extract-btn"',
                   'id="trellis-btn"'):
        assert marker in html, marker
    assert f'<option value="{scene}">' in html


def test_segmented_objects_and_extractions_get_metadata_links(client, scene):
    html = client.get("/").text
    assert '<button class="meta-link"' in html                          # 📄 on every SAM3 object
    assert "MetadataLinks.open('object', `${currentModel}|${prompt}|${m.index}`" in html
    assert "MetadataLinks.open('extraction', data.job_id" in html      # 📄 Info on every extraction


def test_frontend_modules_are_served(client):
    for asset in ("flythrough.js", "rag.js", "features.css"):
        assert client.get(f"/web/{asset}").status_code == 200


def test_viewer_src_encodes_model_names(ss):
    url = ss.viewer_src("my scan (v2).ply")
    params = parse_qs(urlparse(url).query, keep_blank_values=True)
    assert params["content"] == ["/models/my scan (v2).ply"]
    # the viewer fetches this URL as-is: model name stays percent-encoded in its path
    assert params["settings"] == ["/flythrough/settings/my%20scan%20%28v2%29.ply?duration=30"]


def test_uploaded_file_is_a_valid_splat(client, tmp_path):
    src = write_splat_ply(tmp_path / "grid.ply")
    client.post("/upload_model", files={"file": ("grid.ply", src.read_bytes())})
    out = tmp_path / "back.ply"
    out.write_bytes(client.get("/models/grid.ply").content)
    assert read_vertex_count(out) == 100
