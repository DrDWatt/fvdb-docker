""":8086 metadata linking (typed + uploaded), RAG data upload and RAG chat."""
from support.fakes import sse_tokens

OBJ = "/metadata/object/scene.ply%7Ccar%7C0"
EXT = "/metadata/extraction/ab12cd34"


def save(client, url, label, text, model="scene.ply"):
    return client.post(url, data={"label": label, "text": text, "model": model})


def upload(client, url, *files, model="scene.ply"):
    return client.post(f"{url}/upload", data={"model": model},
                       files=[("files", (name, data, ctype)) for name, data, ctype in files])


# ----- typed metadata -----
def test_typed_metadata_is_linked_to_an_object(client):
    assert client.get(OBJ).json()["summary"] is None
    assert save(client, OBJ, "Tesla", "Red Model S at a Supercharger").json()["status"] == "ok"
    summary = client.get(OBJ).json()["summary"]
    assert (summary["label"], summary["text"], summary["model"]) == ("Tesla", "Red Model S at a Supercharger",
                                                                    "scene.ply")
    save(client, OBJ, "Tesla Model S", "Updated notes")
    assert client.get(OBJ).json()["summary"]["label"] == "Tesla Model S"


def test_typed_metadata_is_linked_to_an_extraction(client):
    save(client, EXT, "Charger", "V3 stall 4")
    assert client.get(EXT).json()["summary"]["text"] == "V3 stall 4"


def test_unknown_metadata_kind_is_rejected(client):
    assert save(client, "/metadata/camera/x", "a", "b").status_code == 400


# ----- uploaded metadata documents -----
def test_documents_upload_and_are_served_back(client):
    resp = upload(client, OBJ, ("spec.txt", b"250kW V3 Supercharger", "text/plain"),
                  ("photo.jpg", b"\xff\xd8\xff", "image/jpeg")).json()
    assert [f["name"] for f in resp["uploaded"]] == ["spec.txt", "photo.jpg"]
    assert resp["summary"]["training_data"]["files_count"] == 2
    doc = client.get(f"{OBJ}/file/0")
    assert doc.content == b"250kW V3 Supercharger" and doc.headers["content-type"].startswith("text/plain")
    assert 'filename="spec.txt"' in doc.headers["content-disposition"]
    assert client.get(f"{OBJ}/file/9").status_code == 404


def test_uploads_accumulate_and_keep_typed_fields(client):
    save(client, EXT, "Charger", "notes")
    upload(client, EXT, ("a.md", b"# A", "text/markdown"))
    summary = upload(client, EXT, ("b.md", b"# B", "text/markdown")).json()["summary"]
    assert [f["name"] for f in summary["files"]] == ["a.md", "b.md"] and summary["label"] == "Charger"


def test_metadata_list_is_filtered_by_model(client):
    save(client, OBJ, "Tesla", "x")
    save(client, "/metadata/object/other.ply%7Ctree%7C0", "Tree", "y", model="other.ply")
    labels = {i["label"] for i in client.get("/metadata?model=scene.ply").json()["items"]}
    assert labels == {"Tesla"}
    assert len(client.get("/metadata").json()["items"]) == 2


# ----- RAG data upload (model-level documents, rendering service) -----
def test_model_info_upload_is_stored_and_retrievable(client, services):
    resp = client.post("/upload_summary?model=scene.ply",
                       files={"file": ("site.md", b"Tesla Supercharger site, 8 stalls", "text/markdown")})
    assert resp.status_code == 200 and resp.json()["message"] == "Summary uploaded successfully"
    assert services.summaries["scene.ply"] == "Tesla Supercharger site, 8 stalls"
    summary = client.get("/model_summary?model=scene.ply").json()
    assert summary["has_summary"] and "8 stalls" in summary["summary"]


def test_model_summary_when_rendering_service_is_unreachable(client, monkeypatch):
    import httpx

    def down(*a, **k):
        raise httpx.ConnectError("down")
    monkeypatch.setattr(httpx, "get", down)
    assert client.get("/model_summary?model=scene.ply").json() == {
        "model": "scene.ply", "has_summary": False, "summary": None}


# ----- RAG context + chat -----
def test_rag_context_counts_linked_data(client, ss, scene, services, sam3):
    ss.last_segmentation.update(prompt="car", model=scene, masks=[{"index": 0, "score": 0.9}])
    ss.extractions["j1"] = {"job_id": "j1", "model": scene, "status": "done", "num_gaussians": 1200}
    save(client, OBJ, "Tesla", "Red car")
    upload(client, OBJ, ("spec.txt", b"250kW", "text/plain"))
    services.summaries[scene] = "Supercharger site"
    ctx = client.get(f"/rag/context?model={scene}").json()
    assert ctx["segments_count"] == 1 and ctx["extractions_count"] == 1 and ctx["documents_count"] == 1
    assert ctx["segment_labels"] == ["Tesla"] and ctx["model_summary"] == "Supercharger site"


def test_rag_status_reflects_ollama(client, services):
    assert client.get("/rag/status").json()["model"] == "nemotron-mini:latest"
    services.ollama_up = False
    assert client.get("/rag/status").json()["available"] is False


def test_rag_chat_rejects_empty_query(client, services):
    assert client.post("/rag/query", json={"query": "  "}).status_code == 400


def test_rag_chat_streams_answer_grounded_on_linked_metadata(client, ss, scene, services):
    ss.last_segmentation.update(prompt="car", model=scene, masks=[{"index": 0, "score": 0.93}])
    ss.extractions["j1"] = {"job_id": "j1", "model": scene, "status": "done", "num_gaussians": 1200}
    save(client, OBJ, "Tesla", "Red Model S")
    upload(client, OBJ, ("spec.txt", b"Charger stall 4, 250kW", "text/plain"))
    save(client, "/metadata/object/other.ply%7Ctree%7C0", "Oak tree", "other scene", model="other.ply")
    services.summaries[scene] = "Supercharger site with 8 stalls"
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]

    resp = client.post("/rag/query", json={"query": "What car is at stall 4?", "model": scene, "history": history})
    assert resp.headers["content-type"].startswith("text/event-stream")
    text, done, errors = sse_tokens(resp.text)
    assert (text, done, errors) == ("The scene shows a red car.", True, [])

    prompt = services.last_system_prompt()
    for expected in ("Model: scene.ply, 400 gaussians", "SAM3 segmentation for 'car': 1 object(s)",
                     "GARField 3D extraction j1: 1200 gaussians", "SAM3 object 'Tesla': Red Model S",
                     "Document 'spec.txt': Charger stall 4, 250kW", "Supercharger site with 8 stalls"):
        assert expected in prompt, expected
    assert "Oak tree" not in prompt                      # other model's metadata is not leaked
    messages = services.chats[-1]["messages"]
    assert messages[1:3] == history and messages[-1] == {"role": "user", "content": "What car is at stall 4?"}


def test_rag_chat_without_context_gives_guidance(client, services):
    client.post("/rag/query", json={"query": "what is here?"})
    assert "Segment objects with SAM3" in services.last_system_prompt()
