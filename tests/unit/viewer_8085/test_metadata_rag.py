""":8085 segmented metadata (typed + uploaded), RAG data upload and RAG chat."""
from support.fakes import sse_tokens


def save_object(client, idx, label, text):
    return client.post("/object_summary", data={"segment_idx": idx, "label": label, "text": text})


def upload_object(client, idx, *files):
    return client.post("/object_summary/upload", data={"segment_idx": idx},
                       files=[("files", (n, d, c)) for n, d, c in files])


# ----- per-object metadata -----
def test_typed_object_metadata_and_label(client):
    assert client.get("/object_summary/0").json()["summary"] is None
    assert save_object(client, 0, "Tesla", "Red Model S").json()["text_saved"] is True
    data = client.get("/object_summary/0").json()
    assert data["label"] == "Tesla" and data["summary"]["text"] == "Red Model S"
    assert client.get("/segment/labels").json()["labels"]["0"] == "Tesla"        # label propagates to the segment


def test_object_documents_upload_and_download(client):
    resp = upload_object(client, 1, ("spec.txt", b"250kW charger", "text/plain"),
                         ("pic.png", b"\x89PNG", "image/png")).json()
    assert resp["training_data"]["files_count"] == 2 and resp["training_data"]["status"] == "documents_uploaded"
    files = client.get("/object_summary/1").json()["summary"]["files"]
    assert [(f["idx"], f["name"]) for f in files] == [(0, "spec.txt"), (1, "pic.png")]
    doc = client.get("/object_summary/1/file/0")
    assert doc.content == b"250kW charger" and doc.headers["content-type"].startswith("text/plain")
    assert client.get("/object_summary/1/file/5").status_code == 404
    assert client.get("/object_summary/9/file/0").status_code == 404


def test_object_summaries_overview(client):
    save_object(client, 0, "Tesla", "car")
    upload_object(client, 0, ("a.txt", b"x", "text/plain"))
    overview = client.get("/object_summaries").json()
    assert overview["count"] == 1
    assert overview["summaries"]["0"] == {"label": "Tesla", "text": "car", "files_count": 1, "has_training": True}


# ----- per-extraction metadata -----
def test_extraction_metadata_typed_and_uploaded(client):
    client.post("/extraction_summary", data={"job_id": "ab12", "label": "Charger", "text": "Stall 4"})
    client.post("/extraction_summary/upload", data={"job_id": "ab12"},
                files=[("files", ("manual.md", b"# Charger manual", "text/markdown"))])
    summary = client.get("/extraction_summary/ab12").json()["summary"]
    assert (summary["label"], summary["text"], summary["files"][0]["name"]) == ("Charger", "Stall 4", "manual.md")
    assert client.get("/extraction_summary/ab12/file/0").content == b"# Charger manual"
    assert client.get("/extraction_summaries").json()["summaries"]["ab12"]["files_count"] == 1


# ----- RAG data upload (model documents via the rendering service) -----
def test_model_info_upload_and_summary(client, services):
    resp = client.post("/upload_summary?model=scene.ply",
                       files={"file": ("site.txt", b"Supercharger site with 8 stalls", "text/plain")}).json()
    assert resp["message"] == "Summary uploaded successfully"
    summary = client.get("/model_summary?model=scene.ply").json()
    assert summary["has_summary"] and summary["summary"] == "Supercharger site with 8 stalls"


# ----- RAG chat -----
def test_rag_status(client, services):
    assert client.get("/rag/status").json()["available"] is True
    services.ollama_up = False
    assert client.get("/rag/status").json()["available"] is False


def test_rag_context_summarizes_linked_data(client, ivs, services, loaded_model):
    ivs.current_segments = {"masks": [None, None], "num_segments": 2}
    save_object(client, 0, "Tesla", "Red Model S")
    upload_object(client, 0, ("spec.txt", b"250kW", "text/plain"))
    client.post("/extraction_summary", data={"job_id": "ab12", "label": "Charger", "text": "Stall 4"})
    services.summaries["scene.ply"] = "Supercharger site"
    ctx = client.get("/rag/context?model=scene.ply").json()
    assert ctx["segments_count"] == 2 and ctx["extractions_count"] == 1 and ctx["documents_count"] == 1
    assert ctx["segment_labels"][0] == "Tesla" and ctx["model_summary"] == "Supercharger site"


def test_rag_chat_streams_an_answer_grounded_on_metadata(client, ivs, services, loaded_model):
    ivs.current_segments = {"masks": [None], "num_segments": 1}
    save_object(client, 0, "Tesla", "Red Model S")
    upload_object(client, 0, ("spec.txt", b"Charger stall 4, 250kW", "text/plain"))
    client.post("/extraction_summary", data={"job_id": "ab12", "label": "Charger", "text": "V3 unit"})
    services.summaries["scene.ply"] = "Supercharger site with 8 stalls"
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]

    resp = client.post("/rag/query", json={"query": "Which stall?", "model": "scene.ply", "history": history})
    text, done, errors = sse_tokens(resp.text)
    assert (text, done, errors) == ("The scene shows a red car.", True, [])
    prompt = services.last_system_prompt()
    for expected in ("Model: scene.ply, 100 gaussians", "SAM-2 segmentation: 1 object(s)",
                     "SAM-2 Object 'Tesla': Red Model S", "Document 'spec.txt': Charger stall 4, 250kW",
                     "3D Extraction 'Charger': V3 unit", "Supercharger site with 8 stalls"):
        assert expected in prompt, expected
    assert services.chats[-1]["messages"][1:3] == history


def test_rag_chat_validation_and_guidance(client, services):
    assert client.post("/rag/query", json={"query": ""}).status_code == 400
    client.post("/rag/query", json={"query": "what is here?"})
    assert "Run SAM-2 Auto Segment" in services.last_system_prompt()
