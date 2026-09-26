"""RAG helpers shared by both viewers: document text extraction, prompt, Ollama chat."""
from support.fakes import FakeServices, sse_tokens
from viewer_common.rag import build_system_prompt, extract_file_text, ollama_status, ollama_stream_sse

OLLAMA = "http://ollama:11434"


def test_text_documents_are_extracted_and_truncated():
    doc = {"name": "notes.md", "content_type": "text/markdown", "data": ("x" * 5000).encode()}
    assert extract_file_text(doc) == "x" * 2000
    assert extract_file_text({"name": "a.json", "content_type": "application/json", "data": b'{"k": 1}'}) == '{"k": 1}'


def test_text_is_detected_by_extension_when_content_type_is_generic():
    doc = {"name": "specs.txt", "content_type": "application/octet-stream", "data": b"250kW charger"}
    assert extract_file_text(doc) == "250kW charger"


def test_binary_documents_are_described_not_dumped():
    doc = {"name": "photo.jpg", "content_type": "image/jpeg", "data": b"\xff\xd8", "size": "1.0 KB"}
    assert extract_file_text(doc) == "[Uploaded file: photo.jpg, type: image/jpeg, size: 1.0 KB]"


def test_pdf_without_text_layer_falls_back_to_description():
    doc = {"name": "manual.pdf", "content_type": "application/pdf", "data": b"%PDF-1.4 broken", "size": "2 KB"}
    assert extract_file_text(doc).startswith("[PDF file: manual.pdf")


def test_system_prompt_wraps_context_and_hint():
    prompt = build_system_prompt("Model: car.ply", "SAM3 segmentation results", "No data yet")
    assert "SAM3 segmentation results" in prompt
    assert "=== Scene Context ===\nModel: car.ply\n=== End Context ===" in prompt
    assert "No data yet" in build_system_prompt("", "x", "No data yet")


def test_ollama_status_reports_matching_model(monkeypatch):
    FakeServices().install(monkeypatch)
    status = ollama_status(OLLAMA, "nemotron-mini")
    assert status["available"] and status["model"] == "nemotron-mini:latest"
    assert not ollama_status(OLLAMA, "llama3")["available"]


def test_ollama_status_when_unreachable(monkeypatch):
    FakeServices(ollama_up=False).install(monkeypatch)
    status = ollama_status(OLLAMA, "nemotron-mini")
    assert status["available"] is False and "down" in status["error"]


def test_chat_streams_tokens_as_sse_with_resolved_model(monkeypatch):
    services = FakeServices(reply="Two cars are parked").install(monkeypatch)
    messages = [{"role": "system", "content": "ctx"}, {"role": "user", "content": "what?"}]
    text, done, errors = sse_tokens("".join(ollama_stream_sse(OLLAMA, "nemotron-mini", messages)))
    assert (text, done, errors) == ("Two cars are parked", True, [])
    assert services.chats[0]["model"] == "nemotron-mini:latest"
    assert services.chats[0]["messages"] == messages and services.chats[0]["stream"] is True
