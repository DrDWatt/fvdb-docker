"""RAG helpers shared by the splat viewers: document text extraction and Ollama chat.

Uses httpx (available in both viewer containers) for the LLM calls.
"""
import json

import httpx

SYSTEM_PROMPT_HEADER = (
    "You are an AI assistant for a 3D Gaussian Splat viewer application. "
    "You help users understand what is in their 3D scene based on available data including: "
    "{sources}, uploaded documents/data, and scene metadata. Answer concisely and accurately "
    "based on the available context. If you don't have enough information, say so.\n\n"
)


def extract_file_text(file_info: dict, max_chars: int = 2000) -> str:
    """Extract readable text from an uploaded file ({name, content_type, data, size})."""
    ct = file_info.get("content_type", "") or ""
    data = file_info.get("data", b"")
    name = file_info.get("name", "file")

    if "text" in ct or ct in ("application/json", "application/csv") or \
            name.lower().endswith((".txt", ".md", ".json", ".csv")):
        try:
            return data.decode("utf-8", errors="replace")[:max_chars]
        except Exception:
            return ""

    if ct == "application/pdf" or name.lower().endswith(".pdf"):
        try:
            import fitz  # PyMuPDF (optional)
            doc = fitz.open(stream=data, filetype="pdf")
            text = "\n".join(page.get_text() for page in doc[:10])
            doc.close()
            return text[:max_chars]
        except Exception:
            return f"[PDF file: {name}, {file_info.get('size', 'unknown size')}]"

    return f"[Uploaded file: {name}, type: {ct}, size: {file_info.get('size', 'unknown')}]"


def build_system_prompt(context: str, sources: str, empty_hint: str) -> str:
    """System prompt wrapping the scene context for the LLM."""
    return (
        SYSTEM_PROMPT_HEADER.format(sources=sources)
        + "=== Scene Context ===\n"
        + (context if context else empty_hint)
        + "\n=== End Context ==="
    )


def ollama_status(ollama_url: str, model: str) -> dict:
    """Check that Ollama is reachable and the configured model (prefix match) is pulled."""
    try:
        resp = httpx.get(f"{ollama_url}/api/tags", timeout=5)
        if resp.status_code != 200:
            return {"available": False, "error": f"Ollama returned {resp.status_code}"}
        models = [m["name"] for m in resp.json().get("models", [])]
        matched = [m for m in models if model in m]
        if matched:
            return {"available": True, "model": matched[0], "all_models": models}
        return {"available": False, "error": f"Model '{model}' not found. Available: {models}",
                "all_models": models}
    except Exception as e:
        return {"available": False, "error": str(e)}


def ollama_stream_sse(ollama_url: str, model: str, messages: list):
    """Yield Server-Sent-Event lines ({token} / {done} / {error}) from an Ollama chat."""
    status = ollama_status(ollama_url, model)
    actual_model = status.get("model", model)
    try:
        with httpx.stream("POST", f"{ollama_url}/api/chat",
                          json={"model": actual_model, "messages": messages, "stream": True},
                          timeout=httpx.Timeout(120, connect=10)) as resp:
            if resp.status_code != 200:
                body = resp.read().decode(errors="replace")[:200]
                yield f"data: {json.dumps({'error': f'Ollama error {resp.status_code}: {body}'})}\n\n"
                return
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                token = chunk.get("message", {}).get("content", "")
                if token:
                    yield f"data: {json.dumps({'token': token})}\n\n"
                if chunk.get("done"):
                    yield f"data: {json.dumps({'done': True})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
