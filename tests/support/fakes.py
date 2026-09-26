"""In-process fakes for the viewers' external dependencies.

FakeServices replaces the HTTP calls the viewers make (rendering-service model
summaries, Ollama, TRELLIS.2) for both httpx (:8086, shared RAG helpers) and
requests (:8085). Fake torch / SAM2 / SAM3 objects stand in for the GPU models so
the viewer logic around them can be tested without a GPU.
"""
import contextlib
import json
import sys
import types
from urllib.parse import urlparse

import numpy as np


class FakeResponse:
    def __init__(self, status_code=200, payload=None, lines=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self._lines = lines or []
        self.text = json.dumps(self._payload)
        self.content = self.text.encode()

    def json(self):
        return self._payload

    def read(self):
        return self.content

    def iter_lines(self):
        yield from self._lines


class FakeServices:
    """Routes HTTP calls by path: /summary/<model>, /api/tags, /api/chat, /reconstruct."""

    def __init__(self, reply="The scene shows a red car.", ollama_up=True, trellis_up=True):
        self.summaries = {}
        self.chats = []          # every Ollama chat payload (messages incl. system prompt)
        self.trellis_jobs = []
        self.reply = reply
        self.ollama_up = ollama_up
        self.trellis_up = trellis_up

    # ----- routing -----
    def _get(self, url, **_):
        path = urlparse(url).path
        if path == "/api/tags":
            if not self.ollama_up:
                raise ConnectionError("ollama down")
            return FakeResponse(payload={"models": [{"name": "nemotron-mini:latest"}]})
        if path.startswith("/summary/"):
            model = path.split("/summary/", 1)[1]
            text = self.summaries.get(model)
            return FakeResponse(payload={"model": model, "has_summary": text is not None, "summary": text})
        return FakeResponse(404, {"detail": "not found"})

    def _post(self, url, files=None, **_):
        path = urlparse(url).path
        if path.startswith("/summary/"):
            model = path.split("/summary/", 1)[1]
            name, content, _ctype = files["file"][:3]
            data = content.read() if hasattr(content, "read") else content
            self.summaries[model] = data.decode("utf-8", errors="replace")
            return FakeResponse(payload={"model": model, "filename": name, "message": "Summary uploaded successfully"})
        if path == "/reconstruct":
            if not self.trellis_up:
                import httpx
                raise httpx.ConnectError("trellis down")
            name, fh = files["image"][:2]
            self.trellis_jobs.append({"name": name, "bytes": len(fh.read())})
            return FakeResponse(payload={"job_id": f"trellis{len(self.trellis_jobs)}", "status": "queued"})
        return FakeResponse(404, {"detail": "not found"})

    @contextlib.contextmanager
    def _stream(self, method, url, **kwargs):
        self.chats.append(kwargs.get("json"))
        words = self.reply.split(" ")
        lines = [json.dumps({"message": {"content": w + (" " if i < len(words) - 1 else "")}, "done": False})
                 for i, w in enumerate(words)]
        lines.append(json.dumps({"message": {"content": ""}, "done": True}))
        yield FakeResponse(lines=lines)

    # ----- installation -----
    def install(self, monkeypatch):
        import httpx
        import requests

        services = self

        class AsyncClient:
            def __init__(self, *a, **kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def post(self, url, files=None, **kw):
                return services._post(url, files=files, **kw)

            async def get(self, url, **kw):
                return services._get(url, **kw)

        monkeypatch.setattr(httpx, "get", self._get)
        monkeypatch.setattr(httpx, "post", self._post)
        monkeypatch.setattr(httpx, "stream", self._stream)
        monkeypatch.setattr(httpx, "AsyncClient", AsyncClient)
        monkeypatch.setattr(requests, "get", self._get)
        monkeypatch.setattr(requests, "post", self._post)
        return self

    def last_system_prompt(self):
        return self.chats[-1]["messages"][0]["content"]


def sse_tokens(body: str):
    """Parse an SSE body into (text, done, errors)."""
    text, done, errors = "", False, []
    for line in body.splitlines():
        if line.startswith("data: "):
            msg = json.loads(line[6:])
            text += msg.get("token", "")
            done = done or msg.get("done", False)
            if msg.get("error"):
                errors.append(msg["error"])
    return text, done, errors


def install_fake_torch(monkeypatch):
    """Minimal torch surface used around SAM2/SAM3 calls (only when torch is absent)."""
    try:
        import torch  # noqa: F401
        return
    except ImportError:
        pass
    torch = types.ModuleType("torch")
    torch.bfloat16, torch.float32 = "bfloat16", "float32"
    torch.amp = types.SimpleNamespace(autocast=lambda *a, **k: contextlib.nullcontext())
    torch.cuda = types.SimpleNamespace(empty_cache=lambda: None, is_available=lambda: False)
    torch.set_default_dtype = lambda *_: None
    monkeypatch.setitem(sys.modules, "torch", torch)


def box_mask(h, w, top, left, bottom, right):
    m = np.zeros((h, w), dtype=np.float32)
    m[top:bottom, left:right] = 1.0
    return m


class FakeSam3Processor:
    """SAM3 text-prompt processor returning fixed fractional boxes as masks."""

    def __init__(self, boxes=((0.25, 0.1, 0.75, 0.5), (0.3, 0.6, 0.7, 0.9)), scores=(0.97, 0.81)):
        self.boxes, self.scores, self.prompts = boxes, scores, []

    def set_image(self, image):
        return {"size": image.size}

    def set_text_prompt(self, prompt, state):
        self.prompts.append(prompt)
        w, h = state["size"]
        masks, boxes = [], []
        for top, left, bottom, right in self.boxes:
            t, l, b, r = int(top * h), int(left * w), int(bottom * h), int(right * w)
            masks.append(box_mask(h, w, t, l, b, r)[None])
            boxes.append(np.array([l, t, r, b], dtype=np.float32))
        return {"masks": masks, "boxes": boxes, "scores": np.array(self.scores)}


class FakeSam2Predictor:
    """SAM2 image predictor: point prompts select a box around the click."""

    model = object()

    def set_image(self, image):
        self.shape = image.shape[:2]

    def predict(self, point_coords, point_labels, multimask_output=True):
        h, w = self.shape
        x, y = point_coords[0]
        masks = np.stack([box_mask(h, w, max(0, y - s), max(0, x - s), y + s, x + s) > 0 for s in (10, 40, 80)])
        return masks, np.array([0.5, 0.95, 0.7]), None


def install_fake_sam2(monkeypatch, auto_masks):
    """Automatic mask generator returning `auto_masks` (list of HxW bool arrays)."""
    class Generator:
        def __init__(self, model, **kwargs):
            self.kwargs = kwargs

        def generate(self, image):
            return [{"segmentation": m, "area": int(m.sum()), "stability_score": 0.9 - 0.01 * i}
                    for i, m in enumerate(auto_masks)]

    sam2 = types.ModuleType("sam2")
    amg = types.ModuleType("sam2.automatic_mask_generator")
    amg.SAM2AutomaticMaskGenerator = Generator
    monkeypatch.setitem(sys.modules, "sam2", sam2)
    monkeypatch.setitem(sys.modules, "sam2.automatic_mask_generator", amg)
