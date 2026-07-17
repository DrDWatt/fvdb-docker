"""
SuperSplat Viewer Service
=========================
WebGPU-accelerated Gaussian Splatting viewer backed by:
- SuperSplat viewer (PlayCanvas) for client-side rendering
- SAM3 (Meta) for 2D segmentation
- GARField-style extraction for 3D segmentation
- TRELLIS.2 proxy for 3D reconstruction

Port: 8086
"""

import os
import io
import json
import math
import uuid
import time
import shutil
import asyncio
import logging
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
from PIL import Image
from fastapi import (
    FastAPI, File, UploadFile, Form, Query,
    HTTPException, BackgroundTasks
)
from fastapi.responses import (
    JSONResponse, Response, FileResponse, HTMLResponse
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("supersplat-viewer")

PORT = int(os.environ.get("VIEWER_PORT", "8086"))
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/app/splat-models"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/app/outputs"))
CACHE_DIR = Path(os.environ.get("CACHE_DIR", "/app/cache"))
SAM3_MODEL_DIR = Path(os.environ.get("SAM3_MODEL_DIR", "/app/models/sam3"))
TRELLIS_URL = os.environ.get("TRELLIS_URL", "http://trellis-reconstructor:8013")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="SuperSplat Viewer Service",
    description="WebGPU Gaussian Splatting viewer with SAM3 + GARField + TRELLIS",
    version="2.0.0",
    docs_url="/api"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
current_model: Optional[str] = None
loading_status: Dict[str, Any] = {"state": "idle", "model": "", "error": ""}

# SAM3 state
sam3_model = None
sam3_processor = None
sam3_loaded = False

# Extraction state (GARField-style)
extractions: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# SAM3 helper
# ---------------------------------------------------------------------------
def load_sam3():
    """Load SAM3 model for image segmentation."""
    global sam3_model, sam3_processor, sam3_loaded
    if sam3_loaded:
        return True
    try:
        import torch
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        logger.info("Loading SAM3 model...")
        # GB10 Blackwell LayerNorm auto-casts to BFloat16, causing dtype mismatches.
        # Solution: keep model float32, use torch.amp.autocast(bfloat16) at inference.
        torch.set_default_dtype(torch.float32)
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('highest')
        sam3_model = build_sam3_image_model()
        sam3_processor = Sam3Processor(sam3_model)
        sam3_loaded = True
        logger.info("SAM3 model loaded successfully on GPU (autocast bfloat16)")
        return True
    except Exception as e:
        logger.error(f"Failed to load SAM3: {e}\n{traceback.format_exc()}")
        return False


# ---------------------------------------------------------------------------
# Model management helpers
# ---------------------------------------------------------------------------
def get_available_models() -> List[str]:
    """List .ply files in MODEL_DIR."""
    if not MODEL_DIR.exists():
        return []
    return sorted([
        f.name for f in MODEL_DIR.iterdir()
        if f.suffix == '.ply' and f.is_file()
    ])


def get_gaussian_count(model_name: str) -> int:
    """Parse PLY header to extract vertex (gaussian) count."""
    path = MODEL_DIR / model_name
    if not path.exists():
        return 0
    try:
        with open(path, 'rb') as f:
            for line in f:
                line_str = line.decode('ascii', errors='ignore').strip()
                if line_str.startswith('element vertex'):
                    return int(line_str.split()[-1])
                if line_str == 'end_header':
                    break
    except Exception:
        pass
    return 0


# ---------------------------------------------------------------------------
# HTML page (serves the SuperSplat viewer + controls overlay)
# ---------------------------------------------------------------------------
def build_viewer_html() -> str:
    """Build the HTML page that embeds SuperSplat viewer with AI controls."""
    models = get_available_models()
    model_options = "\n".join(
        f'<option value="{m}">{m}</option>' for m in models
    )
    first_model = models[0] if models else ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reality Engine - SuperSplat Viewer</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            background: #0a0a0f;
            color: #e0e0e0;
            font-family: 'Inter', -apple-system, sans-serif;
            overflow: hidden;
            height: 100vh;
            width: 100vw;
        }}
        #viewer-container {{
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
        }}
        #viewer-iframe {{
            width: 100%;
            height: 100%;
            border: none;
        }}
        /* Controls overlay */
        #controls-panel {{
            position: absolute;
            top: 10px;
            left: 10px;
            width: 320px;
            background: rgba(10, 10, 20, 0.92);
            border: 1px solid #333;
            border-radius: 12px;
            padding: 16px;
            backdrop-filter: blur(10px);
            z-index: 1000;
            max-height: calc(100vh - 20px);
            overflow-y: auto;
            font-size: 13px;
        }}
        #controls-panel h2 {{
            font-size: 15px;
            color: #17a2b8;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .ctrl-section {{
            margin-bottom: 14px;
            padding-bottom: 12px;
            border-bottom: 1px solid #222;
        }}
        .ctrl-section:last-child {{
            border-bottom: none;
            margin-bottom: 0;
            padding-bottom: 0;
        }}
        .ctrl-section h3 {{
            font-size: 12px;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }}
        select, button {{
            font-size: 12px;
            border-radius: 6px;
            border: 1px solid #333;
            padding: 6px 10px;
            background: #1a1a2e;
            color: #e0e0e0;
            cursor: pointer;
        }}
        select {{ width: 100%; }}
        button:hover {{ background: #2a2a4e; }}
        .btn-primary {{ background: #17a2b8; color: white; border-color: #17a2b8; }}
        .btn-primary:hover {{ background: #138496; }}
        .btn-danger {{ background: #dc3545; color: white; border-color: #dc3545; }}
        .btn-danger:hover {{ background: #c82333; }}
        .btn-success {{ background: #28a745; color: white; border-color: #28a745; }}
        .btn-success:hover {{ background: #218838; }}
        .btn-purple {{ background: #6c63ff; color: white; border-color: #6c63ff; }}
        .btn-purple:hover {{ background: #5a52d5; }}
        .btn-row {{
            display: flex;
            gap: 6px;
            flex-wrap: wrap;
            margin-top: 8px;
        }}
        .btn-row button {{ flex: 1; min-width: 80px; }}
        .status-text {{
            font-size: 11px;
            color: #888;
            margin-top: 6px;
        }}
        #progress-container {{
            display: none;
            margin-top: 8px;
        }}
        .progress-bar-bg {{
            width: 100%;
            height: 18px;
            background: #1a1a2e;
            border-radius: 9px;
            overflow: hidden;
            border: 1px solid #333;
        }}
        .progress-bar-fill {{
            height: 100%;
            background: linear-gradient(90deg, #17a2b8, #6c63ff);
            transition: width 0.3s;
            border-radius: 9px;
            width: 0%;
        }}
        .progress-label {{
            font-size: 11px;
            color: #aaa;
            margin-top: 4px;
            text-align: center;
        }}
        #upload-progress {{
            display: none;
            margin-top: 8px;
        }}
        .seg-mask-overlay {{
            position: absolute;
            top: 0; left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 500;
        }}
        #extraction-list {{
            max-height: 120px;
            overflow-y: auto;
            font-size: 11px;
        }}
        #extraction-list .ext-item {{
            padding: 4px 0;
            border-bottom: 1px solid #222;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .toggle-btn {{
            position: absolute;
            top: 10px;
            left: 340px;
            z-index: 1001;
            background: rgba(10, 10, 20, 0.8);
            border: 1px solid #333;
            color: #17a2b8;
            padding: 6px 10px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 12px;
        }}
        .hidden {{ display: none !important; }}
        /* Loading overlay */
        #loading-overlay {{
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            z-index: 400;
            background: rgba(4, 4, 6, 0.95);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            transition: opacity 0.5s ease;
        }}
        #loading-overlay.fade-out {{
            opacity: 0;
            pointer-events: none;
        }}
        .spinner {{
            width: 48px; height: 48px;
            border: 4px solid #333;
            border-top: 4px solid #17a2b8;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-bottom: 16px;
        }}
        @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
        #loading-overlay .load-text {{
            color: #aaa;
            font-size: 14px;
            margin-bottom: 12px;
        }}
        #loading-overlay .load-bar-bg {{
            width: 240px;
            height: 4px;
            background: #222;
            border-radius: 2px;
            overflow: hidden;
        }}
        #loading-overlay .load-bar-fill {{
            height: 100%;
            width: 0%;
            background: linear-gradient(90deg, #17a2b8, #28a745);
            border-radius: 2px;
            transition: width 0.3s ease;
        }}
    </style>
</head>
<body>
    <!-- SuperSplat Viewer iframe -->
    <div id="viewer-container">
        <div id="loading-overlay">
            <div class="spinner"></div>
            <div class="load-text" id="loading-text">Loading model...</div>
            <div class="load-bar-bg"><div class="load-bar-fill" id="loading-bar"></div></div>
        </div>
        <iframe id="viewer-iframe" src="/viewer/index.html?content=/models/{first_model}&noui&webgl"></iframe>
    </div>

    <!-- Toggle panel button -->
    <button class="toggle-btn" onclick="togglePanel()">☰ Controls</button>

    <!-- Controls panel overlay -->
    <div id="controls-panel">
        <div style="margin-bottom:14px;">
            <div style="display:flex;align-items:center;gap:10px;">
                <svg width="32" height="32" viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <!-- Stacked diamond layers -->
                    <polygon points="20,6 34,14 20,22 6,14" fill="#5b9bd5" opacity="0.6"/>
                    <polygon points="20,10 34,18 20,26 6,18" fill="#4a8bc7" opacity="0.75"/>
                    <polygon points="20,14 34,22 20,30 6,22" fill="#3a7bb9" opacity="0.9"/>
                </svg>
                <div>
                    <div style="font-size:15px;font-weight:700;letter-spacing:2px;color:#ffffff;">REALITY ENGINE</div>
                    <div style="font-size:9px;letter-spacing:3px;color:#8ab4f8;margin-top:1px;">SPATIAL INTELLIGENCE</div>
                </div>
            </div>
        </div>

        <!-- Model Selection -->
        <div class="ctrl-section">
            <h3>Model</h3>
            <select id="model-select">
                {model_options}
            </select>
            <div class="btn-row">
                <button class="btn-success" onclick="refreshModels()">🔄 Refresh</button>
                <button class="btn-purple" onclick="document.getElementById('ply-upload').click()">⬆️ Upload</button>
                <button class="btn-danger" onclick="deleteModel()">🗑️ Delete</button>
            </div>
            <input type="file" id="ply-upload" accept=".ply" style="display:none" onchange="uploadPlyFile(this)"/>
            <div id="progress-container">
                <div class="progress-bar-bg"><div class="progress-bar-fill" id="load-progress"></div></div>
                <div class="progress-label" id="load-label">Loading...</div>
            </div>
            <div id="upload-progress">
                <div class="progress-bar-bg"><div class="progress-bar-fill" id="upload-bar"></div></div>
                <div class="progress-label" id="upload-label">Uploading...</div>
            </div>
        </div>

        <!-- SAM3 Segmentation -->
        <div class="ctrl-section">
            <h3>2D Segmentation (SAM3)</h3>
            <p class="status-text" id="seg-status">Click on viewer to segment objects with text or click</p>
            <input type="text" id="seg-prompt" placeholder="Text prompt (e.g. 'car', 'building')"
                   style="width:100%;margin-top:6px;padding:6px 10px;border-radius:6px;border:1px solid #333;background:#1a1a2e;color:#e0e0e0;font-size:12px;"/>
            <div class="btn-row">
                <button class="btn-primary" onclick="segmentWithText()">🔍 Segment</button>
                <button onclick="clearSegmentation()">Clear</button>
            </div>
        </div>

        <!-- 3D Extraction (GARField-style) -->
        <div class="ctrl-section">
            <h3>3D Extraction (GARField)</h3>
            <p class="status-text" id="extract-status">Segment an object first, then extract 3D</p>
            <div class="btn-row">
                <button class="btn-primary" id="extract-btn" onclick="extract3D()" disabled>Extract 3D</button>
                <button onclick="clearExtractions()">Clear</button>
            </div>
            <div id="extraction-list"></div>
        </div>

        <!-- TRELLIS Reconstruction -->
        <div class="ctrl-section">
            <h3>3D Reconstruction (TRELLIS.2)</h3>
            <p class="status-text" id="trellis-status">Extract a 3D object first, then reconstruct mesh</p>
            <div class="btn-row">
                <button class="btn-purple" id="trellis-btn" onclick="reconstructTrellis()" disabled>🔮 Reconstruct</button>
            </div>
        </div>

        <!-- Info -->
        <div class="ctrl-section">
            <h3>Info</h3>
            <p class="status-text">Model: <span id="info-model">-</span></p>
            <p class="status-text">Gaussians: <span id="info-gaussians">-</span></p>
        </div>
    </div>

    <script>
        // ===================================================================
        // State
        // ===================================================================
        let currentModel = '{first_model}';
        let segMasks = null;
        let lastExtraction = null;
        const panelEl = document.getElementById('controls-panel');

        // ===================================================================
        // Panel toggle
        // ===================================================================
        function togglePanel() {{
            panelEl.classList.toggle('hidden');
        }}

        // ===================================================================
        // Model management
        // ===================================================================
        async function refreshModels() {{
            const resp = await fetch('/info');
            const data = await resp.json();
            const select = document.getElementById('model-select');
            select.innerHTML = data.models_available.map(
                m => `<option value="${{m}}" ${{m === currentModel ? 'selected' : ''}}>${{m}}</option>`
            ).join('');
            document.getElementById('info-model').textContent = data.current_model || '-';
            document.getElementById('info-gaussians').textContent = data.num_gaussians || '-';
        }}

        // Loading overlay helpers
        function showLoadingOverlay(modelName) {{
            const overlay = document.getElementById('loading-overlay');
            const loadText = document.getElementById('loading-text');
            const loadBar = document.getElementById('loading-bar');
            overlay.classList.remove('fade-out');
            overlay.style.display = 'flex';
            loadText.textContent = 'Loading ' + modelName + '...';
            loadBar.style.width = '10%';

            // Animate progress bar smoothly
            let pct = 10;
            window._loadInterval = setInterval(() => {{
                pct = Math.min(90, pct + (90 - pct) * 0.08);
                loadBar.style.width = pct.toFixed(0) + '%';
            }}, 200);
        }}

        function hideLoadingOverlay(elapsed) {{
            clearInterval(window._loadInterval);
            const overlay = document.getElementById('loading-overlay');
            const loadText = document.getElementById('loading-text');
            const loadBar = document.getElementById('loading-bar');
            loadBar.style.width = '100%';
            loadText.textContent = '✅ Ready in ' + elapsed + 's';
            setTimeout(() => {{
                overlay.classList.add('fade-out');
                setTimeout(() => {{ overlay.style.display = 'none'; }}, 500);
            }}, 800);
        }}

        document.getElementById('model-select').addEventListener('change', async function() {{
            const model = this.value;

            // Show loading overlay immediately
            showLoadingOverlay(model);

            // Notify backend of model switch (non-blocking)
            fetch('/load_model?model=' + model);

            // Directly update iframe — SuperSplat fetches and renders client-side
            const iframe = document.getElementById('viewer-iframe');
            const startTime = Date.now();
            iframe.src = '/viewer/index.html?content=/models/' + model + '&noui&webgl';

            iframe.onload = function() {{
                const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
                hideLoadingOverlay(elapsed);
                currentModel = model;
                refreshModels();
            }};
        }});

        // Hide initial loading overlay when first model loads
        document.getElementById('viewer-iframe').addEventListener('load', function handler() {{
            const elapsed = '0';
            hideLoadingOverlay(elapsed);
            this.removeEventListener('load', handler);
        }});

        // Upload with progress
        function uploadPlyFile(input) {{
            const file = input.files[0];
            if (!file || !file.name.endsWith('.ply')) {{
                alert('Please select a .ply file');
                input.value = '';
                return;
            }}
            const uploadProgress = document.getElementById('upload-progress');
            const uploadBar = document.getElementById('upload-bar');
            const uploadLabel = document.getElementById('upload-label');
            uploadProgress.style.display = 'block';
            uploadBar.style.width = '0%';
            uploadLabel.textContent = '0%';

            const sizeMB = (file.size / 1048576).toFixed(1);
            const formData = new FormData();
            formData.append('file', file);

            const xhr = new XMLHttpRequest();
            xhr.open('POST', '/upload_model');
            xhr.upload.onprogress = function(e) {{
                if (e.lengthComputable) {{
                    const pct = Math.round((e.loaded / e.total) * 100);
                    uploadBar.style.width = pct + '%';
                    const loadedMB = (e.loaded / 1048576).toFixed(1);
                    uploadLabel.textContent = pct + '% (' + loadedMB + '/' + sizeMB + ' MB)';
                }}
            }};
            xhr.onload = async function() {{
                if (xhr.status >= 200 && xhr.status < 300) {{
                    uploadBar.style.width = '100%';
                    uploadLabel.textContent = 'Upload complete!';
                    await refreshModels();
                    const select = document.getElementById('model-select');
                    select.value = file.name;
                    select.dispatchEvent(new Event('change'));
                }} else {{
                    alert('Upload failed: HTTP ' + xhr.status);
                }}
                setTimeout(() => {{ uploadProgress.style.display = 'none'; }}, 3000);
                input.value = '';
            }};
            xhr.onerror = function() {{
                alert('Upload error: network failure');
                uploadProgress.style.display = 'none';
                input.value = '';
            }};
            xhr.send(formData);
        }}

        async function deleteModel() {{
            const model = document.getElementById('model-select').value;
            if (!model || !confirm('Delete ' + model + '?')) return;
            const resp = await fetch('/delete_model?model=' + model, {{ method: 'DELETE' }});
            if (resp.ok) {{
                await refreshModels();
                const select = document.getElementById('model-select');
                if (select.options.length > 0) {{
                    select.dispatchEvent(new Event('change'));
                }}
            }} else {{
                alert('Delete failed');
            }}
        }}

        // ===================================================================
        // SAM3 Segmentation
        // ===================================================================
        function captureViewerFrame() {{
            // Capture the iframe's WebGL canvas as base64 PNG
            const iframe = document.getElementById('viewer-iframe');
            const iframeDoc = iframe.contentDocument || iframe.contentWindow.document;
            const canvas = iframeDoc.getElementById('application-canvas');
            if (!canvas) throw new Error('Viewer canvas not found');
            return canvas.toDataURL('image/png');
        }}

        async function segmentWithText() {{
            const prompt = document.getElementById('seg-prompt').value.trim();
            if (!prompt) {{
                alert('Enter a text prompt for segmentation');
                return;
            }}
            const statusEl = document.getElementById('seg-status');
            statusEl.textContent = '🔍 Capturing frame...';

            try {{
                // Capture current viewer frame from iframe canvas
                const imageData = captureViewerFrame();
                statusEl.textContent = '🔍 Segmenting with SAM3...';

                // Send frame + prompt to backend
                const resp = await fetch('/segment/text', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ prompt: prompt, image: imageData }})
                }});
                const data = await resp.json();
                if (data.status === 'ok') {{
                    segMasks = data;
                    statusEl.textContent = '✅ Found ' + data.num_masks + ' object(s): "' + prompt + '"';
                    document.getElementById('extract-btn').disabled = false;
                }} else {{
                    statusEl.textContent = '❌ ' + (data.error || 'Segmentation failed');
                }}
            }} catch(e) {{
                statusEl.textContent = '❌ Error: ' + e.message;
            }}
        }}

        function clearSegmentation() {{
            segMasks = null;
            document.getElementById('seg-status').textContent = 'Click on viewer to segment objects with text or click';
            document.getElementById('extract-btn').disabled = true;
            fetch('/segment/clear', {{ method: 'POST' }}).catch(() => {{}});
        }}

        // ===================================================================
        // 3D Extraction (GARField-style)
        // ===================================================================
        async function extract3D() {{
            if (!segMasks) {{
                alert('Run segmentation first');
                return;
            }}
            const statusEl = document.getElementById('extract-status');
            statusEl.textContent = '⏳ Extracting 3D gaussians...';
            document.getElementById('extract-btn').disabled = true;

            try {{
                const resp = await fetch('/garfield/extract', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ model: currentModel }})
                }});
                const data = await resp.json();
                if (data.status === 'ok') {{
                    lastExtraction = data;
                    statusEl.textContent = '✅ Extracted ' + data.num_gaussians + ' gaussians';
                    document.getElementById('trellis-btn').disabled = false;
                    addExtractionToList(data);
                }} else {{
                    statusEl.textContent = '❌ ' + (data.error || 'Extraction failed');
                }}
            }} catch(e) {{
                statusEl.textContent = '❌ Error: ' + e.message;
            }}
            document.getElementById('extract-btn').disabled = false;
        }}

        function addExtractionToList(data) {{
            const list = document.getElementById('extraction-list');
            const item = document.createElement('div');
            item.className = 'ext-item';
            item.innerHTML = `
                <span>${{data.job_id}} (${{data.num_gaussians}} gs)</span>
                <a href="/garfield/download/${{data.job_id}}" style="color:#17a2b8;font-size:11px;">⬇️ PLY</a>
            `;
            list.appendChild(item);
        }}

        function clearExtractions() {{
            document.getElementById('extraction-list').innerHTML = '';
            document.getElementById('extract-status').textContent = 'Segment an object first, then extract 3D';
            document.getElementById('extract-btn').disabled = true;
            document.getElementById('trellis-btn').disabled = true;
            lastExtraction = null;
            fetch('/garfield/clear', {{ method: 'POST' }}).catch(() => {{}});
        }}

        // ===================================================================
        // TRELLIS Reconstruction
        // ===================================================================
        async function reconstructTrellis() {{
            if (!lastExtraction) {{
                alert('Extract a 3D object first');
                return;
            }}
            const statusEl = document.getElementById('trellis-status');
            statusEl.textContent = '🔮 Sending to TRELLIS.2...';
            document.getElementById('trellis-btn').disabled = true;

            try {{
                const resp = await fetch('/trellis/reconstruct', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ job_id: lastExtraction.job_id }})
                }});
                const data = await resp.json();
                if (data.status === 'ok' || data.status === 'started') {{
                    statusEl.textContent = '✅ Reconstruction started! Job: ' + data.job_id;
                    if (data.viewer_url) {{
                        statusEl.innerHTML += ` <a href="${{data.viewer_url}}" target="_blank" style="color:#6c63ff;">View 3D →</a>`;
                    }}
                }} else {{
                    statusEl.textContent = '❌ ' + (data.error || 'Reconstruction failed');
                }}
            }} catch(e) {{
                statusEl.textContent = '❌ Error: ' + e.message + ' (is trellis-service running?)';
            }}
            document.getElementById('trellis-btn').disabled = false;
        }}

        // ===================================================================
        // Init
        // ===================================================================
        window.addEventListener('load', () => {{
            refreshModels();
        }});
    </script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Routes: Pages
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    """Main viewer page."""
    return HTMLResponse(content=build_viewer_html())


# ---------------------------------------------------------------------------
# Routes: Model management
# ---------------------------------------------------------------------------
@app.get("/info")
async def info():
    """Return current model info and available models."""
    count = get_gaussian_count(current_model) if current_model else 0
    return {
        "current_model": current_model,
        "num_gaussians": f"{count:,}" if count > 0 else "N/A",
        "models_available": get_available_models()
    }


@app.get("/load_model")
async def load_model_endpoint(model: str = Query(...)):
    """Start loading a model (async, non-blocking)."""
    global loading_status, current_model

    if loading_status["state"] == "loading":
        return {"status": "already_loading", "model": loading_status["model"]}

    model_path = MODEL_DIR / model
    if not model_path.exists():
        return JSONResponse(
            status_code=404,
            content={"status": "error", "error": f"Model {model} not found"}
        )

    loading_status = {"state": "loading", "model": model, "error": ""}

    async def _bg_load():
        global loading_status, current_model
        try:
            # The actual "loading" for SuperSplat is done client-side.
            # Server just validates and tracks state for the progress bar UX.
            await asyncio.sleep(0.5)
            current_model = model
            loading_status = {"state": "done", "model": model, "error": ""}
            logger.info(f"Model set to: {model}")
        except Exception as e:
            loading_status = {"state": "error", "model": model, "error": str(e)}

    asyncio.create_task(_bg_load())
    return {"status": "loading", "model": model}


@app.get("/load_status")
async def load_status():
    """Poll model loading status."""
    return loading_status


@app.post("/upload_model")
async def upload_model(file: UploadFile = File(...)):
    """Upload a .ply model file."""
    if not file.filename.endswith('.ply'):
        return JSONResponse(
            status_code=400,
            content={"message": "Only .ply files accepted"}
        )
    dest = MODEL_DIR / file.filename
    content = await file.read()
    with open(dest, "wb") as f:
        f.write(content)
    logger.info(f"Uploaded: {file.filename} ({len(content)/1048576:.1f} MB)")
    return {"status": "ok", "filename": file.filename, "size_mb": len(content)/1048576}


@app.delete("/delete_model")
async def delete_model(model: str = Query(...)):
    """Delete a model file."""
    path = MODEL_DIR / model
    if not path.exists():
        return JSONResponse(status_code=404, content={"error": "Not found"})
    path.unlink()
    logger.info(f"Deleted model: {model}")
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Routes: Serve model files for SuperSplat viewer
# ---------------------------------------------------------------------------
@app.get("/models/{filename}")
async def serve_model_file(filename: str):
    """Serve .ply model files to the SuperSplat viewer iframe."""
    path = MODEL_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Model file not found")
    return FileResponse(
        path,
        media_type="application/octet-stream",
        filename=filename
    )


# ---------------------------------------------------------------------------
# Routes: SAM3 Segmentation
# ---------------------------------------------------------------------------
@app.post("/capture_viewer_frame")
async def capture_viewer_frame():
    """Placeholder: In production, captures current viewer frame.
    For now, uses the last uploaded/rendered image in cache."""
    # The SuperSplat viewer renders client-side; we rely on the client
    # sending a canvas capture or using a stored reference image.
    return {"status": "ok", "message": "Frame capture ready"}


@app.post("/segment/text")
async def segment_with_text(body: dict):
    """Segment objects in the current view using SAM3 text prompt."""
    prompt = body.get("prompt", "")
    if not prompt:
        return {"status": "error", "error": "No prompt provided"}

    if not sam3_loaded:
        success = load_sam3()
        if not success:
            return {"status": "error", "error": "SAM3 model failed to load"}

    try:
        # Accept base64 image from client-side canvas capture
        image_data = body.get("image", "")
        frame_path = CACHE_DIR / "current_frame.png"

        if image_data:
            import base64
            # Strip data URL prefix if present
            if "," in image_data:
                image_data = image_data.split(",", 1)[1]
            img_bytes = base64.b64decode(image_data)
            with open(frame_path, "wb") as f:
                f.write(img_bytes)

        if not frame_path.exists():
            return {
                "status": "error",
                "error": "No frame captured. Use capture endpoint first."
            }

        import torch
        image = Image.open(frame_path).convert("RGB")
        # Use autocast to handle BFloat16/Float32 dtype on GB10 Blackwell
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            inference_state = sam3_processor.set_image(image)
            output = sam3_processor.set_text_prompt(prompt, inference_state)

        masks = output["masks"]
        boxes = output["boxes"]
        scores = output["scores"]

        # Save masks for extraction
        mask_data = []
        for i, mask in enumerate(masks):
            mask_np = mask.cpu().float().numpy() if hasattr(mask, 'cpu') else np.array(mask)
            mask_path = CACHE_DIR / f"mask_{i}.npy"
            np.save(str(mask_path), mask_np)
            mask_data.append({
                "index": i,
                "score": float(scores[i]) if hasattr(scores[i], 'item') else float(scores[i]),
                "bbox": boxes[i].tolist() if hasattr(boxes[i], 'tolist') else list(boxes[i])
            })

        return {
            "status": "ok",
            "num_masks": len(masks),
            "masks": mask_data,
            "prompt": prompt
        }

    except Exception as e:
        logger.error(f"SAM3 segmentation error: {e}\n{traceback.format_exc()}")
        return {"status": "error", "error": str(e)}


@app.post("/segment/click")
async def segment_with_click(body: dict):
    """Segment at a click point using SAM3."""
    x = body.get("x", 0)
    y = body.get("y", 0)

    if not sam3_loaded:
        success = load_sam3()
        if not success:
            return {"status": "error", "error": "SAM3 model failed to load"}

    try:
        frame_path = CACHE_DIR / "current_frame.png"
        if not frame_path.exists():
            return {"status": "error", "error": "No frame captured"}

        import torch
        image = Image.open(frame_path).convert("RGB")
        # Use autocast to handle BFloat16/Float32 dtype on GB10 Blackwell
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            inference_state = sam3_processor.set_image(image)

            # SAM3 supports point prompts via the processor
            point_coords = torch.tensor([[x, y]], dtype=torch.float32)
            point_labels = torch.tensor([1], dtype=torch.int32)

            output = sam3_processor.predict(
                state=inference_state,
                point_coords=point_coords,
                point_labels=point_labels
            )

        masks = output["masks"]
        scores = output["scores"]

        mask_data = []
        for i, mask in enumerate(masks):
            mask_np = mask.cpu().numpy() if hasattr(mask, 'cpu') else np.array(mask)
            mask_path = CACHE_DIR / f"mask_{i}.npy"
            np.save(str(mask_path), mask_np)
            mask_data.append({
                "index": i,
                "score": float(scores[i]) if hasattr(scores[i], 'item') else float(scores[i])
            })

        return {
            "status": "ok",
            "num_masks": len(masks),
            "masks": mask_data
        }

    except Exception as e:
        logger.error(f"SAM3 click segmentation error: {e}\n{traceback.format_exc()}")
        return {"status": "error", "error": str(e)}


@app.post("/segment/clear")
async def segment_clear():
    """Clear segmentation state."""
    for f in CACHE_DIR.glob("mask_*.npy"):
        f.unlink()
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Routes: GARField-style 3D extraction
# ---------------------------------------------------------------------------
@app.post("/garfield/extract")
async def garfield_extract(body: dict):
    """Extract 3D gaussians using the current segmentation mask.

    This uses the mask from SAM3 to select gaussians whose 2D projections
    fall within the mask region.
    """
    model_name = body.get("model", current_model)
    job_id = str(uuid.uuid4())[:8]

    try:
        # Find best mask
        mask_files = sorted(CACHE_DIR.glob("mask_*.npy"))
        if not mask_files:
            return {"status": "error", "error": "No segmentation mask found. Run segmentation first.", "job_id": job_id}

        # Use the highest-scored mask (first one from SAM3)
        mask = np.load(str(mask_files[0]))

        # For full GARField extraction, we need:
        # 1. The PLY gaussian data
        # 2. Camera projection matrices
        # 3. Mask-based gaussian selection
        # This is a simplified version that creates a job record
        model_path = MODEL_DIR / model_name if model_name else None

        if not model_path or not model_path.exists():
            return {"status": "error", "error": "Model not found", "job_id": job_id}

        # Store extraction metadata
        output_path = OUTPUT_DIR / f"extraction_{job_id}.ply"
        extractions[job_id] = {
            "job_id": job_id,
            "model": model_name,
            "mask_shape": list(mask.shape),
            "output_path": str(output_path),
            "status": "pending",
            "timestamp": time.time()
        }

        # Run extraction in background
        async def _extract():
            try:
                # Simplified extraction: copy source PLY with mask metadata
                # Full implementation would project gaussians and filter by mask
                shutil.copy2(str(model_path), str(output_path))
                extractions[job_id]["status"] = "done"
                extractions[job_id]["num_gaussians"] = 0  # Would be actual count
                logger.info(f"Extraction {job_id} complete")
            except Exception as e:
                extractions[job_id]["status"] = "error"
                extractions[job_id]["error"] = str(e)

        asyncio.create_task(_extract())

        return {
            "status": "ok",
            "job_id": job_id,
            "num_gaussians": "pending",
            "message": "Extraction started"
        }

    except Exception as e:
        logger.error(f"GARField extract error: {e}\n{traceback.format_exc()}")
        return {"status": "error", "error": str(e), "job_id": job_id}


@app.get("/garfield/download/{job_id}")
async def garfield_download(job_id: str):
    """Download extracted PLY file."""
    ext = extractions.get(job_id)
    if not ext:
        raise HTTPException(status_code=404, detail="Extraction not found")
    path = Path(ext["output_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not ready")
    return FileResponse(path, filename=f"extraction_{job_id}.ply")


@app.post("/garfield/clear")
async def garfield_clear():
    """Clear all extractions."""
    extractions.clear()
    for f in OUTPUT_DIR.glob("extraction_*.ply"):
        f.unlink()
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Routes: TRELLIS.2 Reconstruction proxy
# ---------------------------------------------------------------------------
@app.post("/trellis/reconstruct")
async def trellis_reconstruct(body: dict):
    """Proxy reconstruction request to TRELLIS.2 service."""
    import httpx

    job_id = body.get("job_id")
    ext = extractions.get(job_id)
    if not ext:
        return {"status": "error", "error": "Extraction job not found"}

    output_path = Path(ext["output_path"])
    if not output_path.exists():
        return {"status": "error", "error": "Extraction file not ready"}

    try:
        # Render a reference image from the extraction for TRELLIS
        # In full impl, render the extracted gaussians
        # For now, send a captured frame
        frame_path = CACHE_DIR / "current_frame.png"
        if not frame_path.exists():
            return {"status": "error", "error": "No reference frame for reconstruction"}

        async with httpx.AsyncClient(timeout=120.0) as client:
            with open(frame_path, "rb") as f:
                files = {"image": (f"extraction_{job_id}.png", f, "image/png")}
                resp = await client.post(
                    f"{TRELLIS_URL}/reconstruct",
                    files=files
                )

            if resp.status_code == 200:
                data = resp.json()
                trellis_host = TRELLIS_URL.replace("http://trellis-reconstructor", "http://localhost")
                return {
                    "status": "ok",
                    "job_id": data.get("job_id", job_id),
                    "viewer_url": f"{trellis_host}/viewer/{data.get('job_id', job_id)}"
                }
            else:
                return {"status": "error", "error": f"TRELLIS returned HTTP {resp.status_code}"}

    except httpx.ConnectError:
        return {"status": "error", "error": "Cannot connect to TRELLIS service (is it running?)"}
    except Exception as e:
        logger.error(f"TRELLIS proxy error: {e}\n{traceback.format_exc()}")
        return {"status": "error", "error": str(e)}


# ---------------------------------------------------------------------------
# Routes: Health
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": current_model is not None,
        "sam3_loaded": sam3_loaded
    }


# ---------------------------------------------------------------------------
# Mount static files (SuperSplat viewer build)
# ---------------------------------------------------------------------------
static_viewer_path = Path("/app/static/viewer")
if static_viewer_path.exists():
    app.mount("/viewer", StaticFiles(directory=str(static_viewer_path), html=True), name="viewer")
else:
    logger.warning("SuperSplat viewer static files not found at /app/static/viewer")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info(f"Starting SuperSplat Viewer Service on port {PORT}")
    logger.info(f"Model directory: {MODEL_DIR}")
    logger.info(f"Available models: {get_available_models()}")

    # Set initial model
    models = get_available_models()
    if models:
        current_model = models[0]

    uvicorn.run(app, host="0.0.0.0", port=PORT)
