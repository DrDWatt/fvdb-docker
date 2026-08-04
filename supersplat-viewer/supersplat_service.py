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
import gc
import json
import math
import uuid
import time
import shutil
import asyncio
import logging
import threading
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, List

# Reduce CUDA allocator fragmentation on unified-memory GPUs (GB10).
# Must be set before torch is imported anywhere in this process.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

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
sam3_last_used = 0.0

# Memory policy: unload SAM3 after this many idle minutes so the GB10's
# unified memory is available for training and other services.
SAM3_IDLE_UNLOAD_SECONDS = float(
    os.environ.get("SAM3_IDLE_UNLOAD_MINUTES", "10")
) * 60

_sam3_lock = threading.Lock()          # prevents concurrent double-load
_sam3_semaphore = asyncio.Semaphore(1)  # one GPU inference at a time

# Extraction state (GARField-style)
extractions: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# SAM3 helper
# ---------------------------------------------------------------------------
def load_sam3():
    """Load SAM3 model for image segmentation (thread-safe, lazy)."""
    global sam3_model, sam3_processor, sam3_loaded, sam3_last_used
    with _sam3_lock:
        if sam3_loaded:
            sam3_last_used = time.time()
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
            sam3_last_used = time.time()
            logger.info("SAM3 model loaded successfully on GPU (autocast bfloat16)")
            return True
        except Exception as e:
            logger.error(f"Failed to load SAM3: {e}\n{traceback.format_exc()}")
            return False


def unload_sam3():
    """Release SAM3 and free GPU memory (idle-unload policy)."""
    global sam3_model, sam3_processor, sam3_loaded
    with _sam3_lock:
        if not sam3_loaded:
            return False
        logger.info("Unloading SAM3 (idle) to free GPU memory")
        sam3_model = None
        sam3_processor = None
        sam3_loaded = False
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        return True


def _release_sam3_cache():
    """Refresh idle timer and drop cached CUDA blocks after an inference."""
    global sam3_last_used
    sam3_last_used = time.time()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


async def _sam3_idle_unload_loop():
    """Unload SAM3 after SAM3_IDLE_UNLOAD_SECONDS without a request."""
    while True:
        await asyncio.sleep(60)
        try:
            if not sam3_loaded:
                continue
            if _sam3_semaphore.locked():
                continue  # never unload mid-inference
            if time.time() - sam3_last_used > SAM3_IDLE_UNLOAD_SECONDS:
                await asyncio.to_thread(unload_sam3)
        except Exception as e:
            logger.error(f"[idle-unload] {e}")


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
            opacity: 0.7;
            transition: opacity 0.3s ease;
        }}
        /* Popup modal for 3D viewer */
        .popup-overlay {{
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            z-index: 9000;
            background: rgba(0,0,0,0.75);
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .popup-overlay.hidden {{ display: none; }}
        .popup-container {{
            position: relative;
            width: 80vw;
            height: 80vh;
            max-width: 1200px;
            max-height: 800px;
            background: #0a0a14;
            border: 1px solid #333;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 20px 60px rgba(0,0,0,0.6);
        }}
        .popup-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 16px;
            background: #111;
            border-bottom: 1px solid #333;
        }}
        .popup-header h3 {{
            margin: 0;
            font-size: 14px;
            color: #17a2b8;
        }}
        .popup-close {{
            background: #dc3545;
            border: none;
            color: white;
            width: 28px;
            height: 28px;
            border-radius: 50%;
            cursor: pointer;
            font-size: 14px;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .popup-close:hover {{ background: #c82333; }}
        .popup-body {{
            width: 100%;
            height: calc(100% - 44px);
        }}
        .popup-body iframe {{
            width: 100%;
            height: 100%;
            border: none;
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
        <img id="mask-overlay" class="seg-mask-overlay" style="display:none;" />
        <iframe id="viewer-iframe" src="/viewer/index.html?content=/models/{first_model}&noui&webgl"></iframe>
    </div>

    <!-- Popup modal for 3D viewer -->
    <div id="popup-modal" class="popup-overlay hidden" onclick="if(event.target===this)closePopup()">
        <div class="popup-container">
            <div class="popup-header">
                <h3 id="popup-title">3D Viewer</h3>
                <button class="popup-close" onclick="closePopup()">&times;</button>
            </div>
            <div class="popup-body">
                <iframe id="popup-iframe" src="about:blank"></iframe>
            </div>
        </div>
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
            <div id="seg-object-list" style="max-height:140px;overflow-y:auto;margin-top:6px;"></div>
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
        let selectedMaskIndex = null;
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
            hideMaskOverlay();

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

        function captureCameraMatrices() {{
            // Grab the real view/projection matrices from the PlayCanvas viewer
            try {{
                const iframe = document.getElementById('viewer-iframe');
                const fn = iframe.contentWindow.getCameraMatrices;
                if (typeof fn === 'function') return fn();
            }} catch(e) {{ console.warn('Camera matrices unavailable:', e); }}
            return null;
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
                // Capture current viewer frame + camera matrices from iframe
                const imageData = captureViewerFrame();
                const cameraData = captureCameraMatrices();
                statusEl.textContent = '🔍 Segmenting with SAM3...';

                // Send frame + prompt + camera to backend
                const resp = await fetch('/segment/text', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ prompt: prompt, image: imageData, camera: cameraData }})
                }});
                const data = await resp.json();
                if (data.status === 'ok') {{
                    segMasks = data;
                    selectedMaskIndex = null;
                    statusEl.textContent = '✅ Found ' + data.num_masks + ' object(s): "' + prompt + '" — select ONE below';
                    document.getElementById('extract-btn').disabled = true;
                    buildObjectList(data.masks, prompt);
                    // Show mask overlay on the viewer
                    if (data.overlay) {{
                        showMaskOverlay(data.overlay);
                    }}
                }} else {{
                    statusEl.textContent = '❌ ' + (data.error || 'Segmentation failed');
                }}
            }} catch(e) {{
                statusEl.textContent = '❌ Error: ' + e.message;
            }}
        }}

        function showMaskOverlay(overlayBase64) {{
            const overlayEl = document.getElementById('mask-overlay');
            overlayEl.src = 'data:image/png;base64,' + overlayBase64;
            overlayEl.style.display = 'block';
        }}

        function hideMaskOverlay() {{
            const overlayEl = document.getElementById('mask-overlay');
            overlayEl.style.display = 'none';
            overlayEl.src = '';
        }}

        function clearSegmentation() {{
            segMasks = null;
            selectedMaskIndex = null;
            hideMaskOverlay();
            document.getElementById('seg-object-list').innerHTML = '';
            document.getElementById('seg-status').textContent = 'Click on viewer to segment objects with text or click';
            document.getElementById('extract-btn').disabled = true;
            fetch('/segment/clear', {{ method: 'POST' }}).catch(() => {{}});
        }}

        // Build a selectable list of detected objects (one selection at a time)
        function buildObjectList(masks, prompt) {{
            const list = document.getElementById('seg-object-list');
            list.innerHTML = '';
            masks.forEach((m) => {{
                const item = document.createElement('div');
                item.id = 'seg-obj-' + m.index;
                item.style.cssText = 'display:flex;align-items:center;gap:6px;padding:4px 8px;margin:2px 0;border-radius:6px;cursor:pointer;background:#1a1a2e;border:1px solid #333;font-size:12px;color:#ccc;';
                item.innerHTML = `<input type="radio" name="seg-obj" ${{selectedMaskIndex === m.index ? 'checked' : ''}} style="pointer-events:none;"/>
                    <span>${{prompt}} #${{m.index + 1}} (score ${{m.score.toFixed(2)}})</span>`;
                item.onclick = () => selectObject(m.index);
                list.appendChild(item);
            }});
        }}

        // Select a single object for extraction/reconstruction
        async function selectObject(index) {{
            selectedMaskIndex = index;
            // Update list styling + radio state
            document.querySelectorAll('#seg-object-list > div').forEach((el) => {{
                const isSel = el.id === 'seg-obj-' + index;
                el.style.background = isSel ? '#2a4d6e' : '#1a1a2e';
                el.style.border = isSel ? '1px solid #4da3ff' : '1px solid #333';
                el.querySelector('input').checked = isSel;
            }});
            document.getElementById('extract-btn').disabled = false;
            document.getElementById('seg-status').textContent = '\u2705 Object #' + (index + 1) + ' selected \u2014 ready to extract';
            // Fetch single-object overlay to highlight only the selected mask
            try {{
                const resp = await fetch('/segment/mask_overlay/' + index);
                const data = await resp.json();
                if (data.status === 'ok' && data.overlay) {{
                    showMaskOverlay(data.overlay);
                }}
            }} catch(e) {{ /* keep previous overlay */ }}
        }}

        // ===================================================================
        // Popup modal helpers
        // ===================================================================
        function openPopup(title, url) {{
            document.getElementById('popup-title').textContent = title;
            document.getElementById('popup-iframe').src = url;
            document.getElementById('popup-modal').classList.remove('hidden');
        }}

        function closePopup() {{
            document.getElementById('popup-modal').classList.add('hidden');
            document.getElementById('popup-iframe').src = 'about:blank';
        }}

        // Escape key closes popup
        document.addEventListener('keydown', (e) => {{
            if (e.key === 'Escape') closePopup();
        }});

        // ===================================================================
        // 3D Extraction (GARField-style)
        // ===================================================================
        async function extract3D() {{
            if (!segMasks) {{
                alert('Run segmentation first');
                return;
            }}
            if (selectedMaskIndex === null) {{
                alert('Select ONE object from the list first');
                return;
            }}
            const statusEl = document.getElementById('extract-status');
            statusEl.textContent = '⏳ Extracting 3D gaussians from mask region...';
            document.getElementById('extract-btn').disabled = true;

            try {{
                const resp = await fetch('/garfield/extract', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ model: currentModel, mask_index: selectedMaskIndex }})
                }});
                const data = await resp.json();
                if (data.status === 'ok') {{
                    // Poll for extraction completion
                    statusEl.textContent = '⏳ Extracting... (reading ' + currentModel + ')';
                    const jobId = data.job_id;
                    const pollInterval = setInterval(async () => {{
                        try {{
                            const sResp = await fetch('/garfield/status/' + jobId);
                            const sData = await sResp.json();
                            if (sData.status === 'done') {{
                                clearInterval(pollInterval);
                                data.num_gaussians = sData.num_gaussians;
                                lastExtraction = data;
                                statusEl.textContent = '✅ Extracted ' + sData.num_gaussians.toLocaleString() + ' gaussians from masked region';
                                document.getElementById('trellis-btn').disabled = false;
                                document.getElementById('extract-btn').disabled = false;
                                addExtractionToList(data);
                            }} else if (sData.status === 'error') {{
                                clearInterval(pollInterval);
                                statusEl.textContent = '❌ Extraction failed';
                                document.getElementById('extract-btn').disabled = false;
                            }}
                        }} catch(pe) {{ /* ignore poll errors */ }}
                    }}, 2000);
                }} else {{
                    statusEl.textContent = '❌ ' + (data.error || 'Extraction failed');
                    document.getElementById('extract-btn').disabled = false;
                }}
            }} catch(e) {{
                statusEl.textContent = '❌ Error: ' + e.message;
                document.getElementById('extract-btn').disabled = false;
            }}
        }}

        function addExtractionToList(data) {{
            const list = document.getElementById('extraction-list');
            const item = document.createElement('div');
            item.className = 'ext-item';
            item.innerHTML = `
                <span>${{data.job_id}} (${{data.num_gaussians}} gs)</span>
                <span>
                    <a href="javascript:void(0)" onclick="openPopup('3D Extraction: ${{data.job_id}}', '/viewer/index.html?content=/garfield/download/${{data.job_id}}&noui&webgl')" style="color:#28a745;font-size:11px;margin-right:6px;">👁️ View</a>
                    <a href="/garfield/download/${{data.job_id}}" style="color:#17a2b8;font-size:11px;">⬇️ PLY</a>
                </span>
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
                    statusEl.textContent = '✅ Reconstruction complete! Job: ' + data.job_id;
                    if (data.viewer_url) {{
                        statusEl.innerHTML += ` <a href="javascript:void(0)" onclick="openPopup('TRELLIS Reconstruction: ${{data.job_id}}', '${{data.viewer_url}}')" style="color:#6c63ff;">👁️ View 3D</a>`;
                        statusEl.innerHTML += ` <a href="${{data.viewer_url}}" target="_blank" style="color:#888;font-size:10px;margin-left:6px;">↗ new tab</a>`;
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

    success = await asyncio.to_thread(load_sam3)
    if not success:
        return {"status": "error", "error": "SAM3 model failed to load"}

    # Serialize GPU inference: one request at a time avoids memory spikes
    await _sam3_semaphore.acquire()
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

        # Save camera view/projection matrices for accurate 3D extraction
        camera_data = body.get("camera")
        camera_path = CACHE_DIR / "camera.json"
        if camera_data and camera_data.get("view") and camera_data.get("proj"):
            with open(camera_path, "w") as f:
                json.dump(camera_data, f)
            logger.info("Saved camera matrices for extraction")
        elif camera_path.exists():
            camera_path.unlink()  # stale camera from a previous frame

        import torch
        image = Image.open(frame_path).convert("RGB")
        # Use autocast to handle BFloat16/Float32 dtype on GB10 Blackwell
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            inference_state = sam3_processor.set_image(image)
            output = sam3_processor.set_text_prompt(prompt, inference_state)

        masks = output["masks"]
        boxes = output["boxes"]
        scores = output["scores"]

        # Save masks and generate overlay image for the client
        import base64 as b64mod
        mask_data = []
        # Composite overlay: semi-transparent colored mask for each detection
        overlay_colors = [
            (0, 120, 255, 100),   # blue
            (255, 80, 0, 100),    # orange
            (0, 200, 80, 100),    # green
            (200, 0, 200, 100),   # magenta
        ]
        overlay_img = None

        for i, mask in enumerate(masks):
            mask_np = mask.cpu().float().numpy() if hasattr(mask, 'cpu') else np.array(mask)
            # Squeeze extra dimensions — SAM3 returns (1,H,W) or (H,W)
            while mask_np.ndim > 2:
                mask_np = mask_np[0]
            mask_path = CACHE_DIR / f"mask_{i}.npy"
            np.save(str(mask_path), mask_np)

            # Build RGBA overlay for this mask
            h, w = mask_np.shape
            if overlay_img is None:
                overlay_img = np.zeros((h, w, 4), dtype=np.uint8)
            color = overlay_colors[i % len(overlay_colors)]
            binary = (mask_np > 0.5).astype(np.uint8)
            for c in range(4):
                overlay_img[:, :, c] = np.where(binary, color[c], overlay_img[:, :, c])

            bbox_val = boxes[i].tolist() if hasattr(boxes[i], 'tolist') else list(boxes[i])
            mask_data.append({
                "index": i,
                "score": float(scores[i]) if hasattr(scores[i], 'item') else float(scores[i]),
                "bbox": bbox_val
            })

        # Encode overlay as base64 PNG for the client
        overlay_b64 = ""
        if overlay_img is not None:
            pil_overlay = Image.fromarray(overlay_img, 'RGBA')
            buf = io.BytesIO()
            pil_overlay.save(buf, format='PNG')
            overlay_b64 = b64mod.b64encode(buf.getvalue()).decode('utf-8')

        return {
            "status": "ok",
            "num_masks": len(masks),
            "masks": mask_data,
            "prompt": prompt,
            "overlay": overlay_b64
        }

    except Exception as e:
        logger.error(f"SAM3 segmentation error: {e}\n{traceback.format_exc()}")
        return {"status": "error", "error": str(e)}
    finally:
        _sam3_semaphore.release()
        _release_sam3_cache()


@app.post("/segment/click")
async def segment_with_click(body: dict):
    """Segment at a click point using SAM3."""
    x = body.get("x", 0)
    y = body.get("y", 0)

    success = await asyncio.to_thread(load_sam3)
    if not success:
        return {"status": "error", "error": "SAM3 model failed to load"}

    await _sam3_semaphore.acquire()
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
    finally:
        _sam3_semaphore.release()
        _release_sam3_cache()


@app.post("/segment/clear")
async def segment_clear():
    """Clear segmentation state."""
    for f in CACHE_DIR.glob("mask_*.npy"):
        f.unlink()
    return {"status": "ok"}


@app.get("/segment/mask_overlay/{mask_index}")
async def segment_mask_overlay(mask_index: int):
    """Return a highlight overlay PNG (base64) for a single selected mask."""
    import base64 as b64mod
    mask_path = CACHE_DIR / f"mask_{mask_index}.npy"
    if not mask_path.exists():
        return {"status": "error", "error": f"Mask {mask_index} not found"}
    try:
        mask_np = np.load(str(mask_path))
        while mask_np.ndim > 2:
            mask_np = mask_np[0]
        h, w = mask_np.shape
        binary = (mask_np > 0.5).astype(np.uint8)
        overlay_img = np.zeros((h, w, 4), dtype=np.uint8)
        # Bright green highlight for the selected object
        color = (0, 220, 100, 130)
        for c in range(4):
            overlay_img[:, :, c] = np.where(binary, color[c], 0)
        pil_overlay = Image.fromarray(overlay_img, 'RGBA')
        buf = io.BytesIO()
        pil_overlay.save(buf, format='PNG')
        overlay_b64 = b64mod.b64encode(buf.getvalue()).decode('utf-8')
        return {"status": "ok", "overlay": overlay_b64, "mask_index": mask_index}
    except Exception as e:
        logger.error(f"Mask overlay error: {e}")
        return {"status": "error", "error": str(e)}


# ---------------------------------------------------------------------------
# Routes: GARField-style 3D extraction
# ---------------------------------------------------------------------------
def _parse_ply_header(filepath: Path):
    """Parse PLY header and return (header_end, num_vertices, vertex_properties, bytes_per_vertex).

    Handles mixed-type PLY properties and multi-element PLY files.
    Only returns properties belonging to the vertex element.
    """
    TYPE_SIZES = {
        'float': 4, 'float32': 4,
        'double': 8, 'float64': 8,
        'uchar': 1, 'uint8': 1, 'char': 1, 'int8': 1,
        'short': 2, 'int16': 2, 'ushort': 2, 'uint16': 2,
        'int': 4, 'int32': 4, 'uint': 4, 'uint32': 4,
    }

    with open(filepath, 'rb') as f:
        lines = []
        while True:
            line = f.readline().decode('ascii', errors='ignore').strip()
            lines.append(line)
            if line == 'end_header':
                break
        header_end = f.tell()

    # Parse elements in order — only collect vertex properties
    num_vertices = 0
    vertex_properties = []
    current_element = None
    for line in lines:
        if line.startswith('element'):
            parts = line.split()
            current_element = parts[1]
            if current_element == 'vertex':
                num_vertices = int(parts[2])
        elif line.startswith('property') and current_element == 'vertex':
            parts = line.split()
            dtype = parts[1]
            name = parts[2]
            vertex_properties.append((dtype, name))

    bytes_per_vertex = sum(TYPE_SIZES.get(p[0], 4) for p in vertex_properties)
    return header_end, num_vertices, vertex_properties, bytes_per_vertex


def _extract_gaussians_by_mask(model_path: Path, mask: np.ndarray, output_path: Path,
                               camera: Optional[dict] = None) -> int:
    """Extract gaussians whose 3D positions project into the 2D mask region.

    When camera view/projection matrices are provided (from the viewer at
    frame-capture time), uses accurate perspective projection so the extracted
    gaussians match EXACTLY what the user segmented on screen.
    Falls back to orthographic bounding-box projection when no camera is given.
    Returns count of extracted gaussians.
    """
    import struct

    TYPE_SIZES = {
        'float': 4, 'float32': 4,
        'double': 8, 'float64': 8,
        'uchar': 1, 'uint8': 1, 'char': 1, 'int8': 1,
        'short': 2, 'int16': 2, 'ushort': 2, 'uint16': 2,
        'int': 4, 'int32': 4, 'uint': 4, 'uint32': 4,
    }
    STRUCT_FMTS = {
        'float': 'f', 'float32': 'f',
        'double': 'd', 'float64': 'd',
        'uchar': 'B', 'uint8': 'B', 'char': 'b', 'int8': 'b',
        'short': 'h', 'int16': 'h', 'ushort': 'H', 'uint16': 'H',
        'int': 'i', 'int32': 'i', 'uint': 'I', 'uint32': 'I',
    }

    header_end, num_vertices, properties, bytes_per_vertex = _parse_ply_header(model_path)

    # Build struct format for one vertex and find x,y,z byte offsets
    prop_names = [p[1] for p in properties]
    x_idx = prop_names.index('x') if 'x' in prop_names else 0
    y_idx = prop_names.index('y') if 'y' in prop_names else 1
    z_idx = prop_names.index('z') if 'z' in prop_names else 2

    # Calculate byte offsets for x, y, z
    x_offset = sum(TYPE_SIZES.get(properties[i][0], 4) for i in range(x_idx))
    y_offset = sum(TYPE_SIZES.get(properties[i][0], 4) for i in range(y_idx))
    z_offset = sum(TYPE_SIZES.get(properties[i][0], 4) for i in range(z_idx))
    x_fmt = '<' + STRUCT_FMTS.get(properties[x_idx][0], 'f')
    y_fmt = '<' + STRUCT_FMTS.get(properties[y_idx][0], 'f')
    z_fmt = '<' + STRUCT_FMTS.get(properties[z_idx][0], 'f')
    x_size = TYPE_SIZES.get(properties[x_idx][0], 4)

    # Squeeze mask to 2D
    while mask.ndim > 2:
        mask = mask[0]
    mask_h, mask_w = mask.shape

    # Find mask bounding box
    mask_binary = (mask > 0.5).astype(np.uint8)
    rows = np.any(mask_binary, axis=1)
    cols = np.any(mask_binary, axis=0)
    if not rows.any() or not cols.any():
        return 0
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Read all raw vertex data as bytes
    with open(model_path, 'rb') as f:
        f.seek(header_end)
        raw_data = f.read(num_vertices * bytes_per_vertex)

    if len(raw_data) != num_vertices * bytes_per_vertex:
        raise ValueError(f"PLY data size mismatch: got {len(raw_data)}, expected {num_vertices * bytes_per_vertex}")

    # Extract x,y,z positions using numpy for speed (assuming x,y,z are float32)
    # This works for the common case; for exotic types we fall back to struct
    if all(properties[i][0] in ('float', 'float32') for i in [x_idx, y_idx, z_idx]):
        raw_array = np.frombuffer(raw_data, dtype=np.uint8).reshape(num_vertices, bytes_per_vertex)
        positions_x = np.frombuffer(raw_array[:, x_offset:x_offset+4].tobytes(), dtype=np.float32)
        positions_y = np.frombuffer(raw_array[:, y_offset:y_offset+4].tobytes(), dtype=np.float32)
        positions_z = np.frombuffer(raw_array[:, z_offset:z_offset+4].tobytes(), dtype=np.float32)
    else:
        # Fallback: struct unpack (slower)
        positions_x = np.zeros(num_vertices, dtype=np.float32)
        positions_y = np.zeros(num_vertices, dtype=np.float32)
        positions_z = np.zeros(num_vertices, dtype=np.float32)
        for i in range(num_vertices):
            offset = i * bytes_per_vertex
            positions_x[i] = struct.unpack_from(x_fmt, raw_data, offset + x_offset)[0]
            positions_y[i] = struct.unpack_from(y_fmt, raw_data, offset + y_offset)[0]
            positions_z[i] = struct.unpack_from(z_fmt, raw_data, offset + z_offset)[0]

    # Filter out NaN/Inf positions
    valid = np.isfinite(positions_x) & np.isfinite(positions_y) & np.isfinite(positions_z)
    if not valid.any():
        return 0

    if camera and camera.get("view") and camera.get("proj"):
        # --- Accurate perspective projection using the viewer camera ---
        # PlayCanvas Mat4.data is column-major; transpose to row-major
        view = np.array(camera["view"], dtype=np.float64).reshape(4, 4).T
        proj = np.array(camera["proj"], dtype=np.float64).reshape(4, 4).T
        vp = proj @ view
        # Apply the splat entity's model transform (viewer rotates PLY 180° about Z)
        if camera.get("model"):
            model = np.array(camera["model"], dtype=np.float64).reshape(4, 4).T
            vp = vp @ model

        px = positions_x.astype(np.float64)
        py = positions_y.astype(np.float64)
        pz = positions_z.astype(np.float64)

        clip_x = vp[0, 0] * px + vp[0, 1] * py + vp[0, 2] * pz + vp[0, 3]
        clip_y = vp[1, 0] * px + vp[1, 1] * py + vp[1, 2] * pz + vp[1, 3]
        clip_w = vp[3, 0] * px + vp[3, 1] * py + vp[3, 2] * pz + vp[3, 3]

        in_front = clip_w > 1e-8
        safe_w = np.where(in_front, clip_w, 1.0)
        ndc_x = clip_x / safe_w
        ndc_y = clip_y / safe_w

        # NDC [-1,1] -> pixel coords (y flipped: NDC +y is up, image row 0 is top)
        u_f = (ndc_x * 0.5 + 0.5) * (mask_w - 1)
        v_f = (1.0 - (ndc_y * 0.5 + 0.5)) * (mask_h - 1)

        # Only points that are in front of the camera AND land inside the image
        on_screen = (valid & in_front &
                     (u_f >= 0) & (u_f <= mask_w - 1) &
                     (v_f >= 0) & (v_f <= mask_h - 1))

        proj_u = np.zeros(num_vertices, dtype=np.int32)
        proj_v = np.zeros(num_vertices, dtype=np.int32)
        proj_u[on_screen] = u_f[on_screen].astype(np.int32)
        proj_v[on_screen] = v_f[on_screen].astype(np.int32)

        inside_mask = on_screen.copy()
        inside_mask[on_screen] = mask_binary[proj_v[on_screen], proj_u[on_screen]] > 0
        selected_indices = np.where(inside_mask)[0]
        logger.info(f"Perspective projection: {on_screen.sum()} on-screen, {len(selected_indices)} in mask")
    else:
        # --- Orthographic fallback (no camera data) ---
        x_min, x_max = float(positions_x[valid].min()), float(positions_x[valid].max())
        y_min, y_max = float(positions_y[valid].min()), float(positions_y[valid].max())
        x_range = x_max - x_min if x_max > x_min else 1.0
        y_range = y_max - y_min if y_max > y_min else 1.0

        proj_u = np.full(num_vertices, 0, dtype=np.int32)
        proj_v = np.full(num_vertices, 0, dtype=np.int32)
        proj_u[valid] = ((positions_x[valid] - x_min) / x_range * (mask_w - 1)).astype(np.int32)
        proj_v[valid] = ((1.0 - (positions_y[valid] - y_min) / y_range) * (mask_h - 1)).astype(np.int32)
        proj_u = np.clip(proj_u, 0, mask_w - 1)
        proj_v = np.clip(proj_v, 0, mask_h - 1)

        inside_mask = valid & (mask_binary[proj_v, proj_u] > 0)
        selected_indices = np.where(inside_mask)[0]
        logger.info(f"Orthographic projection (no camera): {len(selected_indices)} in mask")

    if len(selected_indices) == 0:
        return 0

    # Write output PLY with only selected gaussians
    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {len(selected_indices)}",
    ]
    for dtype, name in properties:
        header_lines.append(f"property {dtype} {name}")
    header_lines.append("end_header")
    header_str = "\n".join(header_lines) + "\n"

    with open(output_path, 'wb') as f:
        f.write(header_str.encode('ascii'))
        # Write selected vertex rows from raw bytes
        for idx in selected_indices:
            start = idx * bytes_per_vertex
            f.write(raw_data[start:start + bytes_per_vertex])

    return len(selected_indices)


@app.post("/garfield/extract")
async def garfield_extract(body: dict):
    """Extract 3D gaussians using the current segmentation mask.

    Projects gaussian centers to 2D using orthographic projection and
    selects only those that fall within the SAM3 mask region.
    """
    model_name = body.get("model", current_model)
    mask_index = body.get("mask_index", 0)
    job_id = str(uuid.uuid4())[:8]

    try:
        # Load the user-selected mask (one object at a time)
        mask_path = CACHE_DIR / f"mask_{mask_index}.npy"
        if not mask_path.exists():
            return {"status": "error", "error": f"Mask {mask_index} not found. Run segmentation and select an object first.", "job_id": job_id}

        mask = np.load(str(mask_path))

        # Load camera matrices saved at segmentation time (for accurate projection)
        camera_data = None
        camera_path = CACHE_DIR / "camera.json"
        if camera_path.exists():
            try:
                with open(camera_path) as f:
                    camera_data = json.load(f)
            except Exception:
                logger.warning("Failed to load camera.json — falling back to orthographic projection")

        model_path = MODEL_DIR / model_name if model_name else None
        if not model_path or not model_path.exists():
            return {"status": "error", "error": "Model not found", "job_id": job_id}

        # Store extraction metadata (including which mask was used)
        output_path = OUTPUT_DIR / f"extraction_{job_id}.ply"
        extractions[job_id] = {
            "job_id": job_id,
            "model": model_name,
            "mask_index": mask_index,
            "mask_shape": list(mask.shape),
            "output_path": str(output_path),
            "status": "extracting",
            "timestamp": time.time()
        }

        # Run extraction in background
        async def _extract():
            try:
                count = await asyncio.to_thread(
                    _extract_gaussians_by_mask, model_path, mask, output_path, camera_data
                )
                extractions[job_id]["status"] = "done"
                extractions[job_id]["num_gaussians"] = count
                logger.info(f"Extraction {job_id} complete: {count} gaussians extracted")
            except Exception as e:
                logger.error(f"Extraction {job_id} failed: {e}\n{traceback.format_exc()}")
                extractions[job_id]["status"] = "error"
                extractions[job_id]["error"] = str(e)

        asyncio.create_task(_extract())

        return {
            "status": "ok",
            "job_id": job_id,
            "num_gaussians": "extracting...",
            "message": "Extracting masked gaussians from PLY"
        }

    except Exception as e:
        logger.error(f"GARField extract error: {e}\n{traceback.format_exc()}")
        return {"status": "error", "error": str(e), "job_id": job_id}


@app.get("/garfield/status/{job_id}")
async def garfield_status(job_id: str):
    """Get extraction job status."""
    ext = extractions.get(job_id)
    if not ext:
        return {"status": "error", "error": "Extraction not found"}
    return {
        "status": ext.get("status", "unknown"),
        "num_gaussians": ext.get("num_gaussians", 0),
        "job_id": job_id
    }


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
        # Crop the captured frame to the mask bounding box for TRELLIS
        frame_path = CACHE_DIR / "current_frame.png"
        if not frame_path.exists():
            return {"status": "error", "error": "No reference frame for reconstruction"}

        # Use the SAME mask that was selected for the extraction
        mask_index = ext.get("mask_index", 0)
        mask_path = CACHE_DIR / f"mask_{mask_index}.npy"
        frame_img = Image.open(frame_path).convert("RGB")

        if mask_path.exists():
            mask_np = np.load(str(mask_path))
            while mask_np.ndim > 2:
                mask_np = mask_np[0]
            mask_binary = (mask_np > 0.5).astype(np.uint8)
            rows = np.any(mask_binary, axis=1)
            cols = np.any(mask_binary, axis=0)
            if rows.any() and cols.any():
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                # Add padding (10% each side)
                h, w = mask_np.shape
                pad_r = int((rmax - rmin) * 0.1)
                pad_c = int((cmax - cmin) * 0.1)
                rmin = max(0, rmin - pad_r)
                rmax = min(h - 1, rmax + pad_r)
                cmin = max(0, cmin - pad_c)
                cmax = min(w - 1, cmax + pad_c)
                # Crop and add white background where mask is 0
                cropped = np.array(frame_img)[rmin:rmax+1, cmin:cmax+1]
                mask_crop = mask_binary[rmin:rmax+1, cmin:cmax+1]
                # White background outside mask
                cropped[mask_crop == 0] = [255, 255, 255]
                frame_img = Image.fromarray(cropped)

        # Save the cropped image for reference
        cropped_path = CACHE_DIR / f"trellis_input_{job_id}.png"
        frame_img.save(str(cropped_path))

        async with httpx.AsyncClient(timeout=120.0) as client:
            with open(cropped_path, "rb") as f:
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
        return {"status": "error", "error": "Cannot connect to TRELLIS service. It may have crashed (OOM on DGX Spark). Check: docker logs trellis-reconstructor"}
    except httpx.ReadTimeout:
        return {"status": "error", "error": "TRELLIS timed out (>120s). The model may be too large for available GPU memory."}
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
# Startup: lazy SAM3 with idle unload
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup_sam3_policy():
    """SAM3 is loaded lazily on the first segmentation request and unloaded
    when idle, keeping the GB10's unified memory free for training."""
    asyncio.create_task(_sam3_idle_unload_loop())
    logger.info(
        f"SAM3 policy: lazy load, idle unload after "
        f"{SAM3_IDLE_UNLOAD_SECONDS/60:.0f} min"
    )


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
