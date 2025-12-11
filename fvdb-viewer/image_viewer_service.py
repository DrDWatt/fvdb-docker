#!/usr/bin/env python3
"""
fVDB Image-based Gaussian Splat Viewer Service
Renders to images instead of video stream (works without Vulkan video encoding)
"""

import os
import io
import time
import logging
import math
import numpy as np
from pathlib import Path
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/app/models"))
VIEWER_PORT = int(os.environ.get("VIEWER_PORT", "8085"))

app = FastAPI(title="fVDB Gaussian Splat Viewer")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Global state
gsplat = None
device = None
model_name = ""
model_metadata = None
available_models = []

def get_available_models():
    """Get list of available PLY models"""
    return sorted([m.name for m in MODEL_DIR.glob("*.ply")])

def load_model(model_file=None):
    global gsplat, device, model_name, model_metadata, available_models
    import torch
    import fvdb
    
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
    
    available_models = get_available_models()
    if not available_models:
        logger.error("No models found!")
        return False
    
    # Use specified model or first available
    if model_file and model_file in available_models:
        model_path = MODEL_DIR / model_file
    else:
        model_path = MODEL_DIR / available_models[0]
    
    model_name = model_path.stem
    logger.info(f"Loading {model_path.name}...")
    
    gsplat_obj, metadata = fvdb.GaussianSplat3d.from_ply(str(model_path), device=device)
    gsplat = gsplat_obj
    model_metadata = metadata
    logger.info(f"Loaded {gsplat.num_gaussians} gaussians")
    return True

def render_view(width=800, height=600, azimuth=0, elevation=0, zoom=1.0, cam_idx=0):
    """Render using cameras from PLY metadata with orbit and zoom"""
    import torch
    
    if gsplat is None or model_metadata is None:
        return None
    
    try:
        c2w_all = model_metadata.get('camera_to_world_matrices')
        K_all = model_metadata.get('projection_matrices')
        sizes = model_metadata.get('image_sizes')
        
        if c2w_all is None or K_all is None:
            logger.error("No camera matrices in metadata")
            return None
        
        num_cams = c2w_all.shape[0]
        cam_idx = cam_idx % num_cams
        
        c2w = c2w_all[cam_idx].to(device).clone()
        means = gsplat.means
        center = means.mean(dim=0)
        
        # Get camera position and direction
        cam_pos = c2w[:3, 3].clone()
        
        # Apply zoom - move camera along view direction
        view_dir = center - cam_pos
        view_dir = view_dir / view_dir.norm()
        
        # zoom > 1 = closer, zoom < 1 = farther
        dist_to_center = (center - cam_pos).norm().item()
        new_dist = dist_to_center / zoom
        new_pos = center - view_dir * new_dist
        c2w[:3, 3] = new_pos
        
        # Apply rotation around model center
        if azimuth != 0 or elevation != 0:
            az_rad = math.radians(azimuth)
            el_rad = math.radians(elevation)
            
            # Rotation around Y (azimuth)
            Ry = torch.tensor([
                [math.cos(az_rad), 0, math.sin(az_rad), 0],
                [0, 1, 0, 0],
                [-math.sin(az_rad), 0, math.cos(az_rad), 0],
                [0, 0, 0, 1]
            ], device=device, dtype=torch.float32)
            
            # Rotation around X (elevation)
            Rx = torch.tensor([
                [1, 0, 0, 0],
                [0, math.cos(el_rad), -math.sin(el_rad), 0],
                [0, math.sin(el_rad), math.cos(el_rad), 0],
                [0, 0, 0, 1]
            ], device=device, dtype=torch.float32)
            
            # Translate to center, rotate, translate back
            T_to = torch.eye(4, device=device, dtype=torch.float32)
            T_to[:3, 3] = -center
            T_back = torch.eye(4, device=device, dtype=torch.float32)
            T_back[:3, 3] = center
            
            R_orbit = T_back @ Ry @ Rx @ T_to
            c2w = R_orbit @ c2w
        
        c2w = c2w.unsqueeze(0).contiguous()
        w2c = torch.inverse(c2w).contiguous()
        
        # Scale projection matrix
        orig_h, orig_w = sizes[cam_idx].tolist()
        K = K_all[cam_idx:cam_idx+1].to(device).clone()
        K[:, 0, :] *= width / orig_w
        K[:, 1, :] *= height / orig_h
        K = K.contiguous()
        
        images, alpha = gsplat.render_images(
            world_to_camera_matrices=w2c,
            projection_matrices=K,
            image_width=width,
            image_height=height,
            near=0.01,
            far=100.0
        )
        
        img = images[0].clamp(0, 1).cpu().numpy()
        if img.shape[-1] > 3:
            img = img[..., :3]
        img = (img * 255).astype(np.uint8)
        
        return img
        
    except Exception as e:
        logger.error(f"Render error: {e}")
        import traceback
        traceback.print_exc()
        return None

@app.on_event("startup")
async def startup():
    load_model()

@app.get("/", response_class=HTMLResponse)
async def root():
    models = list(MODEL_DIR.glob("*.ply"))
    model_list = ", ".join([m.name for m in models[:5]])
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>fVDB Gaussian Splat Viewer</title>
        <style>
            body {{ 
                margin: 0; 
                background: #1a1a2e; 
                color: white; 
                font-family: Arial, sans-serif;
                overflow: hidden;
            }}
            #viewer {{ 
                width: 100vw; 
                height: 100vh; 
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
            }}
            #render {{ 
                max-width: 90vw; 
                max-height: 80vh;
                border: 2px solid #76b900;
                border-radius: 8px;
            }}
            #controls {{
                position: fixed;
                top: 10px;
                left: 10px;
                background: rgba(0,0,0,0.8);
                padding: 15px;
                border-radius: 8px;
            }}
            #info {{
                position: fixed;
                top: 10px;
                right: 10px;
                background: rgba(0,0,0,0.8);
                padding: 15px;
                border-radius: 8px;
            }}
            h1 {{ color: #76b900; margin: 0 0 10px 0; font-size: 18px; }}
            .slider-group {{ margin: 10px 0; }}
            label {{ display: block; margin-bottom: 5px; }}
            input[type="range"] {{ width: 150px; }}
            #loading {{ 
                position: fixed; 
                top: 50%; 
                left: 50%; 
                transform: translate(-50%, -50%);
                font-size: 24px;
            }}
        </style>
    </head>
    <body>
        <div id="viewer">
            <div id="loading">Loading...</div>
            <img id="render" style="display:none;" />
        </div>
        
        <div id="controls">
            <h1>🎬 fVDB Viewer</h1>
            <div class="slider-group">
                <label>Model:</label>
                <select id="model-select" style="width:150px;padding:5px;">
                    <option>Loading...</option>
                </select>
                <button onclick="refreshModels()" style="margin-left:5px;padding:5px 8px;background:#28a745;color:white;border:none;border-radius:4px;cursor:pointer;" title="Refresh model list">🔄</button>
                <button onclick="deleteCurrentModel()" style="margin-left:10px;padding:5px 10px;background:#dc3545;color:white;border:none;border-radius:4px;cursor:pointer;">🗑️ Delete</button>
            </div>
            <div class="slider-group">
                <label>Rotation: <span id="az-val">0</span>°</label>
                <input type="range" id="azimuth" min="-180" max="180" value="0">
            </div>
            <div class="slider-group">
                <label>Elevation: <span id="el-val">0</span>°</label>
                <input type="range" id="elevation" min="-45" max="45" value="0">
            </div>
            <div class="slider-group">
                <label>Zoom: <span id="zoom-val">0.5</span>x</label>
                <input type="range" id="zoom" min="0.1" max="2" value="0.5" step="0.1">
            </div>
            <div class="slider-group">
                <label>Camera: <span id="cam-val">0</span></label>
                <input type="range" id="camera" min="0" max="14" value="0" step="1">
            </div>
        </div>
        
        <div id="info">
            <p><strong>Models:</strong> {model_list}</p>
            <p><strong>Gaussians:</strong> <span id="num-gs">Loading...</span></p>
            <p>Drag sliders to rotate view</p>
        </div>
        
        <script>
            const img = document.getElementById('render');
            const loading = document.getElementById('loading');
            const azSlider = document.getElementById('azimuth');
            const elSlider = document.getElementById('elevation');
            const zoomSlider = document.getElementById('zoom');
            const camSlider = document.getElementById('camera');
            
            let debounceTimer;
            
            async function updateRender() {{
                const az = azSlider.value;
                const el = elSlider.value;
                const zoom = zoomSlider.value;
                const cam = camSlider.value;
                
                document.getElementById('az-val').textContent = az;
                document.getElementById('el-val').textContent = el;
                document.getElementById('zoom-val').textContent = zoom;
                document.getElementById('cam-val').textContent = cam;
                
                loading.style.display = 'block';
                
                try {{
                    const response = await fetch(`/render?azimuth=${{az}}&elevation=${{el}}&zoom=${{zoom}}&cam_idx=${{cam}}&width=1024&height=768`);
                    if (response.ok) {{
                        const blob = await response.blob();
                        img.src = URL.createObjectURL(blob);
                        img.style.display = 'block';
                        loading.style.display = 'none';
                    }}
                }} catch(e) {{
                    console.error('Render error:', e);
                }}
            }}
            
            function debouncedUpdate() {{
                clearTimeout(debounceTimer);
                debounceTimer = setTimeout(updateRender, 150);
            }}
            
            azSlider.addEventListener('input', debouncedUpdate);
            elSlider.addEventListener('input', debouncedUpdate);
            zoomSlider.addEventListener('input', debouncedUpdate);
            camSlider.addEventListener('input', debouncedUpdate);
            
            // Model selection
            const modelSelect = document.getElementById('model-select');
            modelSelect.addEventListener('change', async () => {{
                loading.textContent = 'Loading model...';
                loading.style.display = 'block';
                img.style.display = 'none';
                
                const response = await fetch('/load_model?model=' + modelSelect.value);
                if (response.ok) {{
                    // Reset controls
                    azSlider.value = 0;
                    elSlider.value = 0;
                    zoomSlider.value = 0.5;
                    camSlider.value = 0;
                    
                    // Update info and render
                    const info = await fetch('/info').then(r => r.json());
                    document.getElementById('num-gs').textContent = info.num_gaussians || 'N/A';
                    updateRender();
                }}
            }});
            
            // Delete current model
            async function deleteCurrentModel() {{
                const model = modelSelect.value;
                if (!confirm('Delete ' + model + '? This cannot be undone.')) return;
                
                try {{
                    const response = await fetch('/delete_model?model=' + model, {{ method: 'DELETE' }});
                    if (response.ok) {{
                        alert('Deleted ' + model);
                        location.reload();
                    }} else {{
                        alert('Failed to delete: ' + (await response.text()));
                    }}
                }} catch(e) {{
                    alert('Error: ' + e);
                }}
            }}
            
            // Refresh models list from server
            async function refreshModels() {{
                const response = await fetch('/info');
                const data = await response.json();
                const select = document.getElementById('model-select');
                const currentModel = data.model_name + '.ply';
                
                // Clear and repopulate dropdown
                select.innerHTML = '';
                (data.models_available || []).forEach(m => {{
                    const opt = document.createElement('option');
                    opt.value = m;
                    opt.textContent = m;
                    if (m === currentModel || m.replace('.ply','') === data.model_name) {{
                        opt.selected = true;
                    }}
                    select.appendChild(opt);
                }});
                
                document.getElementById('num-gs').textContent = data.num_gaussians || 'N/A';
                
                // Update info panel
                const infoModels = document.querySelector('#info p strong');
                if (infoModels && infoModels.textContent === 'Models:') {{
                    infoModels.parentElement.innerHTML = '<strong>Models:</strong> ' + (data.models_available || []).join(', ');
                }}
            }}
            
            // Initial load
            refreshModels().then(() => updateRender());
        </script>
    </body>
    </html>
    """

@app.get("/render")
async def render(
    width: int = Query(800, ge=100, le=1920),
    height: int = Query(600, ge=100, le=1080),
    azimuth: float = Query(0, ge=-180, le=180),
    elevation: float = Query(0, ge=-45, le=45),
    zoom: float = Query(0.5, ge=0.1, le=2.0),
    cam_idx: int = Query(0, ge=0, le=100)
):
    """Render the gaussian splat and return as PNG"""
    img = render_view(width, height, azimuth, elevation, zoom, cam_idx)
    
    if img is None:
        return Response(content=b"Render failed", status_code=500)
    
    # Encode as PNG
    from PIL import Image
    pil_img = Image.fromarray(img)
    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG')
    buffer.seek(0)
    
    return Response(content=buffer.getvalue(), media_type="image/png")

@app.get("/load_model")
async def load_model_endpoint(model: str = Query(...)):
    """Load a different model"""
    success = load_model(model)
    if success:
        return {"status": "ok", "model": model_name, "num_gaussians": gsplat.num_gaussians}
    return {"status": "error", "message": "Failed to load model"}

@app.get("/info")
async def info():
    # Always refresh the available models list
    current_models = get_available_models()
    return {
        "model_name": model_name,
        "num_gaussians": gsplat.num_gaussians if gsplat else 0,
        "models_available": current_models
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": gsplat is not None}

@app.delete("/delete_model")
async def delete_model_endpoint(model: str = Query(...)):
    """Delete a model PLY file"""
    global gsplat, model_metadata, model_name, available_models
    
    model_path = MODEL_DIR / model
    if not model_path.exists():
        return Response(content=f"Model {model} not found", status_code=404)
    
    # Delete the file
    model_path.unlink()
    logger.info(f"Deleted model: {model}")
    
    # Also delete metadata if exists
    meta_path = MODEL_DIR / f"{model.replace('.ply', '')}_metadata.json"
    if meta_path.exists():
        meta_path.unlink()
    
    # Refresh available models and load a different one
    available_models = get_available_models()
    if available_models:
        load_model(available_models[0])
    else:
        gsplat = None
        model_metadata = None
        model_name = ""
    
    return {"status": "ok", "message": f"Deleted {model}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=VIEWER_PORT)
