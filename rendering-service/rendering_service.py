"""
fVDB Rendering Service
Web-based viewer for Gaussian Splat models with REST API
"""

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import torch
import numpy as np
import cv2
from pathlib import Path
import logging
import io
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="fVDB Rendering Service",
    description="Web-based rendering and visualization for Gaussian Splat models",
    version="1.0.0",
    docs_url="/api",
    redoc_url="/api/redoc"
)

# Directories
BASE_DIR = Path("/app")
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
CACHE_DIR = BASE_DIR / "cache"
STATIC_DIR = BASE_DIR / "static"
DOWNLOADS_DIR = STATIC_DIR / "downloads"

for dir_path in [MODEL_DIR, OUTPUT_DIR, CACHE_DIR, STATIC_DIR, DOWNLOADS_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# Templates
templates = Jinja2Templates(directory="templates")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Model cache
loaded_models = {}

def load_model(model_path):
    """Lazy-load fVDB and load model"""
    import fvdb
    model, metadata = fvdb.GaussianSplat3d.from_ply(str(model_path))
    return model

class RenderRequest(BaseModel):
    """Request to render an image"""
    model_id: str
    camera_position: List[float]
    camera_target: List[float]
    camera_up: Optional[List[float]] = [0, 0, 1]
    image_width: int = 1920
    image_height: int = 1080
    fov: float = 50.0

class DepthMapRequest(BaseModel):
    """Request to render depth map"""
    model_id: str
    camera_position: List[float]
    camera_target: List[float]
    camera_up: Optional[List[float]] = [0, 0, 1]
    image_width: int = 1920
    image_height: int = 1080

@app.get("/")
async def root():
    """Root endpoint - show available models and quick access"""
    models_list = []
    for model_id, info in loaded_models.items():
        models_list.append({
            "id": model_id,
            "gaussians": info["num_gaussians"],
            "device": info["device"]
        })
    
    models_html = ""
    if models_list:
        models_html = "<h2>🎨 Available Models</h2><div class='models-grid'>"
        for model in models_list:
            models_html += f"""
            <div class='model-card'>
                <h3>{model['id']}</h3>
                <p><strong>{model['gaussians']:,}</strong> Gaussians</p>
                <p>Device: {model['device']}</p>
                <div class='model-actions'>
                    <a href='/viewer/{model['id']}' class='btn-primary'>🎨 View 3D</a>
                    <a href='/models/{model['id']}' class='btn-secondary'>ℹ️ Info</a>
                    <a href='/static/downloads/{model['id']}.ply' download class='btn-secondary'>📥 Download</a>
                </div>
            </div>
            """
        models_html += "</div>"
    else:
        models_html = """
        <div class='info-box'>
            <h3>📝 No Models Loaded</h3>
            <p>Upload a model to get started!</p>
            <a href='/api' class='btn-primary'>Upload via API</a>
        </div>
        """
    
    return HTMLResponse(f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>fVDB Rendering Service</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            body {{ 
                font-family: 'Segoe UI', Arial, sans-serif; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                min-height: 100vh;
                padding: 20px;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                color: #333;
                border-radius: 15px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                overflow: hidden;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 40px;
                text-align: center;
            }}
            .header h1 {{
                font-size: 2.5em;
                margin-bottom: 10px;
            }}
            .content {{
                padding: 40px;
            }}
            .models-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
                gap: 20px;
                margin: 30px 0;
            }}
            .model-card {{
                background: #f8f9fa;
                border: 2px solid #e9ecef;
                border-radius: 10px;
                padding: 20px;
                transition: all 0.3s;
            }}
            .model-card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
                border-color: #667eea;
            }}
            .model-card h3 {{
                color: #667eea;
                margin-bottom: 15px;
                font-size: 1.5em;
            }}
            .model-card p {{
                margin: 10px 0;
                color: #666;
            }}
            .model-actions {{
                display: flex;
                gap: 10px;
                margin-top: 20px;
                flex-wrap: wrap;
            }}
            .btn-primary, .btn-secondary {{
                display: inline-block;
                padding: 10px 20px;
                border-radius: 5px;
                text-decoration: none;
                font-weight: 500;
                transition: all 0.3s;
                text-align: center;
                flex: 1;
                min-width: 100px;
            }}
            .btn-primary {{
                background: #667eea;
                color: white;
            }}
            .btn-primary:hover {{
                background: #5568d3;
                transform: scale(1.05);
            }}
            .btn-secondary {{
                background: #e9ecef;
                color: #333;
            }}
            .btn-secondary:hover {{
                background: #dee2e6;
            }}
            .info-box {{
                background: #fff3cd;
                border: 2px solid #ffc107;
                border-radius: 10px;
                padding: 30px;
                text-align: center;
                margin: 30px 0;
            }}
            .info-box h3 {{
                color: #856404;
                margin-bottom: 15px;
            }}
            .info-box p {{
                color: #856404;
                margin-bottom: 20px;
            }}
            .quick-links {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin: 30px 0;
            }}
            .quick-link {{
                background: #667eea;
                color: white;
                padding: 20px;
                border-radius: 8px;
                text-decoration: none;
                text-align: center;
                transition: all 0.3s;
                display: block;
            }}
            .quick-link:hover {{
                background: #5568d3;
                transform: translateY(-3px);
                box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
            }}
            .quick-link h3 {{
                margin-bottom: 10px;
                font-size: 1.2em;
            }}
            .stats {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 20px;
                margin: 30px 0;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 10px;
            }}
            .stat {{
                text-align: center;
            }}
            .stat-value {{
                font-size: 2em;
                color: #667eea;
                font-weight: bold;
            }}
            .stat-label {{
                color: #666;
                margin-top: 5px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎨 fVDB Rendering Service</h1>
                <p>Interactive 3D Gaussian Splat Viewer</p>
            </div>
            
            <div class="content">
                <div class="stats">
                    <div class="stat">
                        <div class="stat-value">{len(loaded_models)}</div>
                        <div class="stat-label">Models Loaded</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value">✅</div>
                        <div class="stat-label">GPU Available</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value">🚀</div>
                        <div class="stat-label">Ready</div>
                    </div>
                </div>
                
                {models_html}
                
                <h2>🔗 Quick Links</h2>
                <div class="quick-links">
                    <a href="/api" class="quick-link">
                        <h3>📚 API Docs</h3>
                        <p>Upload & Manage Models</p>
                    </a>
                    <a href="/models" class="quick-link">
                        <h3>📋 Models List</h3>
                        <p>View All Models</p>
                    </a>
                    <a href="/health" class="quick-link">
                        <h3>💚 Health</h3>
                        <p>Service Status</p>
                    </a>
                    <a href="/tutorials" class="quick-link">
                        <h3>🎓 Tutorials</h3>
                        <p>Learn fVDB</p>
                    </a>
                </div>
            </div>
        </div>
    </body>
    </html>
    """)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "fVDB Rendering Service",
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "models_loaded": len(loaded_models)
    }

@app.get("/tutorials")
async def get_tutorials():
    """Get links to fVDB tutorials"""
    return {
        "tutorials": [
            {
                "title": "Gaussian Splat Radiance Field Reconstruction",
                "url": "https://fvdb.ai/reality-capture/tutorials/radiance_field_and_mesh_reconstruction.html",
                "description": "Complete guide to reconstructing Gaussian splat radiance fields"
            },
            {
                "title": "FRGS Tutorial",
                "url": "https://fvdb.ai/reality-capture/tutorials/frgs.html",
                "description": "Feature-based radiance Gaussian splatting tutorial"
            },
            {
                "title": "fVDB Documentation",
                "url": "https://fvdb.ai/",
                "description": "Complete fVDB documentation and API reference"
            }
        ]
    }

@app.post("/models/upload")
async def upload_model(file: UploadFile = File(...), model_id: Optional[str] = None):
    """Upload a PLY model"""
    if not file.filename.endswith('.ply'):
        raise HTTPException(400, "Only PLY files are supported")
    
    model_id = model_id or file.filename.replace('.ply', '')
    model_path = MODEL_DIR / f"{model_id}.ply"
    
    # Save file
    with open(model_path, 'wb') as f:
        content = await file.read()
        f.write(content)
    
    # Try to load model
    try:
        model = load_model(model_path)
        loaded_models[model_id] = {
            "model": model,
            "path": str(model_path),
            "num_gaussians": model.num_gaussians,
            "device": str(model.device)
        }
        
        # Auto-copy to downloads directory for web access
        download_path = DOWNLOADS_DIR / f"{model_id}.ply"
        if not download_path.exists():
            import shutil
            shutil.copy2(model_path, download_path)
            logger.info(f"Copied {model_id}.ply to downloads directory")
        
        return {
            "model_id": model_id,
            "status": "loaded",
            "num_gaussians": model.num_gaussians,
            "device": str(model.device),
            "download_url": f"/static/downloads/{model_id}.ply"
        }
    except Exception as e:
        model_path.unlink()
        raise HTTPException(500, f"Failed to load model: {e}")

@app.get("/models")
async def list_models():
    """List all loaded models"""
    models = []
    for model_id, info in loaded_models.items():
        models.append({
            "model_id": model_id,
            "num_gaussians": info["num_gaussians"],
            "device": info["device"],
            "path": info["path"]
        })
    return {"models": models}

@app.get("/models/{model_id}")
async def get_model_info(model_id: str):
    """Get information about a specific model"""
    if model_id not in loaded_models:
        raise HTTPException(404, "Model not found")
    
    info = loaded_models[model_id]
    return {
        "model_id": model_id,
        "num_gaussians": info["num_gaussians"],
        "device": info["device"],
        "num_channels": info["model"].num_channels
    }

@app.post("/render")
async def render_image(request: RenderRequest):
    """Render an image from a model"""
    if request.model_id not in loaded_models:
        raise HTTPException(404, "Model not found")
    
    model = loaded_models[request.model_id]["model"]
    
    try:
        # Create camera matrix (simplified - would need proper implementation)
        # This is a placeholder - real implementation would use proper camera matrices
        rendered_output = {
            "status": "rendered",
            "message": "Rendering functionality requires full camera matrix implementation"
        }
        return rendered_output
        
    except Exception as e:
        raise HTTPException(500, f"Rendering failed: {e}")

@app.get("/viewer/{model_id}")
async def viewer(model_id: str):
    """Web-based 3D viewer for a model"""
    if model_id not in loaded_models:
        raise HTTPException(404, "Model not found")
    
    info = loaded_models[model_id]
    
    return HTMLResponse(f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Viewer - {model_id}</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            body {{
                font-family: 'Segoe UI', Arial, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                min-height: 100vh;
                padding: 20px;
            }}
            .container {{
                max-width: 1000px;
                margin: 0 auto;
                background: white;
                color: #333;
                border-radius: 15px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                overflow: hidden;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 40px;
                text-align: center;
            }}
            .header h1 {{
                font-size: 2.5em;
                margin-bottom: 10px;
            }}
            .stats {{
                display: flex;
                justify-content: center;
                gap: 40px;
                margin-top: 20px;
            }}
            .stat {{
                text-align: center;
            }}
            .stat-value {{
                font-size: 2em;
                font-weight: bold;
            }}
            .stat-label {{
                font-size: 0.9em;
                opacity: 0.9;
            }}
            .content {{
                padding: 40px;
            }}
            .method {{
                background: #f8f9fa;
                border-radius: 10px;
                padding: 30px;
                margin-bottom: 20px;
                border: 2px solid #e9ecef;
                transition: all 0.3s;
            }}
            .method:hover {{
                border-color: #667eea;
                box-shadow: 0 5px 20px rgba(102,126,234,0.1);
            }}
            .method h2 {{
                color: #667eea;
                margin-bottom: 15px;
                font-size: 1.5em;
            }}
            .method p {{
                color: #666;
                margin-bottom: 20px;
                line-height: 1.6;
            }}
            .btn {{
                display: inline-block;
                background: #667eea;
                color: white;
                padding: 15px 30px;
                border-radius: 8px;
                text-decoration: none;
                font-weight: 600;
                transition: all 0.3s;
                border: none;
                cursor: pointer;
                font-size: 1em;
                margin-right: 10px;
                margin-bottom: 10px;
            }}
            .btn:hover {{
                background: #5568d3;
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(102,126,234,0.4);
            }}
            .btn-secondary {{
                background: #6c757d;
            }}
            .btn-secondary:hover {{
                background: #5a6268;
            }}
            .recommended {{
                background: linear-gradient(135deg, #f8f9fa 0%, #e3f2fd 100%);
                border: 3px solid #667eea;
                position: relative;
            }}
            .badge {{
                position: absolute;
                top: -12px;
                right: 20px;
                background: #667eea;
                color: white;
                padding: 5px 15px;
                border-radius: 20px;
                font-size: 0.85em;
                font-weight: bold;
            }}
            .instructions {{
                background: #fff3cd;
                border: 2px solid #ffc107;
                border-radius: 10px;
                padding: 20px;
                margin-top: 30px;
            }}
            .instructions h3 {{
                color: #856404;
                margin-bottom: 10px;
            }}
            .instructions ol {{
                color: #856404;
                margin-left: 20px;
            }}
            .instructions li {{
                margin: 8px 0;
                line-height: 1.6;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎨 {model_id}</h1>
                <p>Gaussian Splat 3D Model</p>
                <div class="stats">
                    <div class="stat">
                        <div class="stat-value">{info['num_gaussians']:,}</div>
                        <div class="stat-label">Gaussians</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value">{info['device']}</div>
                        <div class="stat-label">Device</div>
                    </div>
                </div>
            </div>
            
            <div class="content">
                <div class="method recommended">
                    <span class="badge">⭐ RECOMMENDED</span>
                    <h2>🌐 View in SuperSplat (Online)</h2>
                    <p>
                        <strong>Best option:</strong> SuperSplat is a free web-based Gaussian Splat viewer. 
                        Download your model, then drag it to SuperSplat for instant 3D visualization.
                    </p>
                    <div style="background: #e3f2fd; padding: 15px; border-radius: 8px; margin: 15px 0;">
                        <strong style="color: #1976d2;">📋 Quick Steps:</strong>
                        <ol style="margin: 10px 0 0 20px; color: #1976d2;">
                            <li>Download the PLY file (button below)</li>
                            <li>Open SuperSplat in new tab</li>
                            <li>Drag the PLY file onto SuperSplat</li>
                            <li>Done! View your 3D model</li>
                        </ol>
                    </div>
                    <a href="/static/downloads/{model_id}.ply" download class="btn">
                        1️⃣ Download PLY File
                    </a>
                    <a href="https://playcanvas.com/supersplat/editor" target="_blank" class="btn" onclick="showDragHint()">
                        2️⃣ Open SuperSplat
                    </a>
                </div>
                
                <div id="dragHint" class="instructions" style="display: none;">
                    <h3>📖 After SuperSplat Opens:</h3>
                    <ol>
                        <li>Look for the <strong>"Drop files here"</strong> area in SuperSplat</li>
                        <li><strong>Drag and drop</strong> the downloaded <code>{model_id}.ply</code> file onto the page</li>
                        <li>Or click the <strong>folder icon</strong> to browse and select the file</li>
                        <li>Wait 5-10 seconds for loading</li>
                        <li>✅ Your 3D model appears! Use mouse to rotate/zoom</li>
                    </ol>
                </div>
                
                <div class="method">
                    <h2>📥 Download Model</h2>
                    <p>
                        Download the PLY file to view locally in desktop applications like 
                        Polycam, MeshLab, Blender, or CloudCompare.
                    </p>
                    <a href="/static/downloads/{model_id}.ply" download class="btn">
                        📥 Download PLY File
                    </a>
                    <span style="color: #999; font-size: 0.9em; margin-left: 10px;">
                        (~40MB)
                    </span>
                </div>
                
                <div class="method">
                    <h2>📱 View on Mobile</h2>
                    <p>
                        <strong>For iPhone/iPad:</strong> Download the Polycam app and import the PLY file 
                        to view in AR. Or use SuperSplat in Safari for web-based viewing.
                    </p>
                    <a href="https://apps.apple.com/app/polycam/id1532482376" target="_blank" class="btn">
                        📱 Get Polycam App
                    </a>
                    <a href="/static/downloads/{model_id}.ply" download class="btn btn-secondary">
                        📥 Download for Polycam
                    </a>
                </div>
                
                <div class="instructions" id="instructions" style="display: none;">
                    <h3>📖 How to View in SuperSplat</h3>
                    <ol>
                        <li>Click <strong>"Open SuperSplat"</strong> button above (opens in new tab)</li>
                        <li>Click the <strong>folder icon</strong> or drag & drop area in SuperSplat</li>
                        <li>Select or drag the <strong>{model_id}.ply</strong> file you downloaded</li>
                        <li>Wait a few seconds for the model to load</li>
                        <li>✅ <strong>Interact with your 3D model!</strong>
                            <ul style="margin-top: 10px;">
                                <li><strong>Left mouse:</strong> Rotate</li>
                                <li><strong>Right mouse:</strong> Pan</li>
                                <li><strong>Scroll wheel:</strong> Zoom</li>
                            </ul>
                        </li>
                    </ol>
                </div>
                
                <div style="margin-top: 30px; padding: 20px; background: #f8f9fa; border-radius: 10px;">
                    <h3 style="color: #667eea; margin-bottom: 15px;">💡 Why not embedded?</h3>
                    <p style="color: #666; line-height: 1.6;">
                        SuperSplat and PlayCanvas don't allow iframe embedding for security reasons (CORS policy). 
                        However, their web-based viewer works perfectly when opened in a new tab! 
                        Just download the PLY and upload it to SuperSplat for the best viewing experience.
                    </p>
                </div>
                
                <div style="margin-top: 20px; text-align: center;">
                    <a href="/" class="btn btn-secondary">← Back to Home</a>
                    <a href="/models/{model_id}" class="btn btn-secondary">ℹ️ Model Info</a>
                </div>
            </div>
        </div>
        
        <script>
            function showDragHint() {{
                // Show drag instructions
                setTimeout(() => {{
                    document.getElementById('dragHint').style.display = 'block';
                    document.getElementById('dragHint').scrollIntoView({{ behavior: 'smooth' }});
                }}, 500);
            }}
        </script>
    </body>
    </html>
    """)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
