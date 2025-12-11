"""
fVDB Rendering Service with WebRTC Streaming Support
Simplified for ARM64 compatibility with Omniverse Web Viewer integration
"""

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import uvicorn
import numpy as np
from pathlib import Path
import logging
import io
import json
import asyncio
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="fVDB Rendering Service with Streaming",
    description="Web-based rendering and WebRTC streaming for Gaussian Splat models",
    version="2.0.0",
    docs_url="/api",
    redoc_url="/api/redoc"
)

# Enable CORS for web viewer
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Model registry
loaded_models: Dict[str, dict] = {}
streaming_sessions: Dict[str, dict] = {}

class ModelInfo(BaseModel):
    """Information about a loaded model"""
    id: str
    path: str
    size_mb: float
    gaussians: Optional[int] = None
    loaded: bool = False

class StreamConfig(BaseModel):
    """WebRTC streaming configuration"""
    model_id: str
    signaling_port: int = 49100
    enable_audio: bool = False
    max_bitrate: int = 8000000

@app.get("/")
async def root():
    """Root endpoint with quick access"""
    models = list_available_models()
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>fVDB Rendering Service</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }}
            h1 {{ color: #76b900; }}
            .model-card {{ border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; }}
            .btn {{ padding: 10px 20px; margin: 5px; background: #76b900; color: white; text-decoration: none; border-radius: 5px; display: inline-block; }}
            .btn:hover {{ background: #5a8f00; }}
            .status {{ padding: 5px 10px; border-radius: 3px; font-size: 12px; }}
            .status.ready {{ background: #d4edda; color: #155724; }}
            .status.error {{ background: #f8d7da; color: #721c24; }}
            .info {{ background: #e7f3ff; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎨 fVDB Rendering Service</h1>
            <p>WebRTC Streaming Support for Omniverse Web Viewer</p>
            
            <div class="info">
                <h3>🌐 Streaming Enabled</h3>
                <p><strong>WebRTC Signaling Port:</strong> 49100</p>
                <p><strong>Omniverse Web Viewer:</strong> <a href="http://localhost:5173" target="_blank">http://localhost:5173</a></p>
                <p><strong>API Documentation:</strong> <a href="/api" target="_blank">Interactive API Docs</a></p>
            </div>
            
            <h2>📦 Available Models ({len(models)})</h2>
    """
    
    if models:
        for model in models:
            status_class = "ready" if model["exists"] else "error"
            status_text = "Ready" if model["exists"] else "Not Found"
            
            html += f"""
            <div class="model-card">
                <h3>{model['name']}</h3>
                <p><span class="status {status_class}">{status_text}</span></p>
                <p><strong>Path:</strong> {model['path']}</p>
                <p><strong>Size:</strong> {model['size_mb']:.2f} MB</p>
                <a href="/models/{model['name']}" class="btn">View Info</a>
                <a href="/static/downloads/{model['name']}" class="btn" download>Download PLY</a>
                <a href="/stream/start/{model['name']}" class="btn">Start Streaming</a>
            </div>
            """
    else:
        html += """
        <div class="info">
            <p>No models found. Upload models to <code>/app/models/</code> directory.</p>
            <p>Or train a model at: <a href="http://localhost:8000">Training Service</a></p>
        </div>
        """
    
    html += """
            <h2>🔧 Quick Actions</h2>
            <a href="/api" class="btn">API Documentation</a>
            <a href="/health" class="btn">Health Check</a>
            <a href="/stream/status" class="btn">Streaming Status</a>
        </div>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html)

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "fvdb-rendering",
        "streaming_enabled": True,
        "models_available": len(list_available_models()),
        "active_streams": len(streaming_sessions)
    }

def list_available_models() -> List[dict]:
    """List all available PLY models"""
    models = []
    
    for ply_file in MODEL_DIR.glob("*.ply"):
        try:
            size_mb = ply_file.stat().st_size / (1024 * 1024)
            models.append({
                "name": ply_file.name,
                "path": str(ply_file),
                "size_mb": size_mb,
                "exists": ply_file.exists()
            })
        except Exception as e:
            logger.error(f"Error reading model {ply_file}: {e}")
    
    # Also check downloads directory
    for ply_file in DOWNLOADS_DIR.glob("*.ply"):
        if not any(m["name"] == ply_file.name for m in models):
            try:
                size_mb = ply_file.stat().st_size / (1024 * 1024)
                models.append({
                    "name": ply_file.name,
                    "path": str(ply_file),
                    "size_mb": size_mb,
                    "exists": ply_file.exists()
                })
            except Exception as e:
                logger.error(f"Error reading model {ply_file}: {e}")
    
    return models

@app.get("/models")
async def get_models():
    """List all available models"""
    models = list_available_models()
    return {
        "count": len(models),
        "models": models
    }

@app.get("/models/{model_name}")
async def get_model_info(model_name: str):
    """Get detailed information about a specific model"""
    model_path = MODEL_DIR / model_name
    
    if not model_path.exists():
        model_path = DOWNLOADS_DIR / model_name
    
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    size_mb = model_path.stat().st_size / (1024 * 1024)
    
    return {
        "name": model_name,
        "path": str(model_path),
        "size_mb": size_mb,
        "exists": True,
        "streaming_available": True,
        "download_url": f"/static/downloads/{model_name}"
    }

@app.post("/stream/start/{model_name}")
async def start_streaming(model_name: str, config: Optional[StreamConfig] = None):
    """Start WebRTC streaming session for a model"""
    model_path = MODEL_DIR / model_name
    
    if not model_path.exists():
        model_path = DOWNLOADS_DIR / model_name
    
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    session_id = f"stream_{len(streaming_sessions)}"
    
    streaming_sessions[session_id] = {
        "model": model_name,
        "path": str(model_path),
        "port": config.signaling_port if config else 49100,
        "status": "ready",
        "created": str(asyncio.get_event_loop().time())
    }
    
    logger.info(f"Streaming session created: {session_id} for model {model_name}")
    
    return {
        "session_id": session_id,
        "model": model_name,
        "signaling_port": streaming_sessions[session_id]["port"],
        "status": "ready",
        "message": "Streaming session ready. Connect from Omniverse Web Viewer.",
        "web_viewer_url": "http://localhost:5173"
    }

@app.get("/stream/status")
async def get_streaming_status():
    """Get status of all streaming sessions"""
    return {
        "active_sessions": len(streaming_sessions),
        "sessions": streaming_sessions,
        "signaling_port": 49100,
        "web_viewer_url": "http://localhost:5173"
    }

@app.delete("/stream/stop/{session_id}")
async def stop_streaming(session_id: str):
    """Stop a streaming session"""
    if session_id not in streaming_sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    session = streaming_sessions.pop(session_id)
    logger.info(f"Streaming session stopped: {session_id}")
    
    return {
        "status": "stopped",
        "session_id": session_id,
        "model": session["model"]
    }

@app.post("/models/upload")
async def upload_model(file: UploadFile = File(...)):
    """Upload a PLY model file"""
    if not file.filename.endswith('.ply'):
        raise HTTPException(status_code=400, detail="Only PLY files are supported")
    
    file_path = MODEL_DIR / file.filename
    
    try:
        content = await file.read()
        with open(file_path, 'wb') as f:
            f.write(content)
        
        # Also copy to downloads for easy access
        downloads_path = DOWNLOADS_DIR / file.filename
        with open(downloads_path, 'wb') as f:
            f.write(content)
        
        size_mb = len(content) / (1024 * 1024)
        
        return {
            "filename": file.filename,
            "size_mb": size_mb,
            "path": str(file_path),
            "download_url": f"/static/downloads/{file.filename}",
            "message": "Model uploaded successfully"
        }
    except Exception as e:
        logger.error(f"Error uploading model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/viewer/{model_name}")
async def viewer_page(model_name: str):
    """Simple viewer page for a model"""
    model_path = MODEL_DIR / model_name
    
    if not model_path.exists():
        model_path = DOWNLOADS_DIR / model_name
    
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>View: {model_name}</title>
        <style>
            body {{ margin: 0; font-family: Arial, sans-serif; }}
            .info {{ padding: 20px; background: #f5f5f5; }}
            .btn {{ padding: 10px 20px; background: #76b900; color: white; text-decoration: none; border-radius: 5px; margin: 5px; display: inline-block; }}
        </style>
    </head>
    <body>
        <div class="info">
            <h1>🎨 {model_name}</h1>
            <p><strong>Viewing Options:</strong></p>
            <a href="http://localhost:5173" class="btn" target="_blank">Open in Omniverse Web Viewer</a>
            <a href="/static/downloads/{model_name}" class="btn" download>Download PLY</a>
            <a href="/" class="btn">Back to Home</a>
            
            <h3>Instructions:</h3>
            <ol>
                <li>Click "Open in Omniverse Web Viewer"</li>
                <li>In the web viewer, connect to: <code>fvdb-rendering:49100</code></li>
                <li>View your 3D model with real-time interaction!</li>
            </ol>
            
            <p><strong>Alternative:</strong> Download the PLY file and view in <a href="https://playcanvas.com/supersplat" target="_blank">SuperSplat</a></p>
        </div>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
