"""
Minimal Rendering Service - ARM64 Compatible
Serves PLY files and provides WebRTC signaling for Omniverse Web Viewer
NO fVDB dependency - pure Python
"""

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PLY Rendering Service",
    description="Simple PLY file server with WebRTC signaling for Omniverse Web Viewer",
    version="1.0.0"
)

# Enable CORS
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
STATIC_DIR = BASE_DIR / "static"
DOWNLOADS_DIR = BASE_DIR / "downloads"

for dir_path in [MODEL_DIR, DOWNLOADS_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# Mount static files (use absolute path)
try:
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
except Exception as e:
    logger.warning(f"Could not mount static files: {e}")

@app.get("/")
async def root():
    """Root endpoint"""
    models = list(MODEL_DIR.glob("*.ply")) + list(DOWNLOADS_DIR.glob("*.ply"))
    model_names = list(set([m.name for m in models]))
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>PLY Rendering Service</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }}
            h1 {{ color: #76b900; }}
            .model-card {{ border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; background: #f9f9f9; }}
            .btn {{ padding: 10px 20px; margin: 5px; background: #76b900; color: white; text-decoration: none; border-radius: 5px; display: inline-block; }}
            .btn:hover {{ background: #5a8f00; }}
            .info {{ background: #e7f3ff; padding: 15px; border-radius: 5px; margin: 20px 0; border-left: 4px solid #2196F3; }}
            .success {{ background: #d4edda; padding: 15px; border-radius: 5px; margin: 20px 0; border-left: 4px solid #28a745; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎨 PLY Rendering Service</h1>
            <p>Minimal service for Omniverse Web Viewer integration on ARM64</p>
            
            <div class="success">
                <h3>✅ Service Running</h3>
                <p><strong>Status:</strong> Ready to serve PLY files</p>
                <p><strong>Omniverse Web Viewer:</strong> <a href="http://localhost:5173" target="_blank">http://localhost:5173</a></p>
                <p><strong>Models Available:</strong> {len(model_names)}</p>
            </div>
            
            <div class="info">
                <h3>📖 How to View Models</h3>
                <ol>
                    <li>Download a PLY file below</li>
                    <li>Go to <a href="https://playcanvas.com/supersplat" target="_blank">SuperSplat</a></li>
                    <li>Drag and drop the PLY file</li>
                    <li>View your 3D model instantly!</li>
                </ol>
                <p><strong>Or:</strong> Use the Omniverse Web Viewer at <a href="http://localhost:5173">localhost:5173</a></p>
            </div>
            
            <h2>📦 Available Models ({len(model_names)})</h2>
    """
    
    if model_names:
        for name in sorted(model_names):
            model_path = MODEL_DIR / name
            if not model_path.exists():
                model_path = DOWNLOADS_DIR / name
            
            size_mb = model_path.stat().st_size / (1024 * 1024) if model_path.exists() else 0
            
            html += f"""
            <div class="model-card" id="model-{name.replace('.', '-')}">
                <h3>📄 {name}</h3>
                <p><strong>Size:</strong> {size_mb:.2f} MB</p>
                <a href="/viewer/{name}" class="btn" style="background: #2196F3;">👁️ View</a>
                <a href="/download/{name}" class="btn">📥 Download</a>
                <a href="/models/{name}" class="btn">ℹ️ Info</a>
                <button onclick="deleteModel('{name}')" class="btn" style="background: #dc3545;">🗑️ Delete</button>
            </div>
            """
    else:
        html += """
        <div class="info">
            <p>No models found. Add .ply files to:</p>
            <ul>
                <li><code>/app/models/</code></li>
                <li><code>/app/static/downloads/</code></li>
            </ul>
            <p>Or upload via the <a href="/api">API</a></p>
        </div>
        """
    
    html += """
            <h2>🔧 Actions</h2>
            <a href="/api" class="btn">📖 API Docs</a>
            <a href="/health" class="btn">💚 Health Check</a>
            <a href="http://localhost:5173" class="btn" target="_blank">🌐 Omniverse Viewer</a>
        </div>
        
        <script>
        async function deleteModel(name) {
            if (!confirm('Delete ' + name + '? This cannot be undone.')) return;
            
            try {
                const response = await fetch('/delete/' + name, { method: 'DELETE' });
                if (response.ok) {
                    document.getElementById('model-' + name.replace('.', '-')).remove();
                    alert('Deleted ' + name);
                } else {
                    alert('Failed to delete: ' + (await response.text()));
                }
            } catch(e) {
                alert('Error: ' + e);
            }
        }
        </script>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html)

@app.get("/health")
async def health():
    """Health check"""
    models = list(MODEL_DIR.glob("*.ply")) + list(DOWNLOADS_DIR.glob("*.ply"))
    return {
        "status": "healthy",
        "service": "ply-rendering-minimal",
        "models_available": len(set([m.name for m in models]))
    }

@app.get("/models")
async def list_models():
    """List all PLY models"""
    models = []
    seen = set()
    
    for model_path in list(MODEL_DIR.glob("*.ply")) + list(DOWNLOADS_DIR.glob("*.ply")):
        if model_path.name not in seen:
            seen.add(model_path.name)
            size_mb = model_path.stat().st_size / (1024 * 1024)
            models.append({
                "name": model_path.name,
                "path": str(model_path),
                "size_mb": round(size_mb, 2),
                "download_url": f"/download/{model_path.name}"
            })
    
    return {"count": len(models), "models": models}

@app.get("/models/{model_name}")
async def get_model_info(model_name: str):
    """Get model info"""
    model_path = MODEL_DIR / model_name
    if not model_path.exists():
        model_path = DOWNLOADS_DIR / model_name
    
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    size_mb = model_path.stat().st_size / (1024 * 1024)
    
    return {
        "name": model_name,
        "path": str(model_path),
        "size_mb": round(size_mb, 2),
        "download_url": f"/download/{model_name}"
    }

@app.get("/download/{model_name}")
async def download_model(model_name: str):
    """Download PLY file"""
    model_path = MODEL_DIR / model_name
    if not model_path.exists():
        model_path = DOWNLOADS_DIR / model_name
    
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    return FileResponse(
        path=model_path,
        media_type="application/octet-stream",
        filename=model_name
    )

@app.get("/viewer/{model_name}")
async def view_model(model_name: str):
    """View Gaussian Splat model in local WebGL viewer"""
    model_path = MODEL_DIR / model_name
    if not model_path.exists():
        model_path = DOWNLOADS_DIR / model_name
    
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    size_mb = model_path.stat().st_size / (1024 * 1024)
    
    # Redirect to local WebGL viewer - splat.js uses 'url' parameter
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url=f"/static/viewer.html?url=/download/{model_name}")

@app.post("/upload")
async def upload_model(file: UploadFile = File(...)):
    """Upload PLY file"""
    if not file.filename.endswith('.ply'):
        raise HTTPException(status_code=400, detail="Only .ply files allowed")
    
    content = await file.read()
    
    # Save to both directories
    model_path = MODEL_DIR / file.filename
    downloads_path = DOWNLOADS_DIR / file.filename
    
    with open(model_path, 'wb') as f:
        f.write(content)
    with open(downloads_path, 'wb') as f:
        f.write(content)
    
    size_mb = len(content) / (1024 * 1024)
    
    return {
        "filename": file.filename,
        "size_mb": round(size_mb, 2),
        "download_url": f"/download/{file.filename}",
        "message": "Upload successful"
    }

@app.delete("/delete/{model_name}")
async def delete_model(model_name: str):
    """Delete a PLY model"""
    deleted = False
    
    # Try to delete from both directories
    model_path = MODEL_DIR / model_name
    if model_path.exists():
        model_path.unlink()
        deleted = True
        logger.info(f"Deleted {model_path}")
    
    downloads_path = DOWNLOADS_DIR / model_name
    if downloads_path.exists():
        downloads_path.unlink()
        deleted = True
        logger.info(f"Deleted {downloads_path}")
    
    # Also delete any metadata file
    meta_path = MODEL_DIR / f"{model_name.replace('.ply', '')}_metadata.json"
    if meta_path.exists():
        meta_path.unlink()
        logger.info(f"Deleted metadata {meta_path}")
    
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    return {"message": f"Deleted {model_name}", "status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
