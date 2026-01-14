"""
Minimal Rendering Service - ARM64 Compatible
Serves PLY files for download with RAG metadata support
"""

from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PLY Model Service",
    description="PLY file server with RAG metadata for object labeling",
    version="2.0.0"
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
DOWNLOADS_DIR = BASE_DIR / "downloads"
METADATA_DIR = BASE_DIR / "metadata"

for dir_path in [MODEL_DIR, DOWNLOADS_DIR, METADATA_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# RAG metadata storage
model_metadata = {}

@app.get("/")
async def root():
    """Root endpoint"""
    models = list(MODEL_DIR.glob("*.ply")) + list(DOWNLOADS_DIR.glob("*.ply"))
    model_names = list(set([m.name for m in models]))
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>PLY Model Service</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            h1 {{ color: #76b900; }}
            .model-card {{ border: 1px solid #ddd; padding: 20px; margin: 15px 0; border-radius: 8px; background: #fafafa; }}
            .model-card h3 {{ margin: 0 0 10px 0; color: #333; }}
            .btn {{ padding: 10px 20px; margin: 5px; background: #76b900; color: white; text-decoration: none; border-radius: 5px; display: inline-block; font-weight: bold; }}
            .btn:hover {{ background: #5a8f00; }}
            .info {{ background: #e7f3ff; padding: 15px; border-radius: 5px; margin: 20px 0; border-left: 4px solid #2196F3; }}
            .success {{ background: #d4edda; padding: 15px; border-radius: 5px; margin: 20px 0; border-left: 4px solid #28a745; }}
            .size {{ color: #666; font-size: 14px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎨 PLY Model Service</h1>
            <p>Download Gaussian Splat models for viewing in external applications</p>
            
            <div class="success">
                <h3>✅ Service Running</h3>
                <p><strong>Status:</strong> Ready to serve PLY files</p>
                <p><strong>Models Available:</strong> {len(model_names)}</p>
            </div>
            
            <div class="info">
                <h3>📖 How to View Models</h3>
                <ol>
                    <li>Download a PLY file below</li>
                    <li>Open <a href="https://playcanvas.com/supersplat/editor" target="_blank"><strong>SuperSplat Editor</strong></a></li>
                    <li>Drag and drop the PLY file into the viewer</li>
                    <li>Use mouse to rotate, scroll to zoom</li>
                </ol>
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
            <div class="model-card">
                <h3>📄 {name}</h3>
                <p class="size">Size: {size_mb:.2f} MB</p>
                <a href="/download/{name}" class="btn">📥 Download</a>
            </div>
            """
    else:
        html += """
        <div class="info">
            <p>No models found. Train a model first using the training service.</p>
        </div>
        """
    
    html += """
            <h2>🔗 Quick Links</h2>
            <p>
                <a href="https://playcanvas.com/supersplat/editor" target="_blank" class="btn">🎮 Open SuperSplat</a>
                <a href="/health" class="btn" style="background: #6c757d;">💚 Health Check</a>
            </p>
        </div>
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
        "service": "ply-model-service",
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

@app.post("/upload")
async def upload_model(file: UploadFile = File(...)):
    """Upload PLY file"""
    if not file.filename.endswith('.ply'):
        raise HTTPException(status_code=400, detail="Only .ply files allowed")
    
    content = await file.read()
    model_path = MODEL_DIR / file.filename
    
    with open(model_path, 'wb') as f:
        f.write(content)
    
    size_mb = len(content) / (1024 * 1024)
    
    return {
        "filename": file.filename,
        "size_mb": round(size_mb, 2),
        "download_url": f"/download/{file.filename}",
        "message": "Upload successful"
    }


# ===== RAG Metadata Endpoints =====

def load_metadata(model_name: str) -> dict:
    """Load metadata from file"""
    meta_path = METADATA_DIR / f"{model_name}.json"
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            return json.load(f)
    return {}

def save_metadata(model_name: str, metadata: dict):
    """Save metadata to file"""
    meta_path = METADATA_DIR / f"{model_name}.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)


@app.get("/metadata/{model_name}")
async def get_metadata(model_name: str):
    """Get RAG metadata for a model"""
    metadata = load_metadata(model_name)
    return {
        "model": model_name,
        "metadata": metadata,
        "has_metadata": bool(metadata)
    }


@app.post("/metadata/{model_name}")
async def set_metadata(
    model_name: str,
    title: str = Form(None),
    description: str = Form(None),
    objects: str = Form(None),
    labels: str = Form(None),
    details: str = Form(None)
):
    """Set RAG metadata for a model
    
    - title: Model title (e.g., "Tesla Model S 70D")
    - description: Full description
    - objects: Comma-separated list of objects in scene (e.g., "car, Tesla, sedan, electric vehicle")
    - labels: JSON object mapping segment descriptions to labels
    - details: Additional context for labeling
    """
    metadata = load_metadata(model_name)
    
    if title:
        metadata["title"] = title
    if description:
        metadata["description"] = description
    if objects:
        metadata["objects"] = [o.strip() for o in objects.split(",")]
    if labels:
        try:
            metadata["labels"] = json.loads(labels)
        except:
            metadata["labels"] = labels
    if details:
        metadata["details"] = details
    
    save_metadata(model_name, metadata)
    logger.info(f"Updated metadata for {model_name}: {metadata}")
    
    return {
        "model": model_name,
        "metadata": metadata,
        "message": "Metadata updated"
    }


@app.delete("/metadata/{model_name}")
async def delete_metadata(model_name: str):
    """Delete RAG metadata for a model"""
    meta_path = METADATA_DIR / f"{model_name}.json"
    if meta_path.exists():
        meta_path.unlink()
        return {"message": f"Metadata for {model_name} deleted"}
    raise HTTPException(status_code=404, detail="Metadata not found")


@app.get("/metadata")
async def list_all_metadata():
    """List all models with metadata"""
    all_metadata = {}
    for meta_file in METADATA_DIR.glob("*.json"):
        model_name = meta_file.stem
        with open(meta_file, 'r') as f:
            all_metadata[model_name] = json.load(f)
    return {"models": all_metadata, "count": len(all_metadata)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
