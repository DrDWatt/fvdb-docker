"""
GARField Service - 3D Asset Extraction using Group Anything with Radiance Fields
Provides API for extracting 3D assets from Gaussian Splat scenes based on click selection.
"""

import os
import io
import json
import logging
import uuid
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
from PIL import Image
import torch
from fastapi import FastAPI, File, UploadFile, Form, Query, HTTPException
from fastapi.responses import JSONResponse, Response, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
GARFIELD_PORT = int(os.environ.get("GARFIELD_PORT", "8006"))
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/app/models"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/app/outputs"))
CACHE_DIR = Path(os.environ.get("CACHE_DIR", "/app/cache"))

# Ensure directories exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Global state
device = None
sam_model = None
sam_processor = None
extraction_jobs = {}  # Track extraction jobs

app = FastAPI(
    title="GARField 3D Asset Extraction Service",
    description="Extract 3D assets from Gaussian Splat scenes using hierarchical grouping",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except:
    pass


def load_models():
    """Load SAM model for mask generation"""
    global device, sam_model, sam_processor
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    try:
        from transformers import SamModel, SamProcessor
        logger.info("Loading SAM model for mask generation...")
        sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
        sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
        sam_model.eval()
        logger.info("SAM model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to load SAM model: {e}")
        return False


def extract_mask_from_click(image: np.ndarray, x: int, y: int) -> Optional[np.ndarray]:
    """Generate a segmentation mask from a click point using SAM"""
    global sam_model, sam_processor, device
    
    if sam_model is None:
        load_models()
    
    if sam_model is None:
        return None
    
    try:
        pil_image = Image.fromarray(image)
        input_points = [[[x, y]]]
        
        inputs = sam_processor(
            pil_image,
            input_points=input_points,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = sam_model(**inputs)
        
        masks = sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu()
        )
        
        # Get best mask
        scores = outputs.iou_scores.cpu().numpy()[0][0]
        best_idx = np.argmax(scores)
        mask = masks[0][0][best_idx].numpy().astype(np.uint8)
        
        return mask
    except Exception as e:
        logger.error(f"Mask extraction error: {e}")
        return None


def extract_gaussians_by_mask(
    ply_path: Path,
    mask: np.ndarray,
    camera_params: Dict[str, Any],
    scale_level: float = 0.5
) -> Optional[Dict[str, Any]]:
    """
    Extract Gaussian splat points that correspond to the masked region.
    Uses projection and depth-based filtering to identify relevant 3D points.
    """
    try:
        from plyfile import PlyData
        
        # Load PLY file
        plydata = PlyData.read(str(ply_path))
        vertex = plydata['vertex']
        
        # Extract positions
        x = np.array(vertex['x'])
        y = np.array(vertex['y'])
        z = np.array(vertex['z'])
        positions = np.stack([x, y, z], axis=-1)
        
        # Get mask coordinates
        mask_coords = np.where(mask > 0)
        if len(mask_coords[0]) == 0:
            return None
        
        # Calculate mask bounding box for rough filtering
        mask_y_min, mask_y_max = mask_coords[0].min(), mask_coords[0].max()
        mask_x_min, mask_x_max = mask_coords[1].min(), mask_coords[1].max()
        
        # Project 3D points to 2D using camera parameters
        width = camera_params.get('width', 1024)
        height = camera_params.get('height', 768)
        fov = camera_params.get('fov', 60)
        
        # Simple perspective projection
        focal = width / (2 * np.tan(np.radians(fov / 2)))
        cx, cy = width / 2, height / 2
        
        # Apply camera transform (simplified)
        azimuth = np.radians(camera_params.get('azimuth', 0))
        elevation = np.radians(camera_params.get('elevation', 0))
        
        # Rotation matrices
        cos_az, sin_az = np.cos(azimuth), np.sin(azimuth)
        cos_el, sin_el = np.cos(elevation), np.sin(elevation)
        
        # Rotate points
        rotated = positions.copy()
        # Azimuth rotation (around Y axis)
        temp_x = rotated[:, 0] * cos_az + rotated[:, 2] * sin_az
        temp_z = -rotated[:, 0] * sin_az + rotated[:, 2] * cos_az
        rotated[:, 0] = temp_x
        rotated[:, 2] = temp_z
        # Elevation rotation (around X axis)
        temp_y = rotated[:, 1] * cos_el - rotated[:, 2] * sin_el
        temp_z = rotated[:, 1] * sin_el + rotated[:, 2] * cos_el
        rotated[:, 1] = temp_y
        rotated[:, 2] = temp_z
        
        # Project to 2D
        z_vals = rotated[:, 2]
        valid_z = z_vals > 0.1  # Only points in front of camera
        
        proj_x = (rotated[:, 0] / z_vals * focal + cx).astype(int)
        proj_y = (rotated[:, 1] / z_vals * focal + cy).astype(int)
        
        # Filter points that project into the mask
        in_bounds = (
            valid_z &
            (proj_x >= 0) & (proj_x < width) &
            (proj_y >= 0) & (proj_y < height)
        )
        
        # Check which points fall within the mask
        selected_indices = []
        for i in np.where(in_bounds)[0]:
            px, py = proj_x[i], proj_y[i]
            if mask[py, px] > 0:
                selected_indices.append(i)
        
        selected_indices = np.array(selected_indices)
        
        if len(selected_indices) == 0:
            return None
        
        # Apply scale-based filtering (hierarchical grouping simulation)
        # Lower scale = tighter grouping, higher scale = broader grouping
        if scale_level < 1.0:
            # Cluster selected points and keep dense regions
            from sklearn.cluster import DBSCAN
            selected_positions = positions[selected_indices]
            
            eps = 0.1 * (1.0 / max(scale_level, 0.1))
            clustering = DBSCAN(eps=eps, min_samples=5).fit(selected_positions)
            
            # Find the largest cluster
            labels = clustering.labels_
            unique_labels = set(labels) - {-1}
            if unique_labels:
                largest_cluster = max(unique_labels, key=lambda l: np.sum(labels == l))
                cluster_mask = labels == largest_cluster
                selected_indices = selected_indices[cluster_mask]
        
        # Extract all properties for selected gaussians
        extracted = {
            'num_gaussians': len(selected_indices),
            'indices': selected_indices.tolist(),
            'positions': positions[selected_indices].tolist(),
        }
        
        # Copy additional properties if available
        for prop in ['nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity',
                     'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']:
            if prop in vertex.data.dtype.names:
                extracted[prop] = np.array(vertex[prop])[selected_indices].tolist()
        
        return extracted
        
    except Exception as e:
        logger.error(f"Gaussian extraction error: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_extracted_ply(
    original_ply: Path,
    extracted_data: Dict[str, Any],
    output_path: Path
) -> bool:
    """Save extracted gaussians as a new PLY file"""
    try:
        from plyfile import PlyData, PlyElement
        
        # Load original to get structure
        original = PlyData.read(str(original_ply))
        vertex = original['vertex']
        
        indices = np.array(extracted_data['indices'])
        
        # Create new vertex data
        new_data = []
        for i in indices:
            new_data.append(vertex[i])
        
        new_vertex = np.array(new_data, dtype=vertex.data.dtype)
        new_element = PlyElement.describe(new_vertex, 'vertex')
        
        PlyData([new_element]).write(str(output_path))
        logger.info(f"Saved extracted PLY to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save extracted PLY: {e}")
        return False


@app.on_event("startup")
async def startup():
    """Load models on startup"""
    logger.info("Starting GARField service...")
    load_models()
    logger.info("GARField service ready")


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "garfield",
        "device": str(device) if device else "not initialized",
        "sam_loaded": sam_model is not None
    }


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>GARField 3D Asset Extraction</title>
        <style>
            body { font-family: Arial, sans-serif; background: #1a1a2e; color: #eee; padding: 40px; }
            h1 { color: #ff6b6b; }
            .endpoint { background: #16213e; padding: 15px; margin: 10px 0; border-radius: 8px; }
            code { background: #0f3460; padding: 3px 8px; border-radius: 4px; }
        </style>
    </head>
    <body>
        <h1>🎯 GARField 3D Asset Extraction Service</h1>
        <p>Extract 3D assets from Gaussian Splat scenes using hierarchical grouping.</p>
        
        <h2>Endpoints:</h2>
        <div class="endpoint">
            <strong>POST /extract</strong> - Extract 3D asset at click position
            <br><code>x, y, model_name, scale_level, camera_params</code>
        </div>
        <div class="endpoint">
            <strong>GET /jobs/{job_id}</strong> - Get extraction job status
        </div>
        <div class="endpoint">
            <strong>GET /download/{job_id}</strong> - Download extracted PLY
        </div>
        <div class="endpoint">
            <strong>GET /extractions</strong> - List all extractions
        </div>
    </body>
    </html>
    """


@app.post("/extract")
async def extract_asset(
    x: int = Form(...),
    y: int = Form(...),
    model_name: str = Form(...),
    scale_level: float = Form(0.5),
    azimuth: float = Form(0),
    elevation: float = Form(0),
    zoom: float = Form(1.0),
    width: int = Form(1024),
    height: int = Form(768),
    image: UploadFile = File(None)
):
    """
    Extract 3D asset from the scene at the clicked position.
    
    - x, y: Click coordinates
    - model_name: Name of the PLY model
    - scale_level: Grouping scale (0.1-2.0, lower = tighter selection)
    - camera params: Current view parameters
    - image: Optional rendered image for mask generation
    """
    job_id = str(uuid.uuid4())[:8]
    
    try:
        # Find the model
        model_path = None
        for ext in ['.ply', '']:
            test_path = MODEL_DIR / f"{model_name}{ext}"
            if test_path.exists():
                model_path = test_path
                break
        
        if not model_path or not model_path.exists():
            return JSONResponse({
                "error": f"Model not found: {model_name}",
                "job_id": job_id,
                "status": "failed"
            }, status_code=404)
        
        # Get or generate the rendered image for mask extraction
        if image:
            image_data = await image.read()
            pil_image = Image.open(io.BytesIO(image_data))
            img_array = np.array(pil_image)
        else:
            # Create a placeholder - in production, render from the viewer
            img_array = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Generate mask from click point
        mask = extract_mask_from_click(img_array, x, y)
        
        if mask is None:
            # Fallback: create a circular mask around click point
            mask = np.zeros((height, width), dtype=np.uint8)
            radius = int(50 * scale_level)
            yy, xx = np.ogrid[:height, :width]
            circle = (xx - x)**2 + (yy - y)**2 <= radius**2
            mask[circle] = 1
        
        # Extract gaussians
        camera_params = {
            'azimuth': azimuth,
            'elevation': elevation,
            'zoom': zoom,
            'width': width,
            'height': height,
            'fov': 60
        }
        
        extracted = extract_gaussians_by_mask(
            model_path, mask, camera_params, scale_level
        )
        
        if extracted is None or extracted['num_gaussians'] == 0:
            return JSONResponse({
                "job_id": job_id,
                "status": "no_selection",
                "message": "No gaussians found in selected region",
                "click": {"x": x, "y": y}
            })
        
        # Save extracted PLY
        output_path = OUTPUT_DIR / f"{job_id}_extracted.ply"
        save_success = save_extracted_ply(model_path, extracted, output_path)
        
        # Store job info
        extraction_jobs[job_id] = {
            "job_id": job_id,
            "status": "completed" if save_success else "failed",
            "model_name": model_name,
            "num_gaussians": extracted['num_gaussians'],
            "click": {"x": x, "y": y},
            "scale_level": scale_level,
            "output_file": str(output_path) if save_success else None,
            "timestamp": time.time()
        }
        
        return extraction_jobs[job_id]
        
    except Exception as e:
        logger.error(f"Extraction error: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "job_id": job_id,
            "status": "error",
            "error": str(e)
        }, status_code=500)


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get status of an extraction job"""
    if job_id in extraction_jobs:
        return extraction_jobs[job_id]
    return JSONResponse({"error": "Job not found"}, status_code=404)


@app.get("/download/{job_id}")
async def download_extracted(job_id: str):
    """Download extracted PLY file"""
    if job_id not in extraction_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = extraction_jobs[job_id]
    if job["status"] != "completed" or not job.get("output_file"):
        raise HTTPException(status_code=400, detail="Extraction not complete")
    
    output_path = Path(job["output_file"])
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found")
    
    return FileResponse(
        path=output_path,
        filename=f"{job['model_name']}_extracted_{job_id}.ply",
        media_type="application/octet-stream"
    )


@app.get("/extractions")
async def list_extractions():
    """List all extraction jobs"""
    return {
        "extractions": list(extraction_jobs.values()),
        "count": len(extraction_jobs)
    }


@app.delete("/extractions/{job_id}")
async def delete_extraction(job_id: str):
    """Delete an extraction job and its output"""
    if job_id not in extraction_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = extraction_jobs[job_id]
    if job.get("output_file"):
        output_path = Path(job["output_file"])
        if output_path.exists():
            output_path.unlink()
    
    del extraction_jobs[job_id]
    return {"status": "deleted", "job_id": job_id}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=GARFIELD_PORT)
