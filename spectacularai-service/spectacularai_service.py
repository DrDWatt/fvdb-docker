"""
SpectacularAI Gaussian Splatting Service

A FastAPI service that provides SLAM-based 3D Gaussian Splatting training
using the SpectacularAI SDK. This service processes recordings from
SpectacularAI-compatible devices and trains 3DGS models.

Workflow:
1. Upload recording (from Spectacular Rec app or supported devices)
2. Process with sai-cli to extract camera poses and images
3. Train Gaussian Splatting model using Nerfstudio
4. Export PLY file for use with rendering/viewing services
"""

import os
import uuid
import json
import asyncio
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from enum import Enum

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import aiofiles

# Configuration
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", "/app/uploads"))
PROCESSING_DIR = Path(os.environ.get("PROCESSING_DIR", "/app/processing"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/app/outputs"))
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/app/models"))

# Ensure directories exist
for d in [UPLOAD_DIR, PROCESSING_DIR, OUTPUT_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Job tracking
jobs: Dict[str, Dict[str, Any]] = {}


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    TRAINING = "training"
    EXPORTING = "exporting"
    COMPLETED = "completed"
    FAILED = "failed"


class SceneSize(str, Enum):
    SMALL = "small"      # Table-sized objects (key_frame_distance=0.05)
    MEDIUM = "medium"    # Room corners (key_frame_distance=0.10)
    LARGE = "large"      # Full rooms (key_frame_distance=0.15)


class ProcessingConfig(BaseModel):
    """Configuration for SpectacularAI processing"""
    scene_size: SceneSize = Field(default=SceneSize.MEDIUM, description="Size of the scanned scene")
    fast_mode: bool = Field(default=False, description="Trade quality for speed")
    preview_3d: bool = Field(default=False, description="Generate 3D preview during processing")


class TrainingConfig(BaseModel):
    """Configuration for Gaussian Splatting training"""
    max_iterations: int = Field(default=30000, ge=1000, le=100000, description="Maximum training iterations")
    model_name: Optional[str] = Field(default=None, description="Custom model name (auto-generated if not provided)")


class JobResponse(BaseModel):
    """Response model for job status"""
    job_id: str
    status: JobStatus
    progress: float = 0.0
    message: str = ""
    created_at: str
    updated_at: str
    output_path: Optional[str] = None
    model_path: Optional[str] = None


class UploadResponse(BaseModel):
    """Response model for upload"""
    job_id: str
    message: str
    status: JobStatus


# FastAPI app with Swagger documentation
app = FastAPI(
    title="SpectacularAI Gaussian Splatting Service",
    description="""
## SpectacularAI 3D Gaussian Splatting API

This service provides SLAM-based 3D Gaussian Splatting training using the SpectacularAI SDK.

### Workflow
1. **Upload** a recording from Spectacular Rec app or supported devices
2. **Process** the recording to extract camera poses and images
3. **Train** a Gaussian Splatting model
4. **Export** PLY file for rendering/viewing

### Supported Devices
- iPhone/iPad with Spectacular Rec app
- Android devices with Spectacular Rec app
- OAK-D cameras
- Intel RealSense cameras
- Azure Kinect
- Orbbec cameras

### Output Format
The service exports PLY files compatible with:
- fVDB Rendering Service
- fVDB Viewer Service
- gsplat.js web viewers
- SuperSplat editor
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)


@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "spectacularai-service",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/", tags=["System"])
async def root():
    """Service information"""
    return {
        "service": "SpectacularAI Gaussian Splatting Service",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.post("/upload", response_model=UploadResponse, tags=["Upload"])
async def upload_recording(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Recording file (zip or folder)"),
    scene_size: SceneSize = Query(default=SceneSize.MEDIUM, description="Size of scanned scene"),
    fast_mode: bool = Query(default=False, description="Fast processing mode"),
    max_iterations: int = Query(default=30000, ge=1000, le=100000, description="Training iterations"),
    model_name: Optional[str] = Query(default=None, description="Custom model name")
):
    """
    Upload a SpectacularAI recording for processing.
    
    Accepts recordings from:
    - Spectacular Rec iOS/Android app (zip files)
    - OAK-D, RealSense, Kinect recordings
    
    The recording will be processed through the full pipeline:
    1. Extract and validate recording
    2. Process with sai-cli (SLAM + pose estimation)
    3. Train Gaussian Splatting model
    4. Export PLY for rendering
    """
    job_id = str(uuid.uuid4())
    timestamp = datetime.utcnow()
    
    # Create job directory
    job_dir = PROCESSING_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    
    # Save uploaded file
    upload_path = job_dir / file.filename
    async with aiofiles.open(upload_path, 'wb') as f:
        content = await file.read()
        await f.write(content)
    
    # Initialize job tracking
    jobs[job_id] = {
        "job_id": job_id,
        "status": JobStatus.PENDING,
        "progress": 0.0,
        "message": "Recording uploaded, processing queued",
        "created_at": timestamp.isoformat(),
        "updated_at": timestamp.isoformat(),
        "upload_path": str(upload_path),
        "config": {
            "scene_size": scene_size,
            "fast_mode": fast_mode,
            "max_iterations": max_iterations,
            "model_name": model_name or f"spectacularai-{job_id[:8]}"
        }
    }
    
    # Start processing in background
    background_tasks.add_task(process_recording, job_id)
    
    return UploadResponse(
        job_id=job_id,
        message="Recording uploaded successfully. Processing started.",
        status=JobStatus.PENDING
    )


async def process_recording(job_id: str):
    """Background task to process recording through full pipeline"""
    job = jobs.get(job_id)
    if not job:
        return
    
    try:
        job_dir = PROCESSING_DIR / job_id
        config = job["config"]
        upload_path = Path(job["upload_path"])
        
        # Step 1: Extract if zip
        job["status"] = JobStatus.PROCESSING
        job["message"] = "Extracting recording..."
        job["progress"] = 0.1
        job["updated_at"] = datetime.utcnow().isoformat()
        
        input_path = job_dir / "input"
        if upload_path.suffix == ".zip":
            shutil.unpack_archive(upload_path, input_path)
        else:
            input_path = upload_path
        
        # Step 2: Process with sai-cli
        job["message"] = "Processing with SpectacularAI SLAM..."
        job["progress"] = 0.2
        job["updated_at"] = datetime.utcnow().isoformat()
        
        nerfstudio_path = job_dir / "nerfstudio"
        
        # Determine key_frame_distance based on scene size
        kfd_map = {
            SceneSize.SMALL: 0.05,
            SceneSize.MEDIUM: 0.10,
            SceneSize.LARGE: 0.15
        }
        key_frame_distance = kfd_map[config["scene_size"]]
        
        # Build sai-cli command
        sai_cmd = [
            "sai-cli", "process",
            str(input_path),
            f"--key_frame_distance={key_frame_distance}",
            str(nerfstudio_path)
        ]
        if config["fast_mode"]:
            sai_cmd.insert(2, "--fast")
        
        # Run sai-cli process
        process = await asyncio.create_subprocess_exec(
            *sai_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            raise Exception(f"sai-cli processing failed: {stderr.decode()}")
        
        job["progress"] = 0.4
        job["updated_at"] = datetime.utcnow().isoformat()
        
        # Step 3: Train Gaussian Splatting
        job["status"] = JobStatus.TRAINING
        job["message"] = "Training Gaussian Splatting model..."
        job["progress"] = 0.5
        job["updated_at"] = datetime.utcnow().isoformat()
        
        model_name = config["model_name"]
        max_iterations = config["max_iterations"]
        
        # Run ns-train gaussian-splatting
        train_cmd = [
            "ns-train", "gaussian-splatting",
            "--data", str(nerfstudio_path),
            "--max-num-iterations", str(max_iterations),
            "--output-dir", str(job_dir / "training")
        ]
        
        process = await asyncio.create_subprocess_exec(
            *train_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            raise Exception(f"Training failed: {stderr.decode()}")
        
        job["progress"] = 0.8
        job["updated_at"] = datetime.utcnow().isoformat()
        
        # Step 4: Export PLY
        job["status"] = JobStatus.EXPORTING
        job["message"] = "Exporting Gaussian Splat PLY..."
        job["progress"] = 0.9
        job["updated_at"] = datetime.utcnow().isoformat()
        
        # Find the config.yml from training
        training_dir = job_dir / "training"
        config_files = list(training_dir.rglob("config.yml"))
        if not config_files:
            raise Exception("Training config not found")
        
        export_dir = OUTPUT_DIR / job_id
        export_dir.mkdir(parents=True, exist_ok=True)
        
        export_cmd = [
            "ns-export", "gaussian-splat",
            "--load-config", str(config_files[0]),
            "--output-dir", str(export_dir)
        ]
        
        process = await asyncio.create_subprocess_exec(
            *export_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            raise Exception(f"Export failed: {stderr.decode()}")
        
        # Copy PLY to models directory for rendering service
        ply_file = export_dir / "splat.ply"
        if not ply_file.exists():
            ply_file = export_dir / "point_cloud.ply"
        
        if ply_file.exists():
            model_path = MODEL_DIR / f"{model_name}.ply"
            shutil.copy(ply_file, model_path)
            
            # Create metadata for rendering service
            metadata = {
                "name": model_name,
                "source": "spectacularai",
                "job_id": job_id,
                "created_at": datetime.utcnow().isoformat(),
                "scene_size": config["scene_size"],
                "iterations": max_iterations,
                "ply_path": str(model_path)
            }
            metadata_path = MODEL_DIR / f"{model_name}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            job["model_path"] = str(model_path)
        
        # Complete
        job["status"] = JobStatus.COMPLETED
        job["message"] = "Processing complete. Model ready for rendering."
        job["progress"] = 1.0
        job["output_path"] = str(export_dir)
        job["updated_at"] = datetime.utcnow().isoformat()
        
    except Exception as e:
        job["status"] = JobStatus.FAILED
        job["message"] = f"Processing failed: {str(e)}"
        job["updated_at"] = datetime.utcnow().isoformat()


@app.get("/jobs/{job_id}", response_model=JobResponse, tags=["Jobs"])
async def get_job_status(job_id: str):
    """Get the status of a processing job"""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobResponse(
        job_id=job["job_id"],
        status=job["status"],
        progress=job["progress"],
        message=job["message"],
        created_at=job["created_at"],
        updated_at=job["updated_at"],
        output_path=job.get("output_path"),
        model_path=job.get("model_path")
    )


@app.get("/jobs", response_model=List[JobResponse], tags=["Jobs"])
async def list_jobs(
    status: Optional[JobStatus] = Query(default=None, description="Filter by status"),
    limit: int = Query(default=50, ge=1, le=100, description="Maximum results")
):
    """List all processing jobs"""
    result = []
    for job in list(jobs.values())[-limit:]:
        if status is None or job["status"] == status:
            result.append(JobResponse(
                job_id=job["job_id"],
                status=job["status"],
                progress=job["progress"],
                message=job["message"],
                created_at=job["created_at"],
                updated_at=job["updated_at"],
                output_path=job.get("output_path"),
                model_path=job.get("model_path")
            ))
    return result


@app.delete("/jobs/{job_id}", tags=["Jobs"])
async def delete_job(job_id: str, cleanup_files: bool = Query(default=True)):
    """Delete a job and optionally clean up files"""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if cleanup_files:
        # Clean up processing directory
        job_dir = PROCESSING_DIR / job_id
        if job_dir.exists():
            shutil.rmtree(job_dir)
        
        # Clean up output directory
        output_dir = OUTPUT_DIR / job_id
        if output_dir.exists():
            shutil.rmtree(output_dir)
    
    del jobs[job_id]
    return {"message": f"Job {job_id} deleted", "files_cleaned": cleanup_files}


@app.get("/models", tags=["Models"])
async def list_models():
    """List all trained models available for rendering"""
    models = []
    for ply_file in MODEL_DIR.glob("*.ply"):
        metadata_file = MODEL_DIR / f"{ply_file.stem}_metadata.json"
        metadata = {}
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
        
        models.append({
            "name": ply_file.stem,
            "path": str(ply_file),
            "size_bytes": ply_file.stat().st_size,
            "metadata": metadata
        })
    return {"models": models, "count": len(models)}


@app.get("/models/{model_name}/download", tags=["Models"])
async def download_model(model_name: str):
    """Download a trained model PLY file"""
    ply_path = MODEL_DIR / f"{model_name}.ply"
    if not ply_path.exists():
        raise HTTPException(status_code=404, detail="Model not found")
    
    return FileResponse(
        ply_path,
        media_type="application/octet-stream",
        filename=f"{model_name}.ply"
    )


@app.post("/models/{model_name}/notify-rendering", tags=["Integration"])
async def notify_rendering_service(model_name: str):
    """
    Notify the rendering service about a new model.
    This enables the model for rendering without manual intervention.
    """
    import httpx
    
    ply_path = MODEL_DIR / f"{model_name}.ply"
    if not ply_path.exists():
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Get rendering service URL from environment
    rendering_url = os.environ.get("RENDERING_SERVICE_URL", "http://fvdb-rendering:8001")
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{rendering_url}/models/reload",
                timeout=30.0
            )
            return {
                "message": f"Rendering service notified about {model_name}",
                "rendering_response": response.json() if response.status_code == 200 else None
            }
    except Exception as e:
        return {
            "message": f"Model ready but could not notify rendering service: {str(e)}",
            "model_path": str(ply_path)
        }


# Mount static files for web UI
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except:
    pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
