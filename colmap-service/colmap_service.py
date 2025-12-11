"""
COLMAP Processing Service
Handles structure-from-motion processing for image datasets
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
from pathlib import Path
import logging
import shutil
import zipfile
import subprocess
import json
import os
from datetime import datetime
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="COLMAP Processing Service",
    description="Structure-from-Motion processing for 3D reconstruction",
    version="1.0.0",
    docs_url="/api",
    redoc_url="/api/redoc"
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
UPLOAD_DIR = BASE_DIR / "uploads"
PROCESSING_DIR = BASE_DIR / "processing"
OUTPUT_DIR = BASE_DIR / "outputs"
TEMP_DIR = BASE_DIR / "temp"
WORKFLOW_STATE_FILE = BASE_DIR / "workflow_state.json"

for dir_path in [UPLOAD_DIR, PROCESSING_DIR, OUTPUT_DIR, TEMP_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# File-based workflow state (persists across requests)
def load_workflows() -> Dict:
    """Load workflows from file"""
    if WORKFLOW_STATE_FILE.exists():
        try:
            with open(WORKFLOW_STATE_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_workflows(workflows: Dict):
    """Save workflows to file"""
    with open(WORKFLOW_STATE_FILE, 'w') as f:
        json.dump(workflows, f)

def update_workflow(workflow_id: str, updates: Dict):
    """Update a specific workflow"""
    workflows = load_workflows()
    if workflow_id in workflows:
        workflows[workflow_id].update(updates)
    else:
        workflows[workflow_id] = updates
    save_workflows(workflows)

# Job tracking
processing_jobs = {}

class ProcessRequest(BaseModel):
    """Request to process images with COLMAP"""
    dataset_id: str
    quality: str = "medium"  # low, medium, high, extreme
    matcher: str = "exhaustive"  # exhaustive, sequential, vocab_tree
    camera_model: str = "SIMPLE_PINHOLE"  # SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL

class JobStatus(BaseModel):
    """Status of a processing job"""
    job_id: str
    status: str
    progress: float
    message: str
    dataset_id: Optional[str] = None
    num_images: Optional[int] = None
    num_points: Optional[int] = None
    error: Optional[str] = None

@app.get("/")
async def root():
    """Service information"""
    return {
        "service": "COLMAP Processing Service",
        "version": "1.0.0",
        "endpoints": {
            "upload": "/upload",
            "extract_video": "/video/extract",
            "process": "/process",
            "status": "/jobs/{job_id}",
            "download": "/download/{job_id}",
            "api_docs": "/api"
        }
    }

@app.get("/health")
async def health():
    """Health check"""
    # Check COLMAP is available
    try:
        result = subprocess.run(
            ["colmap", "-h"],
            capture_output=True,
            timeout=5
        )
        colmap_available = result.returncode == 0
    except Exception:
        colmap_available = False
    
    return {
        "status": "healthy",
        "colmap_available": colmap_available,
        "active_jobs": len([j for j in processing_jobs.values() if j["status"] == "processing"])
    }

@app.post("/upload")
async def upload_images(
    file: UploadFile = File(...),
    dataset_name: Optional[str] = None
):
    """Upload ZIP file containing images"""
    if not file.filename.endswith('.zip'):
        raise HTTPException(400, "Only ZIP files are supported")
    
    dataset_id = dataset_name or f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    dataset_path = PROCESSING_DIR / dataset_id
    dataset_path.mkdir(exist_ok=True, parents=True)
    
    # Save uploaded file
    zip_path = UPLOAD_DIR / f"{dataset_id}.zip"
    with open(zip_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    # Extract images
    images_dir = dataset_path / "images"
    images_dir.mkdir(exist_ok=True)
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Extract only image files to images directory
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
            for item in zip_ref.namelist():
                if Path(item).suffix.lower() in image_extensions:
                    filename = Path(item).name
                    source = zip_ref.open(item)
                    target = open(images_dir / filename, "wb")
                    with source, target:
                        shutil.copyfileobj(source, target)
        
        # Count images
        image_files = list(images_dir.glob("*"))
        num_images = len([f for f in image_files if f.suffix.lower() in image_extensions])
        
        if num_images == 0:
            raise HTTPException(400, "No images found in ZIP file")
        
        return {
            "dataset_id": dataset_id,
            "num_images": num_images,
            "status": "uploaded",
            "message": f"Uploaded {num_images} images. Ready for COLMAP processing."
        }
    
    except Exception as e:
        shutil.rmtree(dataset_path, ignore_errors=True)
        raise HTTPException(500, f"Failed to process ZIP file: {str(e)}")

@app.post("/video/extract")
async def extract_video_frames(
    file: UploadFile = File(...),
    fps: float = 2.0,
    dataset_name: Optional[str] = None
):
    """Extract frames from video file (supports MP4, MOV, AVI, etc.)"""
    video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv'}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in video_extensions:
        raise HTTPException(400, f"Unsupported video format. Supported: {', '.join(video_extensions)}")
    
    dataset_id = dataset_name or f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    dataset_path = PROCESSING_DIR / dataset_id
    images_dir = dataset_path / "images"
    images_dir.mkdir(exist_ok=True, parents=True)
    
    # Save uploaded video
    video_path = TEMP_DIR / f"{dataset_id}{file_ext}"
    with open(video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    try:
        # Extract frames using ffmpeg
        output_pattern = str(images_dir / "frame_%04d.jpg")
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-vf", f"fps={fps}",
            "-qscale:v", "2",  # High quality
            output_pattern
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode != 0:
            raise Exception(f"FFmpeg failed: {result.stderr}")
        
        # Count extracted frames
        frames = list(images_dir.glob("frame_*.jpg"))
        num_frames = len(frames)
        
        if num_frames == 0:
            raise Exception("No frames were extracted from video")
        
        # Clean up video file
        video_path.unlink()
        
        return {
            "dataset_id": dataset_id,
            "num_images": num_frames,
            "fps": fps,
            "status": "extracted",
            "message": f"Extracted {num_frames} frames at {fps} FPS. Ready for COLMAP processing."
        }
    
    except subprocess.TimeoutExpired:
        raise HTTPException(500, "Video processing timed out")
    except Exception as e:
        shutil.rmtree(dataset_path, ignore_errors=True)
        video_path.unlink(missing_ok=True)
        raise HTTPException(500, f"Failed to extract frames: {str(e)}")

@app.post("/process")
async def process_with_colmap(
    request: ProcessRequest,
    background_tasks: BackgroundTasks
):
    """Start COLMAP processing on uploaded dataset"""
    dataset_path = PROCESSING_DIR / request.dataset_id
    images_dir = dataset_path / "images"
    
    if not images_dir.exists():
        raise HTTPException(404, f"Dataset '{request.dataset_id}' not found. Upload images first.")
    
    image_files = list(images_dir.glob("*"))
    if len(image_files) == 0:
        raise HTTPException(400, "No images found in dataset")
    
    job_id = f"colmap_{request.dataset_id}_{datetime.now().strftime('%H%M%S')}"
    
    processing_jobs[job_id] = {
        "job_id": job_id,
        "dataset_id": request.dataset_id,
        "status": "queued",
        "progress": 0.0,
        "message": "Queued for processing",
        "num_images": len(image_files),
        "started_at": datetime.now().isoformat()
    }
    
    # Start processing in background
    background_tasks.add_task(
        run_colmap_processing,
        job_id,
        dataset_path,
        request.quality,
        request.matcher,
        request.camera_model
    )
    
    return {
        "job_id": job_id,
        "status": "queued",
        "message": "COLMAP processing started",
        "status_url": f"/jobs/{job_id}"
    }

async def run_colmap_processing(
    job_id: str,
    dataset_path: Path,
    quality: str,
    matcher: str,
    camera_model: str
):
    """Run COLMAP processing pipeline"""
    try:
        processing_jobs[job_id]["status"] = "processing"
        processing_jobs[job_id]["progress"] = 0.1
        processing_jobs[job_id]["message"] = "Extracting features..."
        
        images_dir = dataset_path / "images"
        database_path = dataset_path / "database.db"
        sparse_dir = dataset_path / "sparse" / "0"
        sparse_dir.mkdir(exist_ok=True, parents=True)
        
        # Quality settings
        quality_settings = {
            "low": {"max_image_size": 1600, "max_num_features": 4000},
            "medium": {"max_image_size": 2400, "max_num_features": 8000},
            "high": {"max_image_size": 3200, "max_num_features": 16000},
            "extreme": {"max_image_size": 4800, "max_num_features": 32000}
        }
        settings = quality_settings.get(quality, quality_settings["medium"])
        
        # Step 1: Feature extraction
        logger.info(f"[{job_id}] Starting feature extraction")
        
        # Set environment for headless Qt
        env = os.environ.copy()
        env['QT_QPA_PLATFORM'] = 'offscreen'
        
        cmd_extract = [
            "colmap", "feature_extractor",
            "--database_path", str(database_path),
            "--image_path", str(images_dir),
            "--ImageReader.single_camera", "1",
            "--ImageReader.camera_model", camera_model,
            "--SiftExtraction.max_image_size", str(settings["max_image_size"]),
            "--SiftExtraction.max_num_features", str(settings["max_num_features"]),
            "--SiftExtraction.use_gpu", "0"  # Disable GPU to avoid OpenGL context issues
        ]
        
        result = subprocess.run(cmd_extract, capture_output=True, text=True, timeout=1800, env=env)
        if result.returncode != 0:
            raise Exception(f"Feature extraction failed: {result.stderr}")
        
        processing_jobs[job_id]["progress"] = 0.4
        processing_jobs[job_id]["message"] = "Matching features..."
        
        # 2. Feature matching
        logger.info(f"[{job_id}] Starting feature matching ({matcher})")
        if matcher == "exhaustive":
            cmd = [
                "colmap", "exhaustive_matcher",
                "--database_path", str(database_path),
                "--SiftMatching.use_gpu", "0"  # Disable GPU
            ]
        elif matcher == "sequential":
            cmd = [
                "colmap", "sequential_matcher",
                "--database_path", str(database_path),
                "--SequentialMatching.overlap", "10",
                "--SiftMatching.use_gpu", "0"  # Disable GPU
            ]
        else:
            raise Exception(f"Unsupported matcher: {matcher}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=env)
        if result.returncode != 0:
            raise Exception(f"Feature matching failed: {result.stderr}")
        
        processing_jobs[job_id]["progress"] = 0.7
        processing_jobs[job_id]["message"] = "Running sparse reconstruction..."
        
        # 3. Sparse reconstruction (mapper)
        logger.info(f"[{job_id}] Starting sparse reconstruction")
        cmd = [
            "colmap", "mapper",
            "--database_path", str(database_path),
            "--image_path", str(images_dir),
            "--output_path", str(sparse_dir.parent)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, env=env)
        if result.returncode != 0:
            raise Exception(f"Sparse reconstruction failed: {result.stderr}")
        
        processing_jobs[job_id]["progress"] = 0.9
        processing_jobs[job_id]["message"] = "Finalizing..."
        
        # Verify output
        cameras_file = sparse_dir / "cameras.bin"
        images_file = sparse_dir / "images.bin"
        points_file = sparse_dir / "points3D.bin"
        
        if not (cameras_file.exists() and images_file.exists() and points_file.exists()):
            raise Exception("COLMAP reconstruction incomplete - missing output files")
        
        # Get reconstruction stats
        stats_cmd = ["colmap", "model_analyzer", "--path", str(sparse_dir)]
        stats_result = subprocess.run(stats_cmd, capture_output=True, text=True)
        
        # Extract number of points (basic parsing)
        num_points = 0
        if "points3D" in stats_result.stdout:
            for line in stats_result.stdout.split('\n'):
                if "Registered images" in line:
                    # Parse reconstruction stats if available
                    pass
        
        # Create ZIP of complete dataset
        output_zip = OUTPUT_DIR / f"{job_id}.zip"
        shutil.make_archive(
            str(output_zip.with_suffix('')),
            'zip',
            dataset_path
        )
        
        processing_jobs[job_id]["status"] = "completed"
        processing_jobs[job_id]["progress"] = 1.0
        processing_jobs[job_id]["message"] = "COLMAP processing complete"
        processing_jobs[job_id]["output_file"] = f"{job_id}.zip"
        processing_jobs[job_id]["download_url"] = f"/download/{job_id}"
        processing_jobs[job_id]["completed_at"] = datetime.now().isoformat()
        
        logger.info(f"[{job_id}] Processing completed successfully")
        
    except subprocess.TimeoutExpired:
        processing_jobs[job_id]["status"] = "failed"
        processing_jobs[job_id]["error"] = "Processing timed out"
        logger.error(f"[{job_id}] Timeout")
    except Exception as e:
        processing_jobs[job_id]["status"] = "failed"
        processing_jobs[job_id]["error"] = str(e)
        logger.error(f"[{job_id}] Error: {e}")

@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get status of a processing job"""
    if job_id not in processing_jobs:
        raise HTTPException(404, "Job not found")
    
    return processing_jobs[job_id]

@app.get("/jobs")
async def list_jobs():
    """List all processing jobs"""
    return {"jobs": list(processing_jobs.values())}

@app.get("/download/{job_id}")
async def download_result(job_id: str):
    """Download processed dataset with COLMAP data"""
    if job_id not in processing_jobs:
        raise HTTPException(404, "Job not found")
    
    job = processing_jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(400, "Job not completed yet")
    
    output_file = OUTPUT_DIR / f"{job_id}.zip"
    if not output_file.exists():
        raise HTTPException(404, "Output file not found")
    
    return FileResponse(
        output_file,
        media_type="application/zip",
        filename=f"{job['dataset_id']}_colmap.zip"
    )

@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its data"""
    if job_id not in processing_jobs:
        raise HTTPException(404, "Job not found")
    
    job = processing_jobs[job_id]
    dataset_id = job.get("dataset_id")
    
    # Clean up files
    if dataset_id:
        dataset_path = PROCESSING_DIR / dataset_id
        shutil.rmtree(dataset_path, ignore_errors=True)
    
    output_file = OUTPUT_DIR / f"{job_id}.zip"
    output_file.unlink(missing_ok=True)
    
    del processing_jobs[job_id]
    
    return {"message": "Job deleted successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)

# Combined workflow endpoint
@app.post("/workflow/video-to-model")
async def workflow_video_to_model(
    file: UploadFile = File(...),
    dataset_id: str = Form(...),
    fps: float = Form(1.0),
    camera_model: str = Form("SIMPLE_RADIAL"),
    matcher: str = Form("exhaustive"),
    num_training_steps: int = Form(30000),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Complete workflow: Upload video -> Extract frames -> Run COLMAP -> Train Gaussian Splat
    This is a fire-and-forget endpoint. Use GET /workflow/status/{job_id} to monitor.
    """
    import httpx
    
    workflow_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    
    # Store workflow status using file-based storage
    update_workflow(workflow_id, {
        "workflow_id": workflow_id,
        "status": "uploading",
        "progress": 0.0,
        "current_step": "Uploading video",
        "dataset_id": dataset_id,
        "started_at": datetime.now().isoformat(),
        "colmap_job_id": None,
        "training_job_id": None,
        "error": None
    })
    
    async def run_workflow():
        try:
            # Step 1: Save uploaded file
            update_workflow(workflow_id, {"current_step": "Saving video file", "progress": 0.1})
            
            dataset_dir = UPLOAD_DIR / dataset_id
            dataset_dir.mkdir(exist_ok=True, parents=True)
            
            video_path = dataset_dir / file.filename
            with open(video_path, "wb") as f:
                content = await file.read()
                f.write(content)
            
            logger.info(f"[{workflow_id}] Saved video to {video_path}")
            
            # Step 2: Extract frames and process with COLMAP
            update_workflow(workflow_id, {"current_step": "Extracting frames"})
            update_workflow(workflow_id, {"progress": 0.2})
            
            # Create job for COLMAP processing
            job_id = f"colmap_{dataset_id}_{datetime.now().strftime('%H%M%S')}"
            processing_jobs[job_id] = {
                "job_id": job_id,
                "dataset_id": dataset_id,
                "status": "processing",
                "progress": 0.0,
                "message": "Starting COLMAP processing",
                "started_at": datetime.now().isoformat()
            }
            
            update_workflow(workflow_id, {"colmap_job_id": job_id})
            
            # Run COLMAP processing synchronously
            output_dir = OUTPUT_DIR / dataset_id
            output_dir.mkdir(exist_ok=True, parents=True)
            
            images_dir = output_dir / "images"
            images_dir.mkdir(exist_ok=True, parents=True)
            
            # Extract frames
            update_workflow(workflow_id, {"current_step": "Extracting frames from video"})
            update_workflow(workflow_id, {"progress": 0.25})
            
            result = subprocess.run([
                "ffmpeg", "-i", str(video_path),
                "-vf", f"fps={fps}",
                "-q:v", "2",
                str(images_dir / "frame_%04d.jpg")
            ], capture_output=True, text=True, timeout=600)
            
            if result.returncode != 0:
                raise Exception(f"Frame extraction failed: {result.stderr}")
            
            num_images = len(list(images_dir.glob("*.jpg")))
            logger.info(f"[{workflow_id}] Extracted {num_images} frames")
            
            processing_jobs[job_id]["progress"] = 0.3
            processing_jobs[job_id]["message"] = f"Extracted {num_images} frames"
            update_workflow(workflow_id, {"progress": 0.3})
            
            # Run COLMAP
            update_workflow(workflow_id, {"current_step": "Running COLMAP reconstruction"})
            
            sparse_dir = output_dir / "sparse" / "0"
            sparse_dir.mkdir(exist_ok=True, parents=True)
            database_path = output_dir / "database.db"
            
            # Feature extraction
            processing_jobs[job_id]["message"] = "Extracting features..."
            update_workflow(workflow_id, {"progress": 0.4})
            
            env = os.environ.copy()
            env['QT_QPA_PLATFORM'] = 'offscreen'
            
            cmd_extract = [
                "colmap", "feature_extractor",
                "--database_path", str(database_path),
                "--image_path", str(images_dir),
                "--ImageReader.single_camera", "1",
                "--ImageReader.camera_model", camera_model,
                "--SiftExtraction.max_image_size", "2048",
                "--SiftExtraction.max_num_features", "16384",
                "--SiftExtraction.use_gpu", "1"
            ]
            
            result = subprocess.run(cmd_extract, capture_output=True, text=True, timeout=1800, env=env)
            if result.returncode != 0:
                raise Exception(f"Feature extraction failed: {result.stderr}")
            
            processing_jobs[job_id]["progress"] = 0.5
            processing_jobs[job_id]["message"] = "Matching features..."
            update_workflow(workflow_id, {"progress": 0.5})
            
            # Feature matching
            if matcher == "exhaustive":
                cmd_match = [
                    "colmap", "exhaustive_matcher",
                    "--database_path", str(database_path),
                    "--SiftMatching.use_gpu", "1"
                ]
            else:
                cmd_match = [
                    "colmap", "sequential_matcher",
                    "--database_path", str(database_path),
                    "--SequentialMatching.overlap", "10",
                    "--SiftMatching.use_gpu", "1"
                ]
            
            result = subprocess.run(cmd_match, capture_output=True, text=True, timeout=1800, env=env)
            if result.returncode != 0:
                raise Exception(f"Feature matching failed: {result.stderr}")
            
            processing_jobs[job_id]["progress"] = 0.7
            processing_jobs[job_id]["message"] = "Running sparse reconstruction..."
            update_workflow(workflow_id, {"progress": 0.6})
            
            # Sparse reconstruction
            cmd_mapper = [
                "colmap", "mapper",
                "--database_path", str(database_path),
                "--image_path", str(images_dir),
                "--output_path", str(sparse_dir.parent)
            ]
            
            result = subprocess.run(cmd_mapper, capture_output=True, text=True, timeout=3600, env=env)
            if result.returncode != 0:
                raise Exception(f"Sparse reconstruction failed: {result.stderr}")
            
            processing_jobs[job_id]["status"] = "completed"
            processing_jobs[job_id]["progress"] = 1.0
            processing_jobs[job_id]["message"] = "COLMAP processing complete"
            processing_jobs[job_id]["num_images"] = num_images
            processing_jobs[job_id]["completed_at"] = datetime.now().isoformat()
            
            update_workflow(workflow_id, {"current_step": "COLMAP complete, starting training"})
            update_workflow(workflow_id, {"progress": 0.7})
            
            logger.info(f"[{workflow_id}] COLMAP processing complete")
            
            # Step 3: Trigger training on training service
            update_workflow(workflow_id, {"current_step": "Starting Gaussian Splat training"})
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                try:
                    response = await client.post(
                        "http://fvdb-training-gpu:8000/train",
                        json={
                            "dataset_id": dataset_id,
                            "num_training_steps": num_training_steps,
                            "output_name": f"{dataset_id}_model"
                        }
                    )
                    
                    if response.status_code == 200:
                        train_data = response.json()
                        update_workflow(workflow_id, {"training_job_id": train_data.get("job_id")})
                        update_workflow(workflow_id, {"status": "training"})
                        update_workflow(workflow_id, {"progress": 0.75})
                        update_workflow(workflow_id, {"current_step": "Training in progress (check training service)"})
                        logger.info(f"[{workflow_id}] Training started: {train_data.get('job_id')}")
                    else:
                        raise Exception(f"Training service returned {response.status_code}: {response.text}")
                        
                except Exception as e:
                    logger.error(f"[{workflow_id}] Failed to start training: {e}")
                    update_workflow(workflow_id, {"status": "completed_colmap_only"})
                    update_workflow(workflow_id, {"progress": 0.7})
                    update_workflow(workflow_id, {"current_step": "COLMAP complete, training failed to start"})
                    update_workflow(workflow_id, {"error": f"Training failed: {str(e)}"})
                    update_workflow(workflow_id, {"message": "COLMAP complete. Manually start training at http://localhost:8000/api"})
            
        except Exception as e:
            logger.error(f"[{workflow_id}] Workflow failed: {e}")
            update_workflow(workflow_id, {"status": "failed"})
            update_workflow(workflow_id, {"error": str(e)})
            update_workflow(workflow_id, {"current_step": f"Failed: {str(e)}"})
            
            if job_id and job_id in processing_jobs:
                processing_jobs[job_id]["status"] = "failed"
                processing_jobs[job_id]["message"] = str(e)
    
    # Start workflow in background
    background_tasks.add_task(run_workflow)
    
    return {
        "workflow_id": workflow_id,
        "status": "started",
        "message": "Workflow initiated. Monitor progress at GET /workflow/status/{workflow_id}",
        "dataset_id": dataset_id,
        "monitor_url": f"/workflow/status/{workflow_id}",
        "estimated_time_minutes": "35-45 minutes"
    }

@app.get("/workflow/status/{workflow_id}")
async def get_workflow_status(workflow_id: str):
    """Get status of a complete workflow"""
    workflows = load_workflows()
    
    if workflow_id not in workflows:
        raise HTTPException(404, f"Workflow {workflow_id} not found")
    
    workflow = workflows[workflow_id]
    
    # If training started, check training service for updates
    if workflow.get("training_job_id"):
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get("http://fvdb-training-gpu:8000/jobs")
                if response.status_code == 200:
                    jobs_data = response.json()
                    train_job = next((j for j in jobs_data.get("jobs", []) if j["job_id"] == workflow["training_job_id"]), None)
                    if train_job:
                        if train_job["status"] == "completed":
                            workflow["status"] = "completed"
                            workflow["progress"] = 1.0
                            workflow["current_step"] = "Workflow complete! Model ready."
                            workflow["completed_at"] = train_job.get("completed_at")
                            workflow["output_files"] = train_job.get("output_files", [])
                        elif train_job["status"] == "failed":
                            workflow["status"] = "training_failed"
                            workflow["error"] = train_job.get("message")
                        else:
                            # Training in progress
                            train_progress = train_job.get("progress", 0)
                            # COLMAP is 0-0.7, training is 0.7-1.0
                            workflow["progress"] = 0.7 + (train_progress * 0.3)
                            workflow["current_step"] = f"Training: {train_job.get('message', 'In progress')}"
        except Exception as e:
            logger.warning(f"Could not check training status: {e}")
    
    return workflow

@app.get("/workflow/list")
async def list_workflows():
    """List all workflows"""
    workflows = load_workflows()
    return {"workflows": list(workflows.values())}

@app.delete("/workflow/{workflow_id}")
async def delete_workflow(workflow_id: str):
    """Delete a specific workflow"""
    workflows = load_workflows()
    if workflow_id not in workflows:
        raise HTTPException(404, f"Workflow {workflow_id} not found")
    
    del workflows[workflow_id]
    save_workflows(workflows)
    return {"message": f"Workflow {workflow_id} deleted"}

@app.delete("/workflow/clear/all")
async def clear_all_workflows():
    """Clear all workflows"""
    workflows = load_workflows()
    count = len(workflows)
    save_workflows({})
    return {"message": f"Cleared {count} workflows"}

@app.get("/datasets")
async def list_datasets():
    """List all datasets with their sizes"""
    datasets = []
    for path in OUTPUT_DIR.iterdir():
        if path.is_dir():
            # Calculate size
            size_bytes = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
            size_mb = size_bytes / (1024 * 1024)
            datasets.append({
                "name": path.name,
                "size_mb": round(size_mb, 2),
                "path": str(path)
            })
    return {"datasets": sorted(datasets, key=lambda x: x["name"])}

@app.delete("/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """Delete a dataset and its files"""
    dataset_path = OUTPUT_DIR / dataset_id
    upload_path = UPLOAD_DIR / dataset_id
    
    deleted = []
    if dataset_path.exists():
        shutil.rmtree(dataset_path)
        deleted.append(f"outputs/{dataset_id}")
    if upload_path.exists():
        shutil.rmtree(upload_path)
        deleted.append(f"uploads/{dataset_id}")
    
    if not deleted:
        raise HTTPException(404, f"Dataset {dataset_id} not found")
    
    return {"message": f"Deleted {dataset_id}", "deleted_paths": deleted}

@app.delete("/datasets/clear/all")
async def delete_all_datasets():
    """Delete ALL datasets - USE WITH CAUTION"""
    deleted_count = 0
    total_size_mb = 0
    
    # Delete from outputs
    for path in list(OUTPUT_DIR.iterdir()):
        if path.is_dir() and not path.name.startswith('.'):
            size_bytes = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
            total_size_mb += size_bytes / (1024 * 1024)
            shutil.rmtree(path)
            deleted_count += 1
    
    # Delete from uploads
    for path in list(UPLOAD_DIR.iterdir()):
        if path.is_dir() and not path.name.startswith('.'):
            shutil.rmtree(path)
    
    return {
        "message": f"Deleted {deleted_count} datasets",
        "freed_mb": round(total_size_mb, 2)
    }

@app.post("/workflow/photos-to-model")
async def workflow_photos_to_model(
    files: List[UploadFile] = File(...),
    dataset_id: str = Form(...),
    camera_model: str = Form("SIMPLE_RADIAL"),
    matcher: str = Form("exhaustive"),
    num_training_steps: int = Form(30000),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Complete workflow: Upload photos -> Run COLMAP -> Train Gaussian Splat
    Upload multiple JPG/PNG images OR a ZIP file containing images.
    """
    import httpx
    
    workflow_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    
    # Store workflow status using file-based storage
    update_workflow(workflow_id, {
        "workflow_id": workflow_id,
        "status": "uploading",
        "progress": 0.0,
        "current_step": "Uploading photos",
        "dataset_id": dataset_id,
        "started_at": datetime.now().isoformat(),
        "colmap_job_id": None,
        "training_job_id": None,
        "error": None
    })
    
    async def run_photos_workflow():
        try:
            # Step 1: Save uploaded photos
            update_workflow(workflow_id, {"current_step": "Saving photos"})
            update_workflow(workflow_id, {"progress": 0.1})
            
            output_dir = OUTPUT_DIR / dataset_id
            output_dir.mkdir(exist_ok=True, parents=True)
            
            images_dir = output_dir / "images"
            images_dir.mkdir(exist_ok=True, parents=True)
            
            num_images = 0
            for i, file in enumerate(files):
                filename_lower = file.filename.lower()
                
                # Handle ZIP files
                if filename_lower.endswith('.zip'):
                    update_workflow(workflow_id, {"current_step": "Extracting ZIP file"})
                    content = await file.read()
                    zip_path = output_dir / "upload.zip"
                    with open(zip_path, "wb") as f:
                        f.write(content)
                    
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        for zip_info in zip_ref.filelist:
                            if zip_info.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                                # Extract and rename
                                extracted = zip_ref.read(zip_info.filename)
                                img_path = images_dir / f"image_{num_images:04d}.jpg"
                                with open(img_path, "wb") as img_file:
                                    img_file.write(extracted)
                                num_images += 1
                    
                    zip_path.unlink()  # Clean up ZIP
                    logger.info(f"[{workflow_id}] Extracted {num_images} images from ZIP")
                
                # Handle individual images
                elif filename_lower.endswith(('.jpg', '.jpeg', '.png')):
                    content = await file.read()
                    img_path = images_dir / f"image_{num_images:04d}.jpg"
                    with open(img_path, "wb") as f:
                        f.write(content)
                    num_images += 1
            
            logger.info(f"[{workflow_id}] Saved {num_images} photos")
            update_workflow(workflow_id, {"progress": 0.2})
            update_workflow(workflow_id, {"current_step": f"Saved {num_images} photos"})
            
            if num_images < 3:
                raise Exception("Need at least 3 images for reconstruction")
            
            # Step 2: Run COLMAP
            job_id = f"colmap_{dataset_id}_{datetime.now().strftime('%H%M%S')}"
            processing_jobs[job_id] = {
                "job_id": job_id,
                "dataset_id": dataset_id,
                "status": "processing",
                "progress": 0.0,
                "message": "Starting COLMAP processing",
                "started_at": datetime.now().isoformat()
            }
            
            update_workflow(workflow_id, {"colmap_job_id": job_id})
            update_workflow(workflow_id, {"current_step": "Running COLMAP reconstruction"})
            update_workflow(workflow_id, {"progress": 0.3})
            
            sparse_dir = output_dir / "sparse" / "0"
            sparse_dir.mkdir(exist_ok=True, parents=True)
            database_path = output_dir / "database.db"
            
            env = os.environ.copy()
            env['QT_QPA_PLATFORM'] = 'offscreen'
            
            # Feature extraction
            processing_jobs[job_id]["message"] = "Extracting features..."
            update_workflow(workflow_id, {"progress": 0.35})
            
            cmd_extract = [
                "colmap", "feature_extractor",
                "--database_path", str(database_path),
                "--image_path", str(images_dir),
                "--ImageReader.single_camera", "1",
                "--ImageReader.camera_model", camera_model,
                "--SiftExtraction.max_image_size", "2048",
                "--SiftExtraction.max_num_features", "16384",
                "--SiftExtraction.use_gpu", "1"
            ]
            
            result = subprocess.run(cmd_extract, capture_output=True, text=True, timeout=1800, env=env)
            if result.returncode != 0:
                raise Exception(f"Feature extraction failed: {result.stderr}")
            
            processing_jobs[job_id]["progress"] = 0.5
            processing_jobs[job_id]["message"] = "Matching features..."
            update_workflow(workflow_id, {"progress": 0.5})
            
            # Feature matching
            if matcher == "exhaustive":
                cmd_match = [
                    "colmap", "exhaustive_matcher",
                    "--database_path", str(database_path),
                    "--SiftMatching.use_gpu", "1"
                ]
            else:
                cmd_match = [
                    "colmap", "sequential_matcher",
                    "--database_path", str(database_path),
                    "--SequentialMatching.overlap", "10",
                    "--SiftMatching.use_gpu", "1"
                ]
            
            result = subprocess.run(cmd_match, capture_output=True, text=True, timeout=1800, env=env)
            if result.returncode != 0:
                raise Exception(f"Feature matching failed: {result.stderr}")
            
            processing_jobs[job_id]["progress"] = 0.7
            processing_jobs[job_id]["message"] = "Running sparse reconstruction..."
            update_workflow(workflow_id, {"progress": 0.6})
            
            # Sparse reconstruction
            cmd_mapper = [
                "colmap", "mapper",
                "--database_path", str(database_path),
                "--image_path", str(images_dir),
                "--output_path", str(sparse_dir.parent)
            ]
            
            result = subprocess.run(cmd_mapper, capture_output=True, text=True, timeout=3600, env=env)
            if result.returncode != 0:
                raise Exception(f"Sparse reconstruction failed: {result.stderr}")
            
            processing_jobs[job_id]["status"] = "completed"
            processing_jobs[job_id]["progress"] = 1.0
            processing_jobs[job_id]["message"] = "COLMAP processing complete"
            processing_jobs[job_id]["num_images"] = num_images
            processing_jobs[job_id]["completed_at"] = datetime.now().isoformat()
            
            update_workflow(workflow_id, {"current_step": "COLMAP complete, starting training"})
            update_workflow(workflow_id, {"progress": 0.7})
            
            logger.info(f"[{workflow_id}] COLMAP processing complete")
            
            # Step 3: Trigger training
            update_workflow(workflow_id, {"current_step": "Starting Gaussian Splat training"})
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                try:
                    response = await client.post(
                        "http://fvdb-training-gpu:8000/train",
                        json={
                            "dataset_id": dataset_id,
                            "num_training_steps": num_training_steps,
                            "output_name": f"{dataset_id}_model"
                        }
                    )
                    
                    if response.status_code == 200:
                        train_data = response.json()
                        update_workflow(workflow_id, {"training_job_id": train_data.get("job_id")})
                        update_workflow(workflow_id, {"status": "training"})
                        update_workflow(workflow_id, {"progress": 0.75})
                        update_workflow(workflow_id, {"current_step": "Training in progress"})
                        logger.info(f"[{workflow_id}] Training started: {train_data.get('job_id')}")
                    else:
                        raise Exception(f"Training service returned {response.status_code}")
                        
                except Exception as e:
                    logger.error(f"[{workflow_id}] Failed to start training: {e}")
                    update_workflow(workflow_id, {"status": "completed_colmap_only"})
                    update_workflow(workflow_id, {"progress": 0.7})
                    update_workflow(workflow_id, {"current_step": "COLMAP complete, training failed to start"})
                    update_workflow(workflow_id, {"error": f"Training failed: {str(e)}"})
            
        except Exception as e:
            logger.error(f"[{workflow_id}] Workflow failed: {e}")
            update_workflow(workflow_id, {"status": "failed"})
            update_workflow(workflow_id, {"error": str(e)})
            update_workflow(workflow_id, {"current_step": f"Failed: {str(e)}"})
    
    background_tasks.add_task(run_photos_workflow)
    
    return {
        "workflow_id": workflow_id,
        "status": "started",
        "message": "Photos workflow initiated. Monitor at GET /workflow/status/{workflow_id}",
        "dataset_id": dataset_id,
        "num_files": len(files)
    }

