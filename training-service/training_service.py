"""
fVDB Reality Capture Training Service
FastAPI-based REST API for Gaussian Splat training
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, HttpUrl
from typing import Optional, List
import uvicorn
import os
import shutil
import zipfile
import requests
import torch
import logging
from pathlib import Path
import json
from datetime import datetime
import asyncio
import sys
import ssl
import urllib.request
import subprocess

# Disable SSL verification for downloads (training environment)
ssl._create_default_https_context = ssl._create_unverified_context

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add current directory to path for imports
sys.path.insert(0, '/app')

try:
    from extract_frames import extract_frames_from_video, get_video_info, recommend_extraction_params
    VIDEO_EXTRACTION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Video extraction not available: {e}")
    VIDEO_EXTRACTION_AVAILABLE = False

# Initialize FastAPI
app = FastAPI(
    title="fVDB Reality Capture Training Service",
    description="REST API for training Gaussian Splat models from images",
    version="1.0.0",
    docs_url="/api",  # Swagger UI at /api
    redoc_url="/api/redoc"
)

# Directories
BASE_DIR = Path("/app")
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

# Create directories
for dir_path in [UPLOAD_DIR, OUTPUT_DIR, DATA_DIR, MODEL_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# Training jobs store (in-memory cache, synced with state files)
training_jobs = {}
training_processes = {}  # Track running subprocesses

# State file helpers
JOBS_STATE_DIR = Path("/app/outputs")

def get_job_state_file(job_id: str) -> Path:
    """Get state file path for a job"""
    return JOBS_STATE_DIR / f"{job_id}_state.json"

def load_job_state(job_id: str) -> dict:
    """Load job state from file"""
    state_file = get_job_state_file(job_id)
    if state_file.exists():
        try:
            with open(state_file, 'r') as f:
                return json.load(f)
        except:
            pass
    return training_jobs.get(job_id, {})

def save_job_state(job_id: str, state: dict):
    """Save job state to file"""
    state_file = get_job_state_file(job_id)
    with open(state_file, 'w') as f:
        json.dump(state, f)
    training_jobs[job_id] = state

def sync_all_jobs():
    """Sync all job states from files"""
    for state_file in JOBS_STATE_DIR.glob("*_state.json"):
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
                if "job_id" in state:
                    training_jobs[state["job_id"]] = state
        except:
            pass

# Models
class TrainingRequest(BaseModel):
    """Request to start training"""
    dataset_id: str
    num_training_steps: Optional[int] = 62200
    output_name: Optional[str] = None

class DatasetUploadURL(BaseModel):
    """Request to upload dataset from URL"""
    url: HttpUrl
    dataset_name: Optional[str] = None

class EndToEndWorkflow(BaseModel):
    """End-to-end workflow: upload, train, and prepare for rendering"""
    url: Optional[HttpUrl] = None
    dataset_name: Optional[str] = None
    num_steps: int = 1000
    output_name: Optional[str] = None
    auto_start: bool = True

class JobStatus(BaseModel):
    """Training job status"""
    job_id: str
    status: str
    progress: float
    message: str
    created_at: str
    completed_at: Optional[str] = None
    output_files: Optional[List[str]] = None

# Helper functions
def extract_zip(zip_path: Path, extract_to: Path):
    """Extract zip file and handle nested single-directory structures"""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    # Check if extraction created a single top-level directory
    # If so, flatten it (e.g., south-building/* -> *)
    extracted_items = list(extract_to.iterdir())
    if len(extracted_items) == 1 and extracted_items[0].is_dir():
        single_dir = extracted_items[0]
        logger.info(f"Flattening nested directory: {single_dir.name}")
        
        # Move all contents up one level
        for item in single_dir.iterdir():
            shutil.move(str(item), str(extract_to / item.name))
        
        # Remove the now-empty directory
        single_dir.rmdir()

async def download_file(url: str, destination: Path):
    """Download file from URL"""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def find_colmap_dir(dataset_path: Path) -> Optional[Path]:
    """Find COLMAP directory in dataset - searches recursively"""
    
    # First check common direct patterns
    patterns = ["sparse/0", "sparse", "colmap", "."]
    for pattern in patterns:
        colmap_path = dataset_path / pattern
        if colmap_path.exists():
            # Check for COLMAP files
            if (colmap_path / "cameras.bin").exists() or \
               (colmap_path / "cameras.txt").exists():
                return colmap_path
    
    # If not found, recursively search for COLMAP files (up to 3 levels deep)
    for root, dirs, files in os.walk(dataset_path):
        root_path = Path(root)
        depth = len(root_path.relative_to(dataset_path).parts)
        
        # Limit search depth to avoid performance issues
        if depth > 3:
            continue
            
        # Check if this directory contains COLMAP files
        if ("cameras.bin" in files or "cameras.txt" in files) and \
           ("images.bin" in files or "images.txt" in files):
            logger.info(f"Found COLMAP data at: {root_path}")
            return root_path
    
    return None

async def train_gaussian_splat(job_id: str, dataset_path: Path, 
                               num_steps: int, output_name: str):
    """Background task to train Gaussian splat"""
    try:
        # Lazy import fVDB to avoid import errors at startup
        import fvdb
        try:
            import fvdb_reality_capture
            REALITY_CAPTURE_AVAILABLE = True
        except ImportError:
            REALITY_CAPTURE_AVAILABLE = False
            logger.warning("fVDB Reality Capture not available - some features may be limited")
        if REALITY_CAPTURE_AVAILABLE:
            frc = fvdb_reality_capture
        
        training_jobs[job_id]["status"] = "loading_data"
        training_jobs[job_id]["progress"] = 0.1
        training_jobs[job_id]["message"] = "Loading COLMAP scene..."
        
        # Find COLMAP directory
        colmap_dir = find_colmap_dir(dataset_path)
        if not colmap_dir:
            raise Exception("Could not find COLMAP data in dataset")
        
        # fVDB expects the parent directory containing sparse/, not sparse/0 itself
        scene_path = dataset_path if (dataset_path / "sparse").exists() else colmap_dir.parent
        logger.info(f"Loading scene from {scene_path} (COLMAP at {colmap_dir})")
        scene = frc.sfm_scene.SfmScene.from_colmap(str(scene_path))
        
        training_jobs[job_id]["message"] = f"Loaded {len(scene.images)} images"
        training_jobs[job_id]["progress"] = 0.2
        
        # Create reconstruction
        training_jobs[job_id]["status"] = "training"
        training_jobs[job_id]["message"] = "Creating reconstruction..."
        
        # OPTIMIZED FOR SHARP, DENSE RESULTS
        # Key insight: refine_stop_epoch must be close to total training duration
        # With 240 images and batch_size=1: steps_per_epoch = 240
        # So total_epochs = num_steps / 240
        
        # Calculate how many epochs we'll train
        num_images = len(scene.images)
        steps_per_epoch = num_images  # batch_size=1 (default)
        total_epochs = int(num_steps / steps_per_epoch) if num_steps else 200
        
        # Configure to refine until near the end (95% of training)
        refine_until = int(total_epochs * 0.95)
        
        config = frc.radiance_fields.GaussianSplatReconstructionConfig(
            max_steps=num_steps,
            refine_stop_epoch=refine_until,  # Continue refining until 95% done
            refine_every_epoch=0.5  # More frequent refinement (default is 0.65)
        )
        
        logger.info(f"Training config: {num_steps} steps, {total_epochs} epochs, refining until epoch {refine_until}")
        
        runner = frc.radiance_fields.GaussianSplatReconstruction.from_sfm_scene(
            scene,
            config=config
        )
        
        training_jobs[job_id]["message"] = f"Training {num_steps} steps..."
        training_jobs[job_id]["progress"] = 0.3
        
        # Train (no parameters needed - steps configured in config)
        runner.optimize()
        model = runner.model
        
        training_jobs[job_id]["status"] = "exporting"
        training_jobs[job_id]["progress"] = 0.9
        training_jobs[job_id]["message"] = "Exporting model..."
        
        # Save outputs
        output_dir = OUTPUT_DIR / job_id
        output_dir.mkdir(exist_ok=True)
        
        ply_file = output_dir / f"{output_name}.ply"
        model.save_ply(str(ply_file), metadata=runner.reconstruction_metadata)
        
        # Save metadata
        metadata = {
            "num_gaussians": model.num_gaussians,
            "device": str(model.device),
            "num_channels": model.num_channels,
            "num_images": len(scene.images),
            "training_steps": num_steps,
            "created_at": training_jobs[job_id]["created_at"]
        }
        
        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Update job status
        training_jobs[job_id]["status"] = "completed"
        training_jobs[job_id]["progress"] = 1.0
        training_jobs[job_id]["message"] = f"Completed! {model.num_gaussians} Gaussians"
        training_jobs[job_id]["completed_at"] = datetime.now().isoformat()
        training_jobs[job_id]["output_files"] = [
            f"/outputs/{job_id}/{output_name}.ply",
            f"/outputs/{job_id}/metadata.json"
        ]
        
        logger.info(f"Job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        training_jobs[job_id]["status"] = "failed"
        training_jobs[job_id]["message"] = str(e)

# API Endpoints

@app.get("/workflow")
async def workflow_monitor():
    """Complete workflow monitor with real-time status"""
    with open(Path(__file__).parent / "static" / "simple-workflow.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.get("/test-workflows")
async def test_workflows():
    """Test page for workflow detection"""
    with open(Path(__file__).parent / "static" / "test-workflows.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.get("/")
async def index():
    """Interactive web UI for training workflows"""
    gpu_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if gpu_available else 0
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else "N/A"
    
    # Get datasets
    datasets = []
    for dataset_dir in DATA_DIR.iterdir():
        if dataset_dir.is_dir():
            colmap_dir = find_colmap_dir(dataset_dir)
            datasets.append({
                "id": dataset_dir.name,
                "has_colmap": colmap_dir is not None
            })
    
    # Get recent jobs
    recent_jobs = sorted(training_jobs.values(), key=lambda x: x['created_at'], reverse=True)[:5]
    
    return HTMLResponse(f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>🎓 fVDB Training Service</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
            }}
            .header {{
                background: white;
                border-radius: 20px;
                padding: 40px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                margin-bottom: 30px;
                text-align: center;
            }}
            .header h1 {{
                color: #667eea;
                font-size: 2.5em;
                margin-bottom: 10px;
            }}
            .header p {{
                color: #666;
                font-size: 1.2em;
            }}
            .status-box {{
                background: {'linear-gradient(135deg, #11998e 0%, #38ef7d 100%)' if gpu_available else 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)'};
                color: white;
                padding: 25px;
                border-radius: 15px;
                margin: 20px 0;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            }}
            .status-box h3 {{
                margin-bottom: 15px;
                font-size: 1.3em;
            }}
            .status-item {{
                display: flex;
                justify-content: space-between;
                padding: 8px 0;
                border-bottom: 1px solid rgba(255,255,255,0.2);
            }}
            .status-item:last-child {{
                border-bottom: none;
            }}
            .workflow-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
                gap: 25px;
                margin: 30px 0;
            }}
            .workflow-card {{
                background: white;
                border-radius: 15px;
                padding: 30px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }}
            .workflow-card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 15px 40px rgba(0,0,0,0.3);
            }}
            .workflow-card h2 {{
                color: #667eea;
                margin-bottom: 15px;
                font-size: 1.5em;
            }}
            .workflow-card p {{
                color: #666;
                margin-bottom: 20px;
                line-height: 1.6;
            }}
            .step-list {{
                list-style: none;
                margin: 20px 0;
            }}
            .step-list li {{
                padding: 12px 0 12px 40px;
                position: relative;
                color: #555;
                line-height: 1.5;
            }}
            .step-list li:before {{
                content: attr(data-step);
                position: absolute;
                left: 0;
                top: 10px;
                width: 28px;
                height: 28px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: bold;
                font-size: 0.9em;
            }}
            .btn {{
                display: inline-block;
                padding: 12px 30px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                text-decoration: none;
                border-radius: 25px;
                font-weight: 600;
                transition: transform 0.2s ease, box-shadow 0.2s ease;
                border: none;
                cursor: pointer;
                font-size: 1em;
            }}
            .btn:hover {{
                transform: scale(1.05);
                box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
            }}
            .btn-secondary {{
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            }}
            .api-links {{
                display: flex;
                gap: 15px;
                flex-wrap: wrap;
                margin-top: 20px;
            }}
            .code-block {{
                background: #2d3748;
                color: #68d391;
                padding: 15px;
                border-radius: 10px;
                font-family: 'Courier New', monospace;
                font-size: 0.9em;
                overflow-x: auto;
                margin: 15px 0;
            }}
            .jobs-section {{
                background: white;
                border-radius: 15px;
                padding: 30px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                margin-top: 30px;
            }}
            .jobs-section h2 {{
                color: #667eea;
                margin-bottom: 20px;
            }}
            .job-item {{
                padding: 15px;
                border-left: 4px solid #667eea;
                background: #f7fafc;
                margin: 10px 0;
                border-radius: 5px;
            }}
            .job-status {{
                display: inline-block;
                padding: 5px 15px;
                border-radius: 20px;
                font-size: 0.85em;
                font-weight: 600;
                margin-left: 10px;
            }}
            .status-queued {{ background: #fbbf24; color: white; }}
            .status-training {{ background: #3b82f6; color: white; }}
            .status-completed {{ background: #10b981; color: white; }}
            .status-failed {{ background: #ef4444; color: white; }}
            .dataset-list {{
                background: #f7fafc;
                padding: 15px;
                border-radius: 10px;
                margin: 15px 0;
            }}
            .dataset-item {{
                padding: 10px;
                background: white;
                margin: 5px 0;
                border-radius: 5px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            .badge {{
                background: #10b981;
                color: white;
                padding: 3px 10px;
                border-radius: 12px;
                font-size: 0.8em;
            }}
            .warning-box {{
                background: #fef3c7;
                border-left: 4px solid #f59e0b;
                padding: 15px;
                border-radius: 5px;
                margin: 15px 0;
                color: #92400e;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎓 fVDB Training Service</h1>
                <p>Train custom Gaussian Splat models from videos or photos</p>
                
                <div class="status-box">
                    <h3>{'✅ System Ready' if gpu_available else '⚠️ GPU Unavailable'}</h3>
                    <div class="status-item">
                        <span><strong>GPU Available:</strong></span>
                        <span>{'Yes' if gpu_available else 'No (CPU only)'}</span>
                    </div>
                    {f'''<div class="status-item">
                        <span><strong>GPU:</strong></span>
                        <span>{gpu_name}</span>
                    </div>
                    <div class="status-item">
                        <span><strong>GPU Count:</strong></span>
                        <span>{gpu_count}</span>
                    </div>''' if gpu_available else ''}
                    <div class="status-item">
                        <span><strong>Port:</strong></span>
                        <span>8000</span>
                    </div>
                    <div class="status-item">
                        <span><strong>Datasets:</strong></span>
                        <span>{len(datasets)}</span>
                    </div>
                    <div class="status-item">
                        <span><strong>Jobs:</strong></span>
                        <span>{len(training_jobs)}</span>
                    </div>
                </div>
                
                <div class="api-links">
                    <a href="/api" class="btn">📚 API Docs (Swagger)</a>
                    <a href="/health" class="btn btn-secondary">💚 Health Check</a>
                    <a href="/jobs" class="btn">📊 View All Jobs</a>
                    <a href="/datasets" class="btn">📁 View Datasets</a>
                </div>
            </div>

            <div class="workflow-grid">
                <!-- Workflow 1: Video to Gaussian Splat -->
                <div class="workflow-card">
                    <h2>📹 Video → Gaussian Splat</h2>
                    <p>Extract frames from video and train a 3D model</p>
                    
                    <ul class="step-list">
                        <li data-step="1">Upload video (MP4, MOV, AVI)</li>
                        <li data-step="2">Extract frames at desired FPS</li>
                        <li data-step="3">Run COLMAP for camera poses</li>
                        <li data-step="4">Train Gaussian Splat model</li>
                        <li data-step="5">Download PLY file</li>
                    </ul>
                    
                    <div class="code-block">
curl -X POST http://localhost:8000/video/extract \\
  -F "file=@video.mp4" \\
  -F "fps=2.0"
                    </div>
                    
                    <p><small>⚠️ Requires COLMAP processing after frame extraction</small></p>
                </div>

                <!-- Workflow 2: Photos to Gaussian Splat -->
                <div class="workflow-card">
                    <h2>📸 Photos → Gaussian Splat</h2>
                    <p>Upload images and train directly</p>
                    
                    <ul class="step-list">
                        <li data-step="1">Upload 20+ photos (JPG, PNG)</li>
                        <li data-step="2">Run COLMAP for camera poses</li>
                        <li data-step="3">Train Gaussian Splat model</li>
                        <li data-step="4">Download PLY file</li>
                    </ul>
                    
                    <div class="code-block">
curl -X POST http://localhost:8000/upload/images \\
  -F "files=@photo1.jpg" \\
  -F "files=@photo2.jpg"
                    </div>
                    
                    <p><small>💡 Tip: More photos = better quality (50-200 recommended)</small></p>
                </div>

                <!-- Workflow 3: COLMAP Dataset -->
                <div class="workflow-card">
                    <h2>🗂️ COLMAP Dataset → Train</h2>
                    <p>Use pre-processed COLMAP data</p>
                    
                    <ul class="step-list">
                        <li data-step="1">Upload ZIP with COLMAP data</li>
                        <li data-step="2">Validate dataset structure</li>
                        <li data-step="3">Train Gaussian Splat model</li>
                        <li data-step="4">Download PLY file</li>
                    </ul>
                    
                    <div class="code-block">
curl -X POST http://localhost:8000/datasets/upload \\
  -F "file=@dataset.zip"
                    </div>
                    
                    <p><small>✅ Fastest workflow - skips COLMAP processing</small></p>
                </div>
            </div>

            <!-- Datasets Section -->
            {f'''<div class="jobs-section">
                <h2>📁 Available Datasets</h2>
                {f'<div class="dataset-list">{"".join([f"<div class='dataset-item'><span>{d['id']}</span><span class='badge'>{'COLMAP Ready' if d['has_colmap'] else 'No COLMAP'}</span></div>" for d in datasets[:10]])}</div>' if datasets else '<p>No datasets uploaded yet. Upload a dataset to get started!</p>'}
                {f'<p><small>Showing {min(10, len(datasets))} of {len(datasets)} datasets</small></p>' if len(datasets) > 10 else ''}
            </div>''' if datasets else ''}

            <!-- Recent Jobs Section -->
            {f'''<div class="jobs-section">
                <h2>⚡ Recent Training Jobs</h2>
                {"".join([f"<div class='job-item'><strong>{job['job_id']}</strong><span class='job-status status-{job['status']}'>{job['status']}</span><br><small>{job['message']}</small></div>" for job in recent_jobs])}
                <a href="/jobs" class="btn" style="margin-top: 15px;">View All Jobs</a>
            </div>''' if recent_jobs else ''}

            <!-- Quick Start Guide -->
            <div class="jobs-section">
                <h2>🚀 Quick Start Guide</h2>
                
                <h3 style="color: #667eea; margin-top: 20px;">Option 1: Upload & Train in One Step</h3>
                <div class="code-block">
# Complete workflow - upload dataset and start training
curl -X POST http://localhost:8000/workflow/complete \\
  -F "file=@my_dataset.zip" \\
  -F "dataset_name=my_scene" \\
  -F "num_steps=30000" \\
  -F "output_name=my_model"
                </div>

                <h3 style="color: #667eea; margin-top: 20px;">Option 2: Step-by-Step</h3>
                <div class="code-block">
# 1. Upload dataset
curl -X POST http://localhost:8000/datasets/upload \\
  -F "file=@dataset.zip" \\
  -F "dataset_name=my_scene"

# 2. Validate dataset
curl -X POST http://localhost:8000/datasets/my_scene/validate

# 3. Start training
curl -X POST http://localhost:8000/train \\
  -H "Content-Type: application/json" \\
  -d '{{
    "dataset_id": "my_scene",
    "num_training_steps": 30000,
    "output_name": "my_model"
  }}'

# 4. Check progress
curl http://localhost:8000/jobs/job_XXXXXX

# 5. Download result
curl http://localhost:8000/outputs/job_XXXXXX/my_model.ply -o result.ply
                </div>

                <h3 style="color: #667eea; margin-top: 20px;">Training Parameters</h3>
                <ul class="step-list">
                    <li data-step="💡">num_steps: Training iterations (7000-62200, default 30000)</li>
                    <li data-step="⚡">Quick test: 7000 steps (~5 min)</li>
                    <li data-step="✨">Good quality: 30000 steps (~15 min)</li>
                    <li data-step="🎯">Best quality: 62200 steps (~30 min)</li>
                </ul>

                {'''<div class="warning-box">
                    <strong>⚠️ GPU Required</strong><br>
                    Training requires GPU acceleration. The container is currently rebuilding with GPU support.
                    Check back in 10-15 minutes!
                </div>''' if not gpu_available else ''}

                <h3 style="color: #667eea; margin-top: 20px;">📚 Learn More</h3>
                <div class="api-links">
                    <a href="https://fvdb.ai/reality-capture/" class="btn" target="_blank">fVDB Documentation</a>
                    <a href="https://github.com/graphdeco-inria/gaussian-splatting" class="btn btn-secondary" target="_blank">3D Gaussian Splatting</a>
                    <a href="https://colmap.github.io/" class="btn" target="_blank">COLMAP Docs</a>
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
        "service": "fVDB Training Service",
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    }

@app.get("/tutorials")
async def get_tutorials():
    """Get links to fVDB tutorials"""
    return {
        "tutorials": [
            {
                "title": "Gaussian Splat Radiance Field Reconstruction",
                "url": "https://fvdb.ai/reality-capture/tutorials/radiance_field_and_mesh_reconstruction.html",
                "description": "Learn how to reconstruct Gaussian splat radiance fields and meshes"
            },
            {
                "title": "FRGS Tutorial",
                "url": "https://fvdb.ai/reality-capture/tutorials/frgs.html",
                "description": "Full tutorial on feature-based radiance Gaussian splatting"
            },
            {
                "title": "fVDB Documentation",
                "url": "https://fvdb.ai/",
                "description": "Complete fVDB documentation and guides"
            }
        ]
    }

@app.post("/datasets/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    dataset_name: Optional[str] = None
):
    """Upload a dataset as ZIP file containing COLMAP data"""
    if not file.filename.endswith('.zip'):
        raise HTTPException(400, "Only ZIP files are supported")
    
    dataset_id = dataset_name or f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    dataset_path = DATA_DIR / dataset_id
    dataset_path.mkdir(exist_ok=True)
    
    # Save uploaded file
    zip_path = UPLOAD_DIR / file.filename
    with open(zip_path, 'wb') as f:
        shutil.copyfileobj(file.file, f)
    
    # Extract
    try:
        extract_zip(zip_path, dataset_path)
        zip_path.unlink()  # Remove zip after extraction
        
        # Verify COLMAP data exists
        colmap_dir = find_colmap_dir(dataset_path)
        if not colmap_dir:
            shutil.rmtree(dataset_path)
            raise HTTPException(400, "No COLMAP data found in ZIP")
        
        return {
            "dataset_id": dataset_id,
            "status": "uploaded",
            "path": str(dataset_path),
            "colmap_dir": str(colmap_dir)
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to process dataset: {e}")

@app.post("/datasets/upload_url")
async def upload_dataset_from_url(request: DatasetUploadURL):
    """Upload dataset from URL"""
    dataset_id = request.dataset_name
    dataset_path = DATA_DIR / dataset_id
    dataset_path.mkdir(exist_ok=True)
    
    zip_path = UPLOAD_DIR / f"{dataset_id}.zip"
    
    try:
        # Download
        await download_file(str(request.url), zip_path)
        
        # Extract
        extract_zip(zip_path, dataset_path)
        zip_path.unlink()
        
        # Verify COLMAP data
        colmap_dir = find_colmap_dir(dataset_path)
        if not colmap_dir:
            shutil.rmtree(dataset_path)
            raise HTTPException(400, "No COLMAP data found in downloaded ZIP")
        
        return {
            "dataset_id": dataset_id,
            "status": "uploaded",
            "path": str(dataset_path),
            "colmap_dir": str(colmap_dir)
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to download/process dataset: {e}")

@app.get("/datasets")
async def list_datasets():
    """List all uploaded datasets"""
    datasets = []
    for dataset_dir in DATA_DIR.iterdir():
        if dataset_dir.is_dir():
            colmap_dir = find_colmap_dir(dataset_dir)
            datasets.append({
                "dataset_id": dataset_dir.name,
                "path": str(dataset_dir),
                "has_colmap": colmap_dir is not None,
                "colmap_dir": str(colmap_dir) if colmap_dir else None
            })
    return {"datasets": datasets}

@app.post("/datasets/{dataset_id}/validate")
async def validate_dataset(dataset_id: str):
    """
    Validate dataset before training
    
    Checks:
    - COLMAP files exist
    - Data format (binary vs text)
    - Scene can be loaded
    - Image count
    """
    dataset_path = DATA_DIR / dataset_id
    if not dataset_path.exists():
        raise HTTPException(404, "Dataset not found")
    
    validation = {
        "dataset_id": dataset_id,
        "valid": False,
        "checks": {},
        "warnings": [],
        "errors": []
    }
    
    # Check 1: COLMAP directory exists
    colmap_dir = find_colmap_dir(dataset_path)
    validation["checks"]["colmap_found"] = colmap_dir is not None
    
    if not colmap_dir:
        validation["errors"].append("No COLMAP data found")
        return validation
    
    validation["colmap_dir"] = str(colmap_dir)
    
    # Check 2: File format
    has_bin = (colmap_dir / "cameras.bin").exists()
    has_txt = (colmap_dir / "cameras.txt").exists()
    
    validation["checks"]["format"] = "binary" if has_bin else "text" if has_txt else "unknown"
    
    if has_txt and not has_bin:
        validation["warnings"].append(
            "Text format COLMAP files detected. Binary format is more reliable."
        )
    
    # Check 3: Try to load scene
    try:
        import fvdb_reality_capture as frc
        
        scene_path = dataset_path if (dataset_path / "sparse").exists() else colmap_dir.parent
        scene = frc.sfm_scene.SfmScene.from_colmap(str(scene_path))
        
        validation["checks"]["scene_loadable"] = True
        validation["checks"]["num_images"] = len(scene.images)
        validation["checks"]["num_points"] = len(scene.points)
        
        if len(scene.images) < 10:
            validation["warnings"].append(
                f"Only {len(scene.images)} images found. Recommend 20+ for good results."
            )
        
        validation["valid"] = True
        validation["message"] = f"Dataset valid: {len(scene.images)} images, {len(scene.points)} points"
        
    except Exception as e:
        validation["checks"]["scene_loadable"] = False
        validation["errors"].append(f"Failed to load scene: {str(e)}")
        
        # Specific error handling
        if "integer -1 out of bounds" in str(e):
            validation["errors"].append(
                "COLMAP data contains invalid values (-1). "
                "This typically happens with text format files. "
                "Solution: Convert to binary format or use a different dataset."
            )
        
        validation["message"] = "Dataset validation failed"
    
    return validation

@app.post("/train")
async def start_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks
):
    """Start training a Gaussian splat model (runs in subprocess to avoid blocking API)"""
    dataset_path = DATA_DIR / request.dataset_id
    if not dataset_path.exists():
        raise HTTPException(404, f"Dataset {request.dataset_id} not found")
    
    job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    output_name = request.output_name or f"model_{job_id}"
    
    # Initialize job state
    initial_state = {
        "job_id": job_id,
        "status": "starting",
        "progress": 0.0,
        "message": "Starting training subprocess...",
        "created_at": datetime.now().isoformat(),
        "dataset_id": request.dataset_id,
        "num_training_steps": request.num_training_steps
    }
    save_job_state(job_id, initial_state)
    
    # Start training in subprocess (non-blocking!)
    try:
        process = subprocess.Popen(
            [
                "python3", "/app/train_subprocess.py",
                job_id,
                str(dataset_path),
                str(request.num_training_steps),
                output_name,
                str(OUTPUT_DIR)
            ],
            stdout=open(f"/tmp/train_{job_id}.log", "w"),
            stderr=subprocess.STDOUT,
            start_new_session=True  # Detach from parent
        )
        training_processes[job_id] = process
        logger.info(f"Started training subprocess for {job_id} (PID: {process.pid})")
    except Exception as e:
        logger.error(f"Failed to start training subprocess: {e}")
        save_job_state(job_id, {
            **initial_state,
            "status": "failed",
            "message": f"Failed to start: {str(e)}"
        })
        raise HTTPException(500, f"Failed to start training: {str(e)}")
    
    return {
        "job_id": job_id,
        "status": "starting",
        "message": "Training job started in background subprocess"
    }

@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get status of a training job (reads from state file)"""
    # First check state file (most up-to-date)
    state = load_job_state(job_id)
    if state:
        training_jobs[job_id] = state  # Update cache
        return state
    
    if job_id not in training_jobs:
        raise HTTPException(404, "Job not found")
    return training_jobs[job_id]

@app.get("/jobs")
async def list_jobs():
    """List all training jobs (syncs from state files)"""
    sync_all_jobs()  # Refresh from files
    return {"jobs": list(training_jobs.values())}

@app.get("/outputs/{job_id}/{filename}")
async def download_output(job_id: str, filename: str):
    """Download output file from completed job"""
    file_path = OUTPUT_DIR / job_id / filename
    if not file_path.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(file_path)

@app.get("/outputs/{job_id}")
async def list_outputs(job_id: str):
    """List output files for a job"""
    output_dir = OUTPUT_DIR / job_id
    if not output_dir.exists():
        raise HTTPException(404, "Job outputs not found")
    
    files = []
    for file_path in output_dir.iterdir():
        files.append({
            "filename": file_path.name,
            "size": file_path.stat().st_size,
            "download_url": f"/outputs/{job_id}/{file_path.name}"
        })
    return {"files": files}

@app.post("/video/extract")
async def extract_video_frames(
    file: UploadFile = File(...),
    fps: float = 2.0,
    max_frames: Optional[int] = None,
    output_name: Optional[str] = None
):
    """
    Extract frames from uploaded video file for 3D reconstruction (requires ffmpeg)
    
    Args:
        file: Video file (mp4, mov, avi, etc.)
        fps: Frames per second to extract (default: 2.0)
        max_frames: Maximum number of frames (default: None = all)
        output_name: Name for output directory
    
    Returns:
        Information about extracted frames
    """
    if not VIDEO_EXTRACTION_AVAILABLE:
        raise HTTPException(500, "Video extraction not available - extract_frames module not loaded")
    
    try:
        # Validate file type
        video_extensions = [".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv"]
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in video_extensions:
            raise HTTPException(400, f"Unsupported video format: {file_ext}. Supported: {video_extensions}")
        
        # Generate output name
        if not output_name:
            output_name = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Save uploaded video
        video_path = UPLOAD_DIR / f"{output_name}{file_ext}"
        with open(video_path, 'wb') as f:
            content = await file.read()
            f.write(content)
        
        logger.info(f"Video uploaded: {video_path} ({len(content) / 1024 / 1024:.1f} MB)")
        
        # Get video info
        video_info = get_video_info(str(video_path))
        if not video_info["success"]:
            raise Exception(f"Failed to analyze video: {video_info.get('error')}")
        
        # Get recommendations
        recommendations = recommend_extraction_params(video_info["duration"])
        
        # Create output directory for frames
        frames_dir = DATA_DIR / f"{output_name}_frames"
        
        # Extract frames
        result = extract_frames_from_video(
            str(video_path),
            str(frames_dir),
            fps=fps,
            max_frames=max_frames
        )
        
        if not result["success"]:
            raise Exception(f"Frame extraction failed: {result.get('error')}")
        
        return {
            "output_name": output_name,
            "video_info": video_info,
            "recommendations": recommendations,
            "extraction": result,
            "dataset_id": f"{output_name}_frames",
            "message": f"Extracted {result['num_frames']} frames. Use dataset_id '{output_name}_frames' for training."
        }
        
    except Exception as e:
        logger.error(f"Video frame extraction failed: {e}")
        raise HTTPException(500, f"Video processing failed: {e}")


@app.post("/workflow/complete")
async def complete_workflow(
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = None,
    dataset_name: Optional[str] = None,
    num_steps: int = 30000,  # Default to 30000 for professional quality (3DGS standard)
    output_name: Optional[str] = None
):
    """
    End-to-end workflow: Upload dataset (ZIP or URL) → Train → Export
    
    This endpoint handles the complete pipeline:
    1. Upload dataset from file or URL
    2. Automatically start training
    3. Export trained model to shared volume for rendering
    
    Returns job_id to monitor progress via /jobs/{job_id}
    """
    try:
        # Step 1: Upload dataset
        if file:
            # Upload from file
            if not file.filename.endswith('.zip'):
                raise HTTPException(400, "Only ZIP files are supported")
            
            dataset_id = dataset_name or f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            dataset_path = DATA_DIR / dataset_id
            dataset_path.mkdir(exist_ok=True)
            
            zip_path = UPLOAD_DIR / file.filename
            with open(zip_path, 'wb') as f:
                shutil.copyfileobj(file.file, f)
            
            extract_zip(zip_path, dataset_path)
            zip_path.unlink()
            
        elif url:
            # Upload from URL
            dataset_id = dataset_name or f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            dataset_path = DATA_DIR / dataset_id
            dataset_path.mkdir(exist_ok=True)
            
            zip_path = UPLOAD_DIR / f"{dataset_id}.zip"
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            extract_zip(zip_path, dataset_path)
            zip_path.unlink()
        else:
            raise HTTPException(400, "Must provide either 'file' or 'url'")
        
        # Verify COLMAP data
        colmap_dir = find_colmap_dir(dataset_path)
        if not colmap_dir:
            shutil.rmtree(dataset_path)
            raise HTTPException(400, "No COLMAP data found in dataset")
        
        # Step 2: Start training
        model_name = output_name or f"{dataset_id}_model"
        job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        training_jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0.0,
            "message": "Workflow started - training queued",
            "created_at": datetime.now().isoformat(),
            "dataset_id": dataset_id,
            "num_training_steps": num_steps
        }
        
        background_tasks.add_task(
            train_gaussian_splat,
            job_id=job_id,
            dataset_path=dataset_path,
            num_steps=num_steps,
            output_name=model_name
        )
        
        return {
            "job_id": job_id,
            "dataset_id": dataset_id,
            "output_name": model_name,
            "num_steps": num_steps,
            "status": "queued",
            "message": "End-to-end workflow started. Monitor progress via /jobs/{job_id}",
            "endpoints": {
                "status": f"/jobs/{job_id}",
                "outputs": f"/outputs/{job_id}",
                "rendering_service": "http://localhost:8001/api"
            }
        }
        
    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        raise HTTPException(500, f"Workflow failed: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
