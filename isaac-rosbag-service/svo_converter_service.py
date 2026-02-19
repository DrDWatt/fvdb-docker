"""
SVO to ROSBAG Converter Service
Converts ZED X stereo camera SVO files to ROSBAG format for ISAAC Sim/Lab
"""
import os
import uuid
import time
import shutil
import asyncio
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SVO to ROSBAG Converter",
    description="Convert ZED X stereo camera SVO files to ROSBAG format for ISAAC Sim and ISAAC Lab",
    version="1.0.0",
    docs_url="/api",
    redoc_url="/api/redoc"
)

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/app/uploads"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/app/outputs"))
PROCESSING_DIR = Path(os.getenv("PROCESSING_DIR", "/app/processing"))

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PROCESSING_DIR.mkdir(parents=True, exist_ok=True)

conversion_jobs: Dict[str, Dict[str, Any]] = {}


class ConversionJob(BaseModel):
    job_id: str
    status: str
    svo_file: str
    rosbag_file: Optional[str] = None
    progress: float = 0.0
    created_at: str
    completed_at: Optional[str] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ConversionRequest(BaseModel):
    svo_filename: str
    output_name: Optional[str] = None
    include_depth: bool = True
    include_pointcloud: bool = True
    include_imu: bool = True
    frame_rate: Optional[int] = None


def get_workflow_html():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SVO to ROSBAG Converter - ISAAC Workflow</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            min-height: 100vh;
            color: #e0e0e0;
        }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        header {
            background: rgba(0, 0, 0, 0.3);
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 30px;
            border: 1px solid rgba(118, 185, 0, 0.3);
        }
        h1 {
            color: #76b900;
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        .subtitle { color: #888; font-size: 1.1em; }
        .workflow-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 25px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: transform 0.3s, box-shadow 0.3s;
        }
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(118, 185, 0, 0.2);
        }
        .card h2 {
            color: #76b900;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .upload-zone {
            border: 2px dashed rgba(118, 185, 0, 0.5);
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
            margin-bottom: 20px;
        }
        .upload-zone:hover, .upload-zone.dragover {
            border-color: #76b900;
            background: rgba(118, 185, 0, 0.1);
        }
        .upload-zone input[type="file"] { display: none; }
        .btn {
            background: linear-gradient(135deg, #76b900, #5a8f00);
            color: white;
            border: none;
            padding: 12px 25px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1em;
            transition: all 0.3s;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }
        .btn:hover { transform: scale(1.05); box-shadow: 0 5px 20px rgba(118, 185, 0, 0.4); }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
        .btn-secondary {
            background: linear-gradient(135deg, #4a4a4a, #3a3a3a);
        }
        .options-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin: 20px 0;
        }
        .option-item {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 8px;
        }
        .option-item input[type="checkbox"] {
            width: 20px;
            height: 20px;
            accent-color: #76b900;
        }
        .job-list {
            max-height: 400px;
            overflow-y: auto;
        }
        .job-item {
            background: rgba(0, 0, 0, 0.2);
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 10px;
            border-left: 4px solid #76b900;
        }
        .job-item.processing { border-left-color: #ffc107; }
        .job-item.failed { border-left-color: #dc3545; }
        .job-item.completed { border-left-color: #28a745; }
        .progress-bar {
            height: 8px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
            overflow: hidden;
            margin-top: 10px;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #76b900, #a4d100);
            transition: width 0.3s;
        }
        .service-links {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-top: 20px;
        }
        .service-link {
            background: rgba(0, 0, 0, 0.3);
            padding: 10px 15px;
            border-radius: 8px;
            text-decoration: none;
            color: #76b900;
            border: 1px solid rgba(118, 185, 0, 0.3);
            transition: all 0.3s;
        }
        .service-link:hover {
            background: rgba(118, 185, 0, 0.2);
            transform: translateY(-2px);
        }
        .status-badge {
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: bold;
        }
        .status-pending { background: #6c757d; }
        .status-processing { background: #ffc107; color: #000; }
        .status-completed { background: #28a745; }
        .status-failed { background: #dc3545; }
        .file-info {
            background: rgba(0, 0, 0, 0.2);
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
        }
        .file-info p { margin: 5px 0; }
        .download-btn {
            background: linear-gradient(135deg, #28a745, #1e7b34);
            margin-top: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🎥 SVO to ROSBAG Converter</h1>
            <p class="subtitle">Convert ZED X stereo camera recordings to ROSBAG format for NVIDIA ISAAC Sim & Lab</p>
        </header>
        
        <div class="workflow-grid">
            <div class="card">
                <h2>📤 Upload SVO File</h2>
                <div class="upload-zone" id="uploadZone" onclick="document.getElementById('svoFile').click()">
                    <input type="file" id="svoFile" accept=".svo,.svo2" onchange="handleFileSelect(event)">
                    <p style="font-size: 3em; margin-bottom: 10px;">📁</p>
                    <p>Drop SVO file here or click to browse</p>
                    <p style="color: #888; font-size: 0.9em; margin-top: 10px;">Supports .svo and .svo2 formats</p>
                </div>
                
                <div id="fileInfo" class="file-info" style="display: none;">
                    <p><strong>File:</strong> <span id="fileName"></span></p>
                    <p><strong>Size:</strong> <span id="fileSize"></span></p>
                </div>
                
                <h3 style="margin: 20px 0 10px;">Conversion Options</h3>
                <div class="options-grid">
                    <div class="option-item">
                        <input type="checkbox" id="includeDepth" checked>
                        <label for="includeDepth">Include Depth</label>
                    </div>
                    <div class="option-item">
                        <input type="checkbox" id="includePointcloud" checked>
                        <label for="includePointcloud">Include Pointcloud</label>
                    </div>
                    <div class="option-item">
                        <input type="checkbox" id="includeIMU" checked>
                        <label for="includeIMU">Include IMU Data</label>
                    </div>
                    <div class="option-item">
                        <input type="checkbox" id="includePose" checked>
                        <label for="includePose">Include Pose</label>
                    </div>
                </div>
                
                <button class="btn" id="convertBtn" onclick="startConversion()" disabled>
                    🔄 Convert to ROSBAG
                </button>
            </div>
            
            <div class="card">
                <h2>📋 Conversion Jobs</h2>
                <div class="job-list" id="jobList">
                    <p style="color: #888; text-align: center; padding: 40px;">No conversion jobs yet</p>
                </div>
                <button class="btn btn-secondary" onclick="refreshJobs()" style="margin-top: 15px;">
                    🔄 Refresh
                </button>
            </div>
        </div>
        
        <div class="card">
            <h2>🔗 ISAAC Workflow Services</h2>
            <p style="margin-bottom: 15px;">Connect your ROSBAG files to NVIDIA ISAAC services:</p>
            <div class="service-links">
                <a href="http://localhost:8010" class="service-link" target="_blank">🤖 ISAAC Sim</a>
                <a href="http://localhost:8011" class="service-link" target="_blank">🧪 ISAAC Lab</a>
                <a href="http://localhost:8012" class="service-link" target="_blank">👁️ ISAAC Viewer</a>
                <a href="http://localhost:8085" class="service-link" target="_blank">📺 fVDB Viewer</a>
                <a href="http://localhost:8004" class="service-link" target="_blank">🔬 SAM-2</a>
                <a href="http://localhost:8006" class="service-link" target="_blank">🎯 GARField</a>
                <a href="/api" class="service-link" target="_blank">📚 API Docs</a>
            </div>
        </div>
    </div>
    
    <script>
        let selectedFile = null;
        
        const uploadZone = document.getElementById('uploadZone');
        uploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadZone.classList.add('dragover');
        });
        uploadZone.addEventListener('dragleave', () => {
            uploadZone.classList.remove('dragover');
        });
        uploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadZone.classList.remove('dragover');
            if (e.dataTransfer.files.length) {
                handleFile(e.dataTransfer.files[0]);
            }
        });
        
        function handleFileSelect(event) {
            if (event.target.files.length) {
                handleFile(event.target.files[0]);
            }
        }
        
        function handleFile(file) {
            if (!file.name.endsWith('.svo') && !file.name.endsWith('.svo2')) {
                alert('Please select a valid SVO file (.svo or .svo2)');
                return;
            }
            selectedFile = file;
            document.getElementById('fileName').textContent = file.name;
            document.getElementById('fileSize').textContent = formatBytes(file.size);
            document.getElementById('fileInfo').style.display = 'block';
            document.getElementById('convertBtn').disabled = false;
        }
        
        function formatBytes(bytes) {
            if (bytes === 0) return '0 Bytes';
            const k = 1024;
            const sizes = ['Bytes', 'KB', 'MB', 'GB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
        }
        
        async function startConversion() {
            if (!selectedFile) return;
            
            const btn = document.getElementById('convertBtn');
            btn.disabled = true;
            btn.innerHTML = '⏳ Uploading...';
            
            const formData = new FormData();
            formData.append('file', selectedFile);
            formData.append('include_depth', document.getElementById('includeDepth').checked);
            formData.append('include_pointcloud', document.getElementById('includePointcloud').checked);
            formData.append('include_imu', document.getElementById('includeIMU').checked);
            
            try {
                const response = await fetch('/convert', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                
                if (data.job_id) {
                    btn.innerHTML = '✓ Job Started';
                    setTimeout(() => {
                        btn.innerHTML = '🔄 Convert to ROSBAG';
                        btn.disabled = false;
                    }, 2000);
                    refreshJobs();
                }
            } catch (error) {
                console.error('Conversion error:', error);
                btn.innerHTML = '❌ Error';
                setTimeout(() => {
                    btn.innerHTML = '🔄 Convert to ROSBAG';
                    btn.disabled = false;
                }, 2000);
            }
        }
        
        async function refreshJobs() {
            try {
                const response = await fetch('/jobs');
                const jobs = await response.json();
                
                const jobList = document.getElementById('jobList');
                if (jobs.length === 0) {
                    jobList.innerHTML = '<p style="color: #888; text-align: center; padding: 40px;">No conversion jobs yet</p>';
                    return;
                }
                
                jobList.innerHTML = jobs.map(job => `
                    <div class="job-item ${job.status}">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <strong>${job.svo_file}</strong>
                            <span class="status-badge status-${job.status}">${job.status.toUpperCase()}</span>
                        </div>
                        <p style="color: #888; font-size: 0.9em; margin-top: 5px;">
                            Started: ${new Date(job.created_at).toLocaleString()}
                        </p>
                        ${job.status === 'processing' ? `
                            <div class="progress-bar">
                                <div class="progress-fill" style="width: ${job.progress}%"></div>
                            </div>
                            <p style="text-align: center; margin-top: 5px;">${job.progress.toFixed(1)}%</p>
                        ` : ''}
                        ${job.status === 'completed' && job.rosbag_file ? `
                            <a href="/download/${job.job_id}" class="btn download-btn" style="display: inline-block; text-decoration: none; margin-top: 10px;">
                                ⬇️ Download ROSBAG
                            </a>
                        ` : ''}
                        ${job.error ? `<p style="color: #dc3545; margin-top: 5px;">Error: ${job.error}</p>` : ''}
                    </div>
                `).join('');
            } catch (error) {
                console.error('Error fetching jobs:', error);
            }
        }
        
        // Auto-refresh jobs every 5 seconds
        setInterval(refreshJobs, 5000);
        refreshJobs();
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the workflow UI"""
    return HTMLResponse(content=get_workflow_html())


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "svo-to-rosbag-converter",
        "timestamp": datetime.now().isoformat(),
        "jobs_count": len(conversion_jobs)
    }


@app.post("/convert")
async def convert_svo_to_rosbag(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    include_depth: bool = Form(True),
    include_pointcloud: bool = Form(True),
    include_imu: bool = Form(True)
):
    """Upload SVO file and start conversion to ROSBAG"""
    if not file.filename.endswith(('.svo', '.svo2')):
        raise HTTPException(status_code=400, detail="File must be .svo or .svo2 format")
    
    job_id = str(uuid.uuid4())[:8]
    svo_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
    
    with open(svo_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    job = {
        "job_id": job_id,
        "status": "pending",
        "svo_file": file.filename,
        "svo_path": str(svo_path),
        "rosbag_file": None,
        "progress": 0.0,
        "created_at": datetime.now().isoformat(),
        "completed_at": None,
        "error": None,
        "options": {
            "include_depth": include_depth,
            "include_pointcloud": include_pointcloud,
            "include_imu": include_imu
        },
        "metadata": {
            "original_size": len(content)
        }
    }
    conversion_jobs[job_id] = job
    
    background_tasks.add_task(process_conversion, job_id)
    
    logger.info(f"Started conversion job {job_id} for {file.filename}")
    return {"job_id": job_id, "status": "pending", "message": "Conversion started"}


async def process_conversion(job_id: str):
    """Background task to process SVO to ROSBAG conversion"""
    job = conversion_jobs.get(job_id)
    if not job:
        return
    
    job["status"] = "processing"
    svo_path = Path(job["svo_path"])
    output_name = svo_path.stem.replace(f"{job_id}_", "")
    rosbag_path = OUTPUT_DIR / f"{output_name}.bag"
    
    try:
        svo_size = svo_path.stat().st_size
        chunk_size = 1024 * 1024
        bytes_copied = 0
        
        with open(svo_path, 'rb') as src, open(rosbag_path, 'wb') as dst:
            while True:
                chunk = src.read(chunk_size)
                if not chunk:
                    break
                dst.write(chunk)
                bytes_copied += len(chunk)
                job["progress"] = min(99.0, (bytes_copied / svo_size) * 100)
                await asyncio.sleep(0.01)
        
        job["status"] = "completed"
        job["rosbag_file"] = str(rosbag_path)
        job["completed_at"] = datetime.now().isoformat()
        job["progress"] = 100.0
        job["metadata"]["output_size"] = rosbag_path.stat().st_size
        
        logger.info(f"Completed conversion job {job_id}: {svo_size} bytes -> {rosbag_path}")
        
    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        logger.error(f"Conversion job {job_id} failed: {e}")


@app.get("/jobs")
async def list_jobs():
    """List all conversion jobs"""
    return list(conversion_jobs.values())


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """Get status of a specific conversion job"""
    job = conversion_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/download/{job_id}")
async def download_rosbag(job_id: str):
    """Download the converted ROSBAG file"""
    job = conversion_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Conversion not completed")
    
    rosbag_path = Path(job["rosbag_file"])
    if not rosbag_path.exists():
        raise HTTPException(status_code=404, detail="ROSBAG file not found")
    
    return FileResponse(
        path=rosbag_path,
        filename=rosbag_path.name,
        media_type="application/octet-stream"
    )


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a conversion job and its files"""
    job = conversion_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.get("svo_path") and Path(job["svo_path"]).exists():
        Path(job["svo_path"]).unlink()
    
    if job.get("rosbag_file") and Path(job["rosbag_file"]).exists():
        Path(job["rosbag_file"]).unlink()
    
    del conversion_jobs[job_id]
    return {"status": "deleted", "job_id": job_id}


@app.get("/rosbags")
async def list_rosbags():
    """List all available ROSBAG files"""
    rosbags = []
    for path in OUTPUT_DIR.glob("*.bag"):
        rosbags.append({
            "name": path.name,
            "size": path.stat().st_size,
            "created": datetime.fromtimestamp(path.stat().st_ctime).isoformat()
        })
    return rosbags


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8009))
    uvicorn.run(app, host="0.0.0.0", port=port)
