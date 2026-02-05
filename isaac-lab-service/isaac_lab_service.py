"""
ISAAC Lab Service
Reinforcement learning framework for robotics built on ISAAC Sim
"""
import os
import uuid
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ISAAC Lab Service",
    description="Reinforcement learning framework for robotics training and evaluation",
    version="1.0.0",
    docs_url="/api",
    redoc_url="/api/redoc"
)

CHECKPOINT_DIR = Path(os.getenv("CHECKPOINT_DIR", "/app/checkpoints"))
LOG_DIR = Path(os.getenv("LOG_DIR", "/app/logs"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/app/outputs"))

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

training_jobs: Dict[str, Dict[str, Any]] = {}


class TrainingConfig(BaseModel):
    task: str = "Isaac-Velocity-Flat-Anymal-D-v0"
    algorithm: str = "PPO"
    num_envs: int = 4096
    max_iterations: int = 1000
    checkpoint_interval: int = 100
    seed: int = 42


def get_ui_html():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ISAAC Lab - RL Training</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            min-height: 100vh;
            color: #e0e0e0;
        }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        header {
            background: rgba(0, 0, 0, 0.3);
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 30px;
            border: 1px solid rgba(118, 185, 0, 0.3);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        h1 { color: #76b900; font-size: 2.2em; }
        .subtitle { color: #888; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 25px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .card h2 { color: #76b900; margin-bottom: 20px; }
        .form-group { margin-bottom: 15px; }
        .form-group label { display: block; margin-bottom: 5px; color: #aaa; }
        .form-group select, .form-group input {
            width: 100%;
            padding: 10px;
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            background: rgba(0, 0, 0, 0.3);
            color: #fff;
        }
        .btn {
            background: linear-gradient(135deg, #76b900, #5a8f00);
            color: white;
            border: none;
            padding: 12px 25px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1em;
            margin-right: 10px;
            margin-top: 10px;
        }
        .btn:hover { transform: scale(1.02); }
        .btn-stop { background: linear-gradient(135deg, #dc3545, #a71d2a); }
        .job-item {
            background: rgba(0, 0, 0, 0.2);
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 10px;
            border-left: 4px solid #76b900;
        }
        .job-item.running { border-left-color: #ffc107; animation: pulse 2s infinite; }
        .job-item.completed { border-left-color: #28a745; }
        .job-item.failed { border-left-color: #dc3545; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.7; } }
        .progress-bar {
            height: 10px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 5px;
            overflow: hidden;
            margin-top: 10px;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #76b900, #a4d100);
            transition: width 0.5s;
        }
        .metrics {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            margin-top: 15px;
        }
        .metric {
            background: rgba(0, 0, 0, 0.2);
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }
        .metric-value { font-size: 1.5em; color: #76b900; font-weight: bold; }
        .metric-label { color: #888; font-size: 0.9em; }
        .links { display: flex; gap: 10px; flex-wrap: wrap; }
        .link {
            background: rgba(0, 0, 0, 0.3);
            padding: 10px 15px;
            border-radius: 8px;
            text-decoration: none;
            color: #76b900;
            border: 1px solid rgba(118, 185, 0, 0.3);
        }
        .link:hover { background: rgba(118, 185, 0, 0.2); }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>🧪 NVIDIA ISAAC Lab</h1>
                <p class="subtitle">Reinforcement learning framework for robotics</p>
            </div>
            <div class="links">
                <a href="/api" class="link">📚 API Docs</a>
                <a href="http://localhost:8010" class="link">🤖 ISAAC Sim</a>
                <a href="http://localhost:8009" class="link">🔄 SVO Converter</a>
            </div>
        </header>
        
        <div class="grid">
            <div class="card">
                <h2>🎯 Training Configuration</h2>
                <div class="form-group">
                    <label>Task Environment</label>
                    <select id="task">
                        <option value="Isaac-Velocity-Flat-Anymal-D-v0">Anymal-D Velocity (Flat)</option>
                        <option value="Isaac-Velocity-Rough-Anymal-D-v0">Anymal-D Velocity (Rough)</option>
                        <option value="Isaac-Reach-Franka-v0">Franka Reach</option>
                        <option value="Isaac-Lift-Cube-Franka-v0">Franka Lift Cube</option>
                        <option value="Isaac-Navigation-Flat-Anymal-D-v0">Anymal-D Navigation</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Algorithm</label>
                    <select id="algorithm">
                        <option value="PPO">PPO (Proximal Policy Optimization)</option>
                        <option value="SAC">SAC (Soft Actor-Critic)</option>
                        <option value="TD3">TD3 (Twin Delayed DDPG)</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Number of Environments</label>
                    <input type="number" id="numEnvs" value="4096">
                </div>
                <div class="form-group">
                    <label>Max Iterations</label>
                    <input type="number" id="maxIter" value="1000">
                </div>
                <button class="btn" onclick="startTraining()">🚀 Start Training</button>
                <button class="btn btn-stop" onclick="stopTraining()">⏹️ Stop</button>
            </div>
            
            <div class="card">
                <h2>📊 Training Jobs</h2>
                <div id="jobList"></div>
            </div>
        </div>
        
        <div class="card" style="margin-top: 20px;">
            <h2>📈 Live Metrics</h2>
            <div class="metrics">
                <div class="metric">
                    <div class="metric-value" id="metricReward">--</div>
                    <div class="metric-label">Mean Reward</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="metricEpisodes">--</div>
                    <div class="metric-label">Episodes</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="metricFPS">--</div>
                    <div class="metric-label">Training FPS</div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        async function startTraining() {
            const config = {
                task: document.getElementById('task').value,
                algorithm: document.getElementById('algorithm').value,
                num_envs: parseInt(document.getElementById('numEnvs').value),
                max_iterations: parseInt(document.getElementById('maxIter').value)
            };
            
            try {
                const response = await fetch('/training/start', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(config)
                });
                const data = await response.json();
                alert('Training started: ' + data.job_id);
                refreshJobs();
            } catch (e) { console.error(e); }
        }
        
        async function stopTraining() {
            try {
                await fetch('/training/stop', { method: 'POST' });
                refreshJobs();
            } catch (e) { console.error(e); }
        }
        
        async function refreshJobs() {
            try {
                const response = await fetch('/jobs');
                const jobs = await response.json();
                const list = document.getElementById('jobList');
                
                if (jobs.length === 0) {
                    list.innerHTML = '<p style="color: #888; text-align: center; padding: 40px;">No training jobs</p>';
                    return;
                }
                
                list.innerHTML = jobs.map(job => `
                    <div class="job-item ${job.status}">
                        <div style="display: flex; justify-content: space-between;">
                            <strong>${job.task}</strong>
                            <span>${job.status.toUpperCase()}</span>
                        </div>
                        <p style="color: #888; font-size: 0.9em;">${job.algorithm} - ${job.num_envs} envs</p>
                        ${job.status === 'running' ? `
                            <div class="progress-bar">
                                <div class="progress-fill" style="width: ${(job.current_iteration / job.max_iterations) * 100}%"></div>
                            </div>
                            <p style="text-align: center; margin-top: 5px;">${job.current_iteration} / ${job.max_iterations}</p>
                        ` : ''}
                    </div>
                `).join('');
                
                const runningJob = jobs.find(j => j.status === 'running');
                if (runningJob && runningJob.metrics) {
                    document.getElementById('metricReward').textContent = runningJob.metrics.mean_reward?.toFixed(2) || '--';
                    document.getElementById('metricEpisodes').textContent = runningJob.metrics.episodes || '--';
                    document.getElementById('metricFPS').textContent = runningJob.metrics.fps?.toFixed(0) || '--';
                }
            } catch (e) { console.error(e); }
        }
        
        refreshJobs();
        setInterval(refreshJobs, 2000);
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(content=get_ui_html())


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "isaac-lab",
        "timestamp": datetime.now().isoformat(),
        "jobs_count": len(training_jobs)
    }


@app.post("/training/start")
async def start_training(config: TrainingConfig, background_tasks: BackgroundTasks):
    """Start a new RL training job"""
    job_id = str(uuid.uuid4())[:8]
    
    job = {
        "job_id": job_id,
        "status": "running",
        "task": config.task,
        "algorithm": config.algorithm,
        "num_envs": config.num_envs,
        "max_iterations": config.max_iterations,
        "current_iteration": 0,
        "started_at": datetime.now().isoformat(),
        "metrics": {
            "mean_reward": 0.0,
            "episodes": 0,
            "fps": 0.0
        }
    }
    training_jobs[job_id] = job
    
    background_tasks.add_task(simulate_training, job_id)
    
    logger.info(f"Started training job {job_id}")
    return job


async def simulate_training(job_id: str):
    """Simulate training progress"""
    job = training_jobs.get(job_id)
    if not job:
        return
    
    import random
    for i in range(job["max_iterations"]):
        if job["status"] != "running":
            break
        
        await asyncio.sleep(0.1)
        job["current_iteration"] = i + 1
        job["metrics"]["mean_reward"] = 100 + i * 0.5 + random.uniform(-10, 10)
        job["metrics"]["episodes"] = (i + 1) * job["num_envs"]
        job["metrics"]["fps"] = 50000 + random.uniform(-5000, 5000)
    
    if job["status"] == "running":
        job["status"] = "completed"
        job["completed_at"] = datetime.now().isoformat()


@app.post("/training/stop")
async def stop_training(job_id: Optional[str] = None):
    """Stop training job(s)"""
    if job_id:
        if job_id in training_jobs:
            training_jobs[job_id]["status"] = "stopped"
            return {"status": "stopped", "job_id": job_id}
        raise HTTPException(status_code=404, detail="Job not found")
    
    for jid in training_jobs:
        if training_jobs[jid]["status"] == "running":
            training_jobs[jid]["status"] = "stopped"
    return {"status": "all_stopped"}


@app.get("/jobs")
async def list_jobs():
    """List all training jobs"""
    return list(training_jobs.values())


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """Get specific training job"""
    job = training_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/tasks")
async def list_tasks():
    """List available RL tasks"""
    return [
        {"id": "Isaac-Velocity-Flat-Anymal-D-v0", "name": "Anymal-D Velocity (Flat)", "type": "locomotion"},
        {"id": "Isaac-Velocity-Rough-Anymal-D-v0", "name": "Anymal-D Velocity (Rough)", "type": "locomotion"},
        {"id": "Isaac-Reach-Franka-v0", "name": "Franka Reach", "type": "manipulation"},
        {"id": "Isaac-Lift-Cube-Franka-v0", "name": "Franka Lift Cube", "type": "manipulation"},
        {"id": "Isaac-Navigation-Flat-Anymal-D-v0", "name": "Anymal-D Navigation", "type": "navigation"}
    ]


@app.get("/checkpoints")
async def list_checkpoints():
    """List available model checkpoints"""
    checkpoints = []
    for path in CHECKPOINT_DIR.glob("*.pt"):
        checkpoints.append({"name": path.name, "size": path.stat().st_size})
    return checkpoints


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8011))
    uvicorn.run(app, host="0.0.0.0", port=port)
