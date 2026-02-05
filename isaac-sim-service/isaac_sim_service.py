"""
ISAAC Sim Service
Provides API access to NVIDIA ISAAC Sim for robotics simulation
"""
import os
import uuid
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ISAAC Sim Service",
    description="NVIDIA ISAAC Sim robotics simulation platform API",
    version="1.0.0",
    docs_url="/api",
    redoc_url="/api/redoc"
)

ROSBAG_DIR = Path(os.getenv("ROSBAG_DIR", "/app/rosbags"))
SCENE_DIR = Path(os.getenv("SCENE_DIR", "/app/scenes"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/app/outputs"))

ROSBAG_DIR.mkdir(parents=True, exist_ok=True)
SCENE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

simulation_sessions: Dict[str, Dict[str, Any]] = {}


class SimulationConfig(BaseModel):
    scene_name: str = "warehouse"
    robot_type: str = "carter"
    rosbag_file: Optional[str] = None
    physics_dt: float = 1/60
    render_dt: float = 1/30
    headless: bool = True


def get_ui_html():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ISAAC Sim - Robotics Simulation</title>
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
        .grid { display: grid; grid-template-columns: 1fr 2fr; gap: 20px; }
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
        .viewport {
            background: #000;
            border-radius: 10px;
            height: 500px;
            display: flex;
            align-items: center;
            justify-content: center;
            border: 2px solid rgba(118, 185, 0, 0.3);
        }
        .viewport-placeholder {
            text-align: center;
            color: #666;
        }
        .session-list { margin-top: 20px; }
        .session-item {
            background: rgba(0, 0, 0, 0.2);
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 10px;
            border-left: 4px solid #76b900;
        }
        .session-item.running { border-left-color: #28a745; }
        .status-badge {
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8em;
        }
        .status-running { background: #28a745; }
        .status-stopped { background: #6c757d; }
        .links { display: flex; gap: 10px; margin-top: 20px; flex-wrap: wrap; }
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
                <h1>🤖 NVIDIA ISAAC Sim</h1>
                <p class="subtitle">Photorealistic robotics simulation platform</p>
            </div>
            <div class="links">
                <a href="/api" class="link">📚 API Docs</a>
                <a href="http://localhost:8009" class="link">🔄 SVO Converter</a>
                <a href="http://localhost:8011" class="link">🧪 ISAAC Lab</a>
            </div>
        </header>
        
        <div class="grid">
            <div class="card">
                <h2>⚙️ Simulation Config</h2>
                <div class="form-group">
                    <label>Scene</label>
                    <select id="scene">
                        <option value="warehouse">Warehouse</option>
                        <option value="hospital">Hospital</option>
                        <option value="office">Office</option>
                        <option value="simple_room">Simple Room</option>
                        <option value="custom">Custom USD</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Robot Type</label>
                    <select id="robot">
                        <option value="carter">Carter</option>
                        <option value="jetbot">JetBot</option>
                        <option value="franka">Franka Panda</option>
                        <option value="ur10">UR10</option>
                        <option value="spot">Boston Dynamics Spot</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>ROSBAG Playback</label>
                    <select id="rosbag">
                        <option value="">None</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Physics FPS</label>
                    <input type="number" id="physicsFps" value="60">
                </div>
                <button class="btn" onclick="startSimulation()">▶️ Start Simulation</button>
                <button class="btn btn-stop" onclick="stopSimulation()">⏹️ Stop</button>
                
                <div class="session-list" id="sessionList">
                    <h3 style="margin: 20px 0 10px; color: #76b900;">Active Sessions</h3>
                </div>
            </div>
            
            <div class="card">
                <h2>🖥️ Simulation Viewport</h2>
                <div class="viewport" id="viewport">
                    <div class="viewport-placeholder">
                        <p style="font-size: 4em;">🤖</p>
                        <p style="margin-top: 10px;">Start a simulation to view</p>
                        <p style="color: #555; font-size: 0.9em;">Requires ISAAC Sim runtime</p>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        async function loadRosbags() {
            try {
                const response = await fetch('/rosbags');
                const rosbags = await response.json();
                const select = document.getElementById('rosbag');
                rosbags.forEach(bag => {
                    const option = document.createElement('option');
                    option.value = bag.name;
                    option.textContent = bag.name;
                    select.appendChild(option);
                });
            } catch (e) { console.error(e); }
        }
        
        async function startSimulation() {
            const config = {
                scene_name: document.getElementById('scene').value,
                robot_type: document.getElementById('robot').value,
                rosbag_file: document.getElementById('rosbag').value || null,
                physics_dt: 1 / parseInt(document.getElementById('physicsFps').value)
            };
            
            try {
                const response = await fetch('/simulation/start', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(config)
                });
                const data = await response.json();
                alert('Simulation started: ' + data.session_id);
                refreshSessions();
            } catch (e) { console.error(e); }
        }
        
        async function stopSimulation() {
            try {
                await fetch('/simulation/stop', { method: 'POST' });
                refreshSessions();
            } catch (e) { console.error(e); }
        }
        
        async function refreshSessions() {
            try {
                const response = await fetch('/sessions');
                const sessions = await response.json();
                const list = document.getElementById('sessionList');
                list.innerHTML = '<h3 style="margin: 20px 0 10px; color: #76b900;">Active Sessions</h3>';
                sessions.forEach(s => {
                    list.innerHTML += `
                        <div class="session-item ${s.status}">
                            <strong>${s.session_id}</strong>
                            <span class="status-badge status-${s.status}">${s.status}</span>
                            <p style="color: #888; font-size: 0.9em;">${s.scene_name} - ${s.robot_type}</p>
                        </div>
                    `;
                });
            } catch (e) { console.error(e); }
        }
        
        loadRosbags();
        refreshSessions();
        setInterval(refreshSessions, 5000);
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
        "service": "isaac-sim",
        "timestamp": datetime.now().isoformat(),
        "sessions_count": len(simulation_sessions)
    }


@app.post("/simulation/start")
async def start_simulation(config: SimulationConfig):
    """Start a new ISAAC Sim simulation session"""
    session_id = str(uuid.uuid4())[:8]
    
    session = {
        "session_id": session_id,
        "status": "running",
        "scene_name": config.scene_name,
        "robot_type": config.robot_type,
        "rosbag_file": config.rosbag_file,
        "physics_dt": config.physics_dt,
        "render_dt": config.render_dt,
        "started_at": datetime.now().isoformat()
    }
    simulation_sessions[session_id] = session
    
    logger.info(f"Started simulation session {session_id}")
    return session


@app.post("/simulation/stop")
async def stop_simulation(session_id: Optional[str] = None):
    """Stop simulation session(s)"""
    if session_id:
        if session_id in simulation_sessions:
            simulation_sessions[session_id]["status"] = "stopped"
            return {"status": "stopped", "session_id": session_id}
        raise HTTPException(status_code=404, detail="Session not found")
    
    for sid in simulation_sessions:
        simulation_sessions[sid]["status"] = "stopped"
    return {"status": "all_stopped"}


@app.get("/sessions")
async def list_sessions():
    """List all simulation sessions"""
    return list(simulation_sessions.values())


@app.get("/rosbags")
async def list_rosbags():
    """List available ROSBAG files"""
    rosbags = []
    for path in ROSBAG_DIR.glob("*.bag"):
        rosbags.append({"name": path.name, "size": path.stat().st_size})
    return rosbags


@app.get("/scenes")
async def list_scenes():
    """List available simulation scenes"""
    return [
        {"id": "warehouse", "name": "Warehouse", "description": "Industrial warehouse environment"},
        {"id": "hospital", "name": "Hospital", "description": "Hospital corridors and rooms"},
        {"id": "office", "name": "Office", "description": "Office building environment"},
        {"id": "simple_room", "name": "Simple Room", "description": "Basic room for testing"}
    ]


@app.get("/robots")
async def list_robots():
    """List available robot models"""
    return [
        {"id": "carter", "name": "Carter", "type": "mobile", "description": "NVIDIA Carter mobile robot"},
        {"id": "jetbot", "name": "JetBot", "type": "mobile", "description": "NVIDIA JetBot"},
        {"id": "franka", "name": "Franka Panda", "type": "arm", "description": "Franka Emika Panda arm"},
        {"id": "ur10", "name": "UR10", "type": "arm", "description": "Universal Robots UR10"},
        {"id": "spot", "name": "Spot", "type": "quadruped", "description": "Boston Dynamics Spot"}
    ]


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8010))
    uvicorn.run(app, host="0.0.0.0", port=port)
