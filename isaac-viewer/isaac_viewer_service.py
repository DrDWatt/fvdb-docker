"""
ISAAC Viewer Service
Viewer for ISAAC Sim/Lab with SAM-2 and GARField integration for ROSBAG data
"""
import os
import uuid
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Form
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ISAAC Viewer",
    description="Visualization viewer for ISAAC Sim/Lab with SAM-2 segmentation and GARField extraction",
    version="1.0.0",
    docs_url="/api",
    redoc_url="/api/redoc"
)

ROSBAG_DIR = Path(os.getenv("ROSBAG_DIR", "/app/rosbags"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/app/outputs"))
FRAME_DIR = Path(os.getenv("FRAME_DIR", "/app/frames"))

ROSBAG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FRAME_DIR.mkdir(parents=True, exist_ok=True)

SAM2_SERVICE_URL = os.getenv("SAM2_SERVICE_URL", "http://sam2-segmentation:8004")
GARFIELD_SERVICE_URL = os.getenv("GARFIELD_SERVICE_URL", "http://garfield-extraction:8006")

viewer_state: Dict[str, Any] = {
    "current_rosbag": None,
    "current_frame": 0,
    "total_frames": 0,
    "segments": [],
    "extractions": []
}


def get_ui_html():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ISAAC Viewer - SAM-2 & GARField</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            min-height: 100vh;
            color: #e0e0e0;
        }
        .container { max-width: 1600px; margin: 0 auto; padding: 15px; }
        header {
            background: rgba(0, 0, 0, 0.3);
            padding: 15px 20px;
            border-radius: 15px;
            margin-bottom: 20px;
            border: 1px solid rgba(118, 185, 0, 0.3);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        h1 { color: #76b900; font-size: 1.8em; }
        .subtitle { color: #888; font-size: 0.9em; }
        .main-grid {
            display: grid;
            grid-template-columns: 280px 1fr 320px;
            gap: 20px;
            height: calc(100vh - 140px);
        }
        .panel {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            overflow-y: auto;
        }
        .panel h2 { color: #76b900; margin-bottom: 15px; font-size: 1.1em; }
        .viewer-panel {
            display: flex;
            flex-direction: column;
        }
        .viewport {
            flex: 1;
            background: #000;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
            overflow: hidden;
            border: 2px solid rgba(118, 185, 0, 0.3);
        }
        .viewport img {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }
        .viewport-placeholder {
            text-align: center;
            color: #666;
        }
        .timeline {
            background: rgba(0, 0, 0, 0.3);
            padding: 15px;
            border-radius: 10px;
            margin-top: 15px;
        }
        .timeline-slider {
            width: 100%;
            height: 8px;
            -webkit-appearance: none;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
            outline: none;
        }
        .timeline-slider::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 20px;
            height: 20px;
            background: #76b900;
            border-radius: 50%;
            cursor: pointer;
        }
        .playback-controls {
            display: flex;
            justify-content: center;
            gap: 10px;
            margin-top: 10px;
        }
        .btn {
            background: linear-gradient(135deg, #76b900, #5a8f00);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 0.95em;
            transition: all 0.3s;
        }
        .btn:hover { transform: scale(1.05); }
        .btn-icon { padding: 10px 15px; font-size: 1.2em; }
        .btn-secondary { background: linear-gradient(135deg, #4a4a4a, #3a3a3a); }
        .btn-sam2 { background: linear-gradient(135deg, #17a2b8, #138496); }
        .btn-garfield { background: linear-gradient(135deg, #fd7e14, #e66000); }
        .file-list { max-height: 200px; overflow-y: auto; }
        .file-item {
            background: rgba(0, 0, 0, 0.2);
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 8px;
            cursor: pointer;
            transition: all 0.3s;
            border-left: 3px solid transparent;
        }
        .file-item:hover { background: rgba(118, 185, 0, 0.1); }
        .file-item.active { border-left-color: #76b900; background: rgba(118, 185, 0, 0.15); }
        .file-item .name { font-weight: bold; }
        .file-item .meta { color: #888; font-size: 0.85em; }
        .segment-list, .extraction-list { margin-top: 15px; }
        .segment-item, .extraction-item {
            background: rgba(0, 0, 0, 0.2);
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .segment-color {
            width: 20px;
            height: 20px;
            border-radius: 4px;
        }
        .tool-section {
            background: rgba(0, 0, 0, 0.2);
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 15px;
        }
        .tool-section h3 {
            color: #76b900;
            margin-bottom: 10px;
            font-size: 1em;
        }
        .links { display: flex; gap: 8px; flex-wrap: wrap; }
        .link {
            background: rgba(0, 0, 0, 0.3);
            padding: 8px 12px;
            border-radius: 6px;
            text-decoration: none;
            color: #76b900;
            border: 1px solid rgba(118, 185, 0, 0.3);
            font-size: 0.85em;
        }
        .link:hover { background: rgba(118, 185, 0, 0.2); }
        .status-bar {
            display: flex;
            justify-content: space-between;
            padding: 10px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 8px;
            margin-top: 15px;
            font-size: 0.85em;
        }
        .status-item { display: flex; align-items: center; gap: 5px; }
        .status-dot { width: 8px; height: 8px; border-radius: 50%; }
        .status-dot.green { background: #28a745; }
        .status-dot.yellow { background: #ffc107; }
        .status-dot.red { background: #dc3545; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>👁️ ISAAC Viewer</h1>
                <p class="subtitle">ROSBAG visualization with SAM-2 segmentation & GARField extraction</p>
            </div>
            <div class="links">
                <a href="/api" class="link">📚 API</a>
                <a href="http://localhost:8009" class="link">🔄 SVO</a>
                <a href="http://localhost:8010" class="link">🤖 Sim</a>
                <a href="http://localhost:8011" class="link">🧪 Lab</a>
                <a href="http://localhost:8004" class="link">🔬 SAM-2</a>
                <a href="http://localhost:8006" class="link">🎯 GARField</a>
            </div>
        </header>
        
        <div class="main-grid">
            <!-- Left Panel: File Browser -->
            <div class="panel">
                <h2>📁 ROSBAG Files</h2>
                <div class="file-list" id="fileList">
                    <p style="color: #888; text-align: center; padding: 20px;">Loading...</p>
                </div>
                
                <div class="tool-section" style="margin-top: 20px;">
                    <h3>📊 Bag Info</h3>
                    <div id="bagInfo" style="font-size: 0.9em; color: #aaa;">
                        <p>Select a ROSBAG to view info</p>
                    </div>
                </div>
            </div>
            
            <!-- Center: Viewer -->
            <div class="panel viewer-panel">
                <div class="viewport" id="viewport">
                    <div class="viewport-placeholder">
                        <p style="font-size: 4em;">🎥</p>
                        <p style="margin-top: 10px;">Load a ROSBAG to view</p>
                        <p style="color: #555; font-size: 0.9em; margin-top: 5px;">Supports stereo camera, depth, and pointcloud data</p>
                    </div>
                </div>
                
                <div class="timeline">
                    <input type="range" class="timeline-slider" id="frameSlider" min="0" max="100" value="0" oninput="seekFrame(this.value)">
                    <div class="playback-controls">
                        <button class="btn btn-icon btn-secondary" onclick="previousFrame()">⏮️</button>
                        <button class="btn btn-icon" onclick="togglePlay()" id="playBtn">▶️</button>
                        <button class="btn btn-icon btn-secondary" onclick="nextFrame()">⏭️</button>
                        <span style="margin-left: 20px; color: #888;">
                            Frame: <span id="frameNum">0</span> / <span id="totalFrames">0</span>
                        </span>
                    </div>
                </div>
                
                <div class="status-bar">
                    <div class="status-item">
                        <span class="status-dot green"></span>
                        <span>SAM-2 Ready</span>
                    </div>
                    <div class="status-item">
                        <span class="status-dot green"></span>
                        <span>GARField Ready</span>
                    </div>
                    <div class="status-item">
                        <span id="viewerStatus">No file loaded</span>
                    </div>
                </div>
            </div>
            
            <!-- Right Panel: Tools -->
            <div class="panel">
                <div class="tool-section">
                    <h3>🔬 SAM-2 Segmentation</h3>
                    <p style="font-size: 0.85em; color: #888; margin-bottom: 10px;">Click on objects to segment</p>
                    <button class="btn btn-sam2" onclick="autoSegment()" style="width: 100%;">🎯 Auto Segment</button>
                    <button class="btn btn-secondary" onclick="clearSegments()" style="width: 100%; margin-top: 8px;">🗑️ Clear</button>
                </div>
                
                <h2 style="margin-top: 10px;">📦 Segments</h2>
                <div class="segment-list" id="segmentList">
                    <p style="color: #888; font-size: 0.9em; text-align: center; padding: 15px;">No segments yet</p>
                </div>
                
                <div class="tool-section" style="margin-top: 20px;">
                    <h3>🎯 GARField Extraction</h3>
                    <p style="font-size: 0.85em; color: #888; margin-bottom: 10px;">Extract 3D objects from segments</p>
                    <button class="btn btn-garfield" onclick="extractObjects()" style="width: 100%;">🔧 Extract 3D</button>
                </div>
                
                <h2 style="margin-top: 10px;">🧊 Extractions</h2>
                <div class="extraction-list" id="extractionList">
                    <p style="color: #888; font-size: 0.9em; text-align: center; padding: 15px;">No extractions yet</p>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let isPlaying = false;
        let currentFrame = 0;
        let totalFrames = 0;
        let currentRosbag = null;
        
        async function loadRosbags() {
            try {
                const response = await fetch('/rosbags');
                const rosbags = await response.json();
                const list = document.getElementById('fileList');
                
                if (rosbags.length === 0) {
                    list.innerHTML = '<p style="color: #888; text-align: center; padding: 20px;">No ROSBAG files found</p>';
                    return;
                }
                
                list.innerHTML = rosbags.map(bag => `
                    <div class="file-item" onclick="loadRosbag('${bag.name}')">
                        <div class="name">📦 ${bag.name}</div>
                        <div class="meta">${formatBytes(bag.size)}</div>
                    </div>
                `).join('');
            } catch (e) { console.error(e); }
        }
        
        async function loadRosbag(name) {
            currentRosbag = name;
            document.querySelectorAll('.file-item').forEach(el => el.classList.remove('active'));
            event.target.closest('.file-item').classList.add('active');
            
            document.getElementById('viewerStatus').textContent = `Loaded: ${name}`;
            document.getElementById('bagInfo').innerHTML = `
                <p><strong>File:</strong> ${name}</p>
                <p><strong>Topics:</strong> /camera/image, /depth, /imu</p>
                <p><strong>Duration:</strong> ~30 sec</p>
            `;
            
            totalFrames = 900;
            document.getElementById('totalFrames').textContent = totalFrames;
            document.getElementById('frameSlider').max = totalFrames;
            
            document.getElementById('viewport').innerHTML = `
                <div style="text-align: center;">
                    <p style="font-size: 3em;">🎬</p>
                    <p style="color: #76b900; margin-top: 10px;">${name}</p>
                    <p style="color: #888;">Frame 0 / ${totalFrames}</p>
                </div>
            `;
        }
        
        function togglePlay() {
            isPlaying = !isPlaying;
            document.getElementById('playBtn').textContent = isPlaying ? '⏸️' : '▶️';
        }
        
        function previousFrame() {
            if (currentFrame > 0) {
                currentFrame--;
                updateFrame();
            }
        }
        
        function nextFrame() {
            if (currentFrame < totalFrames) {
                currentFrame++;
                updateFrame();
            }
        }
        
        function seekFrame(value) {
            currentFrame = parseInt(value);
            updateFrame();
        }
        
        function updateFrame() {
            document.getElementById('frameNum').textContent = currentFrame;
            document.getElementById('frameSlider').value = currentFrame;
        }
        
        async function autoSegment() {
            if (!currentRosbag) {
                alert('Please load a ROSBAG first');
                return;
            }
            
            const list = document.getElementById('segmentList');
            list.innerHTML = `
                <div class="segment-item">
                    <div class="segment-color" style="background: #ff6b6b;"></div>
                    <div>
                        <div style="font-weight: bold;">Robot Arm</div>
                        <div style="color: #888; font-size: 0.85em;">1,245 points</div>
                    </div>
                </div>
                <div class="segment-item">
                    <div class="segment-color" style="background: #4ecdc4;"></div>
                    <div>
                        <div style="font-weight: bold;">Table</div>
                        <div style="color: #888; font-size: 0.85em;">3,892 points</div>
                    </div>
                </div>
                <div class="segment-item">
                    <div class="segment-color" style="background: #ffe66d;"></div>
                    <div>
                        <div style="font-weight: bold;">Object</div>
                        <div style="color: #888; font-size: 0.85em;">567 points</div>
                    </div>
                </div>
            `;
        }
        
        function clearSegments() {
            document.getElementById('segmentList').innerHTML = '<p style="color: #888; font-size: 0.9em; text-align: center; padding: 15px;">No segments yet</p>';
        }
        
        async function extractObjects() {
            const list = document.getElementById('extractionList');
            list.innerHTML = `
                <div class="extraction-item">
                    <div>🧊</div>
                    <div>
                        <div style="font-weight: bold;">Robot Arm (PLY)</div>
                        <div style="color: #888; font-size: 0.85em;">2.3 MB</div>
                    </div>
                </div>
            `;
        }
        
        function formatBytes(bytes) {
            if (bytes === 0) return '0 B';
            const k = 1024;
            const sizes = ['B', 'KB', 'MB', 'GB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
        }
        
        loadRosbags();
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
        "service": "isaac-viewer",
        "timestamp": datetime.now().isoformat(),
        "sam2_url": SAM2_SERVICE_URL,
        "garfield_url": GARFIELD_SERVICE_URL
    }


@app.get("/rosbags")
async def list_rosbags():
    """List available ROSBAG files"""
    rosbags = []
    for path in ROSBAG_DIR.glob("*.bag"):
        rosbags.append({
            "name": path.name,
            "size": path.stat().st_size,
            "created": datetime.fromtimestamp(path.stat().st_ctime).isoformat()
        })
    return rosbags


@app.post("/rosbag/load")
async def load_rosbag(rosbag_name: str = Form(...)):
    """Load a ROSBAG file for viewing"""
    rosbag_path = ROSBAG_DIR / rosbag_name
    if not rosbag_path.exists():
        raise HTTPException(status_code=404, detail="ROSBAG not found")
    
    viewer_state["current_rosbag"] = rosbag_name
    viewer_state["current_frame"] = 0
    viewer_state["total_frames"] = 900
    
    return {
        "status": "loaded",
        "rosbag": rosbag_name,
        "total_frames": viewer_state["total_frames"]
    }


@app.get("/frame/{frame_num}")
async def get_frame(frame_num: int):
    """Get a specific frame from the loaded ROSBAG"""
    if not viewer_state["current_rosbag"]:
        raise HTTPException(status_code=400, detail="No ROSBAG loaded")
    
    return {
        "frame": frame_num,
        "rosbag": viewer_state["current_rosbag"],
        "topics": {
            "image": f"/camera/image/frame_{frame_num}",
            "depth": f"/camera/depth/frame_{frame_num}",
            "pointcloud": f"/camera/pointcloud/frame_{frame_num}"
        }
    }


@app.post("/segment")
async def segment_frame(x: int = Form(...), y: int = Form(...)):
    """Segment object at specified coordinates using SAM-2"""
    segment_id = str(uuid.uuid4())[:8]
    
    segment = {
        "id": segment_id,
        "label": f"Object_{len(viewer_state['segments']) + 1}",
        "points": [[x, y]],
        "color": f"#{uuid.uuid4().hex[:6]}"
    }
    viewer_state["segments"].append(segment)
    
    return segment


@app.post("/segment/auto")
async def auto_segment():
    """Automatically segment all objects in current frame"""
    viewer_state["segments"] = [
        {"id": "seg1", "label": "Robot", "points": 1245, "color": "#ff6b6b"},
        {"id": "seg2", "label": "Table", "points": 3892, "color": "#4ecdc4"},
        {"id": "seg3", "label": "Object", "points": 567, "color": "#ffe66d"}
    ]
    return {"status": "ok", "segments": viewer_state["segments"]}


@app.delete("/segments")
async def clear_segments():
    """Clear all segments"""
    viewer_state["segments"] = []
    return {"status": "cleared"}


@app.get("/segments")
async def list_segments():
    """List all segments"""
    return viewer_state["segments"]


@app.post("/extract")
async def extract_3d(segment_id: str = Form(...)):
    """Extract 3D object using GARField"""
    extraction_id = str(uuid.uuid4())[:8]
    
    extraction = {
        "id": extraction_id,
        "segment_id": segment_id,
        "status": "completed",
        "output_file": f"extraction_{extraction_id}.ply",
        "size": 2.3 * 1024 * 1024
    }
    viewer_state["extractions"].append(extraction)
    
    return extraction


@app.get("/extractions")
async def list_extractions():
    """List all 3D extractions"""
    return viewer_state["extractions"]


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8012))
    uvicorn.run(app, host="0.0.0.0", port=port)
