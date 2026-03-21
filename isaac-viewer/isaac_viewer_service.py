"""
ISAAC Viewer Service
SVO/ROSBAG Viewer with SVO to Gaussian Splat workflow pipeline
"""
import os
import io
import json
import base64
import asyncio
import logging
import subprocess
import shutil
import requests
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
import zipfile
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ISAAC Viewer",
    description="SVO/ROSBAG Viewer with SVO to Gaussian Splat workflow",
    version="2.0.0",
    docs_url="/api",
    redoc_url="/api/redoc"
)

# Directories
SVO_DIR = Path(os.getenv("SVO_DIR", "/app/svo"))
ROSBAG_DIR = Path(os.getenv("ROSBAG_DIR", "/app/rosbags"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/app/outputs"))
FRAME_DIR = Path(os.getenv("FRAME_DIR", "/app/frames"))

for d in [SVO_DIR, ROSBAG_DIR, OUTPUT_DIR, FRAME_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Service URLs
COLMAP_SERVICE_URL = os.getenv("COLMAP_SERVICE_URL", "http://colmap-processor:8003")
CUSFM_SERVICE_URL = os.getenv("CUSFM_SERVICE_URL", "http://cusfm-service:8014")
TRAINING_SERVICE_URL = os.getenv("TRAINING_SERVICE_URL", "http://fvdb-training-gpu:8000")
FVDB_VIEWER_URL = os.getenv("FVDB_VIEWER_URL", "http://fvdb-viewer:8085")
DEPTH_SERVICE_URL = os.getenv("DEPTH_SERVICE_URL", "http://host.docker.internal:8013")

# State
current_file: Optional[str] = None
current_frame_idx: int = 0
total_frames: int = 0
current_image: Optional[np.ndarray] = None
zoom_level: float = 1.0
view_mode: str = "combined"  # combined, left, right, depth
video_capture: Any = None  # Store video capture object
video_capture_file: Optional[str] = None  # Track which file is open
raw_frame_cache: Dict[int, np.ndarray] = {}  # Cache raw frames by index
raw_frame_cache_file: Optional[str] = None  # Track which file cache is for

# Workflow state
active_workflows: Dict[str, Dict] = {}

# Try to import OpenCV for frame handling
try:
    import cv2
    CV2_AVAILABLE = True
    logger.info("OpenCV available for frame processing")
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not available - using simulated frames")

# Try to import PIL
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def get_available_files() -> List[Dict]:
    """Get list of available SVO and ROSBAG files"""
    files = []
    
    # SVO files
    for ext in ['*.svo', '*.svo2']:
        for path in SVO_DIR.glob(ext):
            files.append({
                "name": path.name,
                "type": "svo",
                "size": path.stat().st_size,
                "path": str(path)
            })
    
    # ROSBAG files
    for ext in ['*.bag', '*.db3', '*.mcap']:
        for path in ROSBAG_DIR.glob(ext):
            files.append({
                "name": path.name,
                "type": "rosbag",
                "size": path.stat().st_size,
                "path": str(path)
            })
    
    return sorted(files, key=lambda x: x["name"])


def extract_frame_from_svo(file_path: str, frame_idx: int) -> Optional[np.ndarray]:
    """Extract a frame from SVO file"""
    global current_image
    
    if not CV2_AVAILABLE:
        return generate_simulated_frame(frame_idx)
    
    try:
        # Try to open as video file (some SVO files are compatible)
        cap = cv2.VideoCapture(file_path)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()
            if ret:
                # Convert BGR to RGB
                current_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return current_image
    except Exception as e:
        logger.warning(f"Could not read SVO as video: {e}")
    
    # Fall back to simulated frame
    return generate_simulated_frame(frame_idx)


def generate_simulated_frame(frame_idx: int, width: int = 1280, height: int = 720, view: str = "left") -> np.ndarray:
    """Generate a simulated frame for demo purposes - optimized with numpy"""
    global current_image
    
    # Use numpy vectorized operations for speed
    file_hash = hash(current_file or "default") % 360
    view_offset = {"left": 0, "right": 30, "depth": 60}.get(view, 0)
    
    # Create gradient using numpy (much faster than loop)
    y_vals = np.arange(height).reshape(-1, 1)
    hue = ((file_hash + view_offset + y_vals // 10) % 180).astype(np.uint8)
    sat = np.full((height, 1), 100 if view != "depth" else 50, dtype=np.uint8)
    val = (50 + (y_vals * 100 // height)).astype(np.uint8)
    
    # Broadcast to full width
    hsv = np.zeros((height, width, 3), dtype=np.uint8)
    hsv[:, :, 0] = np.broadcast_to(hue, (height, width))
    hsv[:, :, 1] = np.broadcast_to(sat, (height, width))
    hsv[:, :, 2] = np.broadcast_to(val, (height, width))
    
    if CV2_AVAILABLE:
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    else:
        img = hsv
    
    # Add moving elements based on frame
    offset = (frame_idx * 5) % width
    stereo_offset = 10 if view == "right" else 0
    
    # Draw some rectangles to simulate objects
    if CV2_AVAILABLE:
        if view == "depth":
            # Simple depth gradient
            for i in range(5):
                depth_val = int(50 + i * 40)
                y_pos = 150 + i * 100
                cv2.rectangle(img, (100, y_pos), (width - 100, y_pos + 60), (depth_val, depth_val, 255 - depth_val), -1)
            cv2.putText(img, "DEPTH MAP (Simulated)", (width // 2 - 120, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            # Simple scene with moving objects
            cv2.rectangle(img, (100 + offset % 200 + stereo_offset, 250), (200 + offset % 200 + stereo_offset, 400), (150, 150, 150), -1)
            cv2.rectangle(img, (0, 550), (width, height), (60, 60, 60), -1)
        
        # Frame info overlay
        view_label = view.upper()
        cv2.putText(img, f"{view_label} | Frame: {frame_idx}", (20, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if current_file:
            short_name = current_file[:40] + "..." if len(current_file) > 40 else current_file
            cv2.putText(img, short_name, (20, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(img, "(Demo - file not directly readable)", (20, 55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 0), 1)
    
    current_image = img
    return img


def get_current_frame() -> Optional[np.ndarray]:
    """Get current frame as numpy array"""
    global current_image
    
    if current_image is not None:
        return current_image
    
    if current_file:
        file_path = None
        for d in [SVO_DIR, ROSBAG_DIR]:
            p = d / current_file
            if p.exists():
                file_path = str(p)
                break
        
        if file_path and file_path.endswith(('.svo', '.svo2')):
            return extract_frame_from_svo(file_path, current_frame_idx)
    
    return generate_simulated_frame(current_frame_idx)


def frame_to_png(frame: np.ndarray) -> bytes:
    """Convert numpy frame to PNG bytes"""
    if CV2_AVAILABLE:
        # Convert RGB to BGR for OpenCV
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.png', bgr)
        return buffer.tobytes()
    elif PIL_AVAILABLE:
        img = Image.fromarray(frame)
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return buf.getvalue()
    else:
        # Return empty PNG
        return b''


def frame_to_base64(frame: np.ndarray) -> str:
    """Convert numpy frame to base64 string"""
    png_bytes = frame_to_png(frame)
    return base64.b64encode(png_bytes).decode('utf-8')


def filter_sharp_frames(images_dir: Path, images_right_dir: Path = None,
                        workflow_id: str = "") -> int:
    """Filter blurry frames using sharp-frames library (outlier-removal method).
    Keeps stereo pairs in sync by filtering left images and retaining matching right images.
    Returns the number of frames kept after filtering."""
    filtered_dir = images_dir.parent / "images_sharp_filtered"
    filtered_dir.mkdir(exist_ok=True, parents=True)

    try:
        result = subprocess.run(
            [
                "sharp-frames",
                str(images_dir),
                str(filtered_dir),
                "--selection-method", "outlier-removal",
                "--outlier-sensitivity", "50",
                "--force-overwrite",
            ],
            capture_output=True, text=True, timeout=300
        )

        if result.returncode != 0:
            logger.warning(f"[{workflow_id}] sharp-frames failed: {result.stderr[:500]}")
            if filtered_dir.exists():
                shutil.rmtree(filtered_dir)
            return len(list(images_dir.glob("*.jpg"))) + len(list(images_dir.glob("*.png")))

        # Determine which original filenames were kept
        kept_names = set(f.name for f in filtered_dir.iterdir()
                         if f.suffix.lower() in ('.jpg', '.jpeg', '.png'))
        num_kept = len(kept_names)
        total_original = len(list(images_dir.glob("*.jpg"))) + len(list(images_dir.glob("*.png")))
        logger.info(f"[{workflow_id}] sharp-frames kept {num_kept}/{total_original} frames")

        if num_kept == 0:
            logger.warning(f"[{workflow_id}] sharp-frames kept 0 frames, skipping filter")
            shutil.rmtree(filtered_dir)
            return total_original

        # Replace left images with filtered set, re-numbered sequentially
        for f in images_dir.iterdir():
            if f.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                f.unlink()

        # Also filter right images to keep stereo pairs in sync
        kept_right_names = set()
        if images_right_dir and images_right_dir.exists():
            for f in images_right_dir.iterdir():
                if f.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                    if f.name in kept_names:
                        kept_right_names.add(f.name)
                    else:
                        f.unlink()

        # Re-number left frames sequentially
        for idx, name in enumerate(sorted(kept_names)):
            src = filtered_dir / name
            dst = images_dir / f"frame_{idx:04d}.jpg"
            shutil.move(str(src), str(dst))

        # Re-number right frames sequentially (matching left order)
        if images_right_dir and images_right_dir.exists() and kept_right_names:
            temp_right = images_right_dir.parent / "images_right_temp"
            temp_right.mkdir(exist_ok=True)
            for name in sorted(kept_right_names):
                shutil.move(str(images_right_dir / name), str(temp_right / name))
            for idx, name in enumerate(sorted(kept_right_names)):
                shutil.move(str(temp_right / name), str(images_right_dir / f"frame_{idx:04d}.jpg"))
            shutil.rmtree(temp_right, ignore_errors=True)

        shutil.rmtree(filtered_dir, ignore_errors=True)
        return num_kept

    except FileNotFoundError:
        logger.warning(f"[{workflow_id}] sharp-frames not installed, skipping blur filter")
        return len(list(images_dir.glob("*.jpg"))) + len(list(images_dir.glob("*.png")))
    except subprocess.TimeoutExpired:
        logger.warning(f"[{workflow_id}] sharp-frames timed out, skipping blur filter")
        shutil.rmtree(filtered_dir, ignore_errors=True)
        return len(list(images_dir.glob("*.jpg"))) + len(list(images_dir.glob("*.png")))


def get_ui_html():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ISAAC Viewer - SVO to Gaussian Splat</title>
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
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px 20px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 10px;
            margin-bottom: 15px;
            border: 1px solid rgba(118, 185, 0, 0.3);
        }
        .logo { color: #76b900; font-size: 1.5em; font-weight: bold; }
        .nav-links { display: flex; gap: 10px; }
        .nav-btn {
            background: rgba(118, 185, 0, 0.2);
            border: 1px solid #76b900;
            color: #76b900;
            padding: 8px 15px;
            border-radius: 5px;
            cursor: pointer;
            text-decoration: none;
            font-size: 0.85em;
        }
        .nav-btn:hover { background: rgba(118, 185, 0, 0.4); }
        .main-layout {
            display: grid;
            grid-template-columns: 280px 1fr 300px;
            gap: 15px;
            height: calc(100vh - 130px);
        }
        .panel {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            padding: 15px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            overflow-y: auto;
        }
        .panel h2 {
            color: #76b900;
            font-size: 1em;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .btn {
            background: linear-gradient(135deg, #76b900, #5a8f00);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.9em;
            width: 100%;
            margin-bottom: 10px;
            transition: all 0.3s;
        }
        .btn:hover { transform: scale(1.02); box-shadow: 0 4px 15px rgba(118, 185, 0, 0.3); }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
        .btn-secondary { background: linear-gradient(135deg, #4a4a4a, #3a3a3a); }
        .btn-warning { background: linear-gradient(135deg, #ffc107, #e0a800); color: #000; }
        .btn-danger { background: linear-gradient(135deg, #dc3545, #c82333); }
        .file-list { max-height: 300px; overflow-y: auto; }
        .file-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 6px;
            margin-bottom: 8px;
            cursor: pointer;
            transition: all 0.2s;
        }
        .file-item:hover { background: rgba(118, 185, 0, 0.2); }
        .file-item.active { background: rgba(118, 185, 0, 0.3); border-left: 3px solid #76b900; }
        .file-item .name { font-size: 0.85em; word-break: break-all; }
        .file-item .meta { font-size: 0.75em; color: #888; }
        .file-item .type-badge {
            font-size: 0.7em;
            padding: 2px 6px;
            border-radius: 3px;
            background: #76b900;
            color: #000;
        }
        .file-item .type-badge.rosbag { background: #17a2b8; color: #fff; }
        .file-item .delete-btn {
            background: #dc3545;
            border: none;
            color: white;
            padding: 2px 6px;
            border-radius: 3px;
            cursor: pointer;
            font-size: 0.7em;
            margin-left: 5px;
        }
        .file-item .delete-btn:hover { background: #c82333; }
        .viewer-container {
            display: flex;
            flex-direction: column;
            height: 100%;
        }
        .viewer-frame {
            flex: 1;
            background: #000;
            border-radius: 8px;
            overflow: hidden;
            position: relative;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .viewer-frame img {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }
        .viewer-frame.click-mode { cursor: crosshair; }
        .viewer-placeholder {
            text-align: center;
            color: #888;
        }
        .viewer-placeholder p { font-size: 3em; margin-bottom: 10px; }
        .controls {
            display: flex;
            gap: 10px;
            align-items: center;
            padding: 10px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 8px;
            margin-top: 10px;
        }
        .controls input[type="range"] { flex: 1; }
        .controls .frame-info { font-size: 0.85em; min-width: 100px; }
        .workflow-step {
            padding: 10px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 6px;
            margin-bottom: 8px;
            border-left: 3px solid rgba(255, 255, 255, 0.2);
            font-size: 0.85em;
        }
        .workflow-step.active { border-left-color: #76b900; background: rgba(118, 185, 0, 0.1); }
        .workflow-step.completed { border-left-color: #28a745; }
        .workflow-step.failed { border-left-color: #dc3545; }
        .workflow-step .step-title { font-weight: bold; margin-bottom: 4px; }
        .workflow-step .step-detail { font-size: 0.8em; color: #888; }
        .workflow-progress {
            width: 100%;
            height: 6px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 3px;
            margin-top: 4px;
            overflow: hidden;
        }
        .workflow-progress .fill {
            height: 100%;
            background: linear-gradient(90deg, #76b900, #5a8f00);
            border-radius: 3px;
            transition: width 0.5s ease;
        }
        .workflow-log {
            background: rgba(0, 0, 0, 0.3);
            border-radius: 6px;
            padding: 8px;
            font-size: 0.75em;
            font-family: monospace;
            max-height: 120px;
            overflow-y: auto;
            color: #aaa;
        }
        .workflow-log .log-line { padding: 2px 0; border-bottom: 1px solid rgba(255,255,255,0.05); }
        .workflow-param {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 6px 0;
            font-size: 0.85em;
        }
        .workflow-param label { color: #aaa; }
        .workflow-param input, .workflow-param select {
            background: rgba(0,0,0,0.3);
            border: 1px solid rgba(255,255,255,0.2);
            color: #fff;
            padding: 4px 8px;
            border-radius: 4px;
            width: 120px;
            font-size: 0.9em;
        }
        .workflow-param input:focus, .workflow-param select:focus { outline: none; border-color: #76b900; }
        .status-bar {
            display: flex;
            justify-content: space-between;
            padding: 8px 15px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 6px;
            margin-top: 10px;
            font-size: 0.8em;
        }
        .status-indicator {
            display: flex;
            align-items: center;
            gap: 5px;
        }
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #28a745;
        }
        .status-dot.warning { background: #ffc107; }
        .status-dot.error { background: #dc3545; }
        .upload-zone {
            border: 2px dashed rgba(118, 185, 0, 0.5);
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            cursor: pointer;
            margin-bottom: 15px;
            transition: all 0.3s;
        }
        .upload-zone:hover { border-color: #76b900; background: rgba(118, 185, 0, 0.1); }
        .upload-zone input { display: none; }
        .mode-toggle {
            display: flex;
            gap: 5px;
            margin-bottom: 10px;
        }
        .mode-btn {
            flex: 1;
            padding: 8px;
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: #fff;
            border-radius: 5px;
            cursor: pointer;
            font-size: 0.8em;
        }
        .mode-btn.active { background: #76b900; color: #000; border-color: #76b900; }
        .loading-overlay {
            position: absolute;
            inset: 0;
            background: rgba(0, 0, 0, 0.85);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            color: #76b900;
            font-size: 1.2em;
            gap: 15px;
        }
        .loading-overlay .progress-container {
            width: 60%;
            max-width: 400px;
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            height: 20px;
            overflow: hidden;
        }
        .loading-overlay .progress-bar {
            height: 100%;
            background: linear-gradient(90deg, #76b900, #5a8f00);
            border-radius: 10px;
            transition: width 0.3s ease;
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        .loading-overlay .loading-text {
            font-size: 0.9em;
            color: #ccc;
        }
        .hidden { display: none !important; }
        .view-layout {
            display: grid;
            gap: 5px;
            height: 100%;
        }
        .view-layout.combined {
            grid-template-columns: 1fr 1fr;
            grid-template-rows: 1fr 1fr;
        }
        .view-layout.single { grid-template-columns: 1fr; grid-template-rows: 1fr; }
        .view-panel {
            background: #000;
            border-radius: 5px;
            overflow: hidden;
            position: relative;
        }
        .view-panel img { width: 100%; height: 100%; object-fit: contain; }
        .view-panel .view-label {
            position: absolute;
            top: 5px;
            left: 5px;
            background: rgba(0,0,0,0.7);
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 0.75em;
            color: #76b900;
        }
        .zoom-controls {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 8px;
            background: rgba(0,0,0,0.3);
            border-radius: 5px;
            margin-top: 5px;
        }
        .zoom-controls button {
            background: #4a4a4a;
            border: none;
            color: white;
            padding: 5px 12px;
            border-radius: 4px;
            cursor: pointer;
        }
        .zoom-controls button:hover { background: #76b900; }
        .zoom-display {
            font-size: 0.85em;
            min-width: 120px;
            text-align: center;
        }
        .view-mode-btns {
            display: flex;
            gap: 5px;
            margin-bottom: 5px;
        }
        .view-mode-btns button {
            flex: 1;
            padding: 6px;
            background: rgba(0,0,0,0.3);
            border: 1px solid rgba(255,255,255,0.2);
            color: #fff;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.75em;
        }
        .view-mode-btns button.active { background: #76b900; color: #000; border-color: #76b900; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="logo">🎥 ISAAC Viewer</div>
            <div class="nav-links">
                <a href="" class="nav-btn" target="_blank" id="navConverter">📤 SVO Converter</a>
                <a href="" class="nav-btn" target="_blank" id="navViewer">🔮 fVDB Viewer</a>
                <a href="/api" class="nav-btn" target="_blank">📚 API</a>
            </div>
        </header>
        
        <div class="main-layout">
            <!-- Left Panel: Files -->
            <div class="panel">
                <h2>📁 Files</h2>
                
                <div class="upload-zone" onclick="document.getElementById('fileInput').click()">
                    <input type="file" id="fileInput" accept=".svo,.svo2,.bag,.db3,.mcap" onchange="uploadFile(this)">
                    <p style="font-size: 1.5em;">📤</p>
                    <p style="font-size: 0.85em;">Upload SVO or ROSBAG</p>
                </div>
                
                <div style="background: rgba(0,0,0,0.2); padding: 10px; border-radius: 6px; margin-bottom: 10px; font-size: 0.75em;">
                    <p style="color: #76b900; margin-bottom: 5px;"><strong>📌 File Types:</strong></p>
                    <p style="color: #aaa; margin-bottom: 3px;"><strong>SVO/SVO2:</strong> ZED camera recordings (stereo video)</p>
                    <p style="color: #aaa;"><strong>BAG/DB3/MCAP:</strong> ROS bag files (robot sensor data)</p>
                    <p style="color: #888; margin-top: 5px; font-style: italic;">Either format works - just select any file to view</p>
                </div>
                
                <div id="uploadProgress" class="hidden" style="margin-bottom: 10px;">
                    <div style="background: rgba(0,0,0,0.3); border-radius: 4px; height: 8px;">
                        <div id="uploadBar" style="background: #76b900; height: 100%; border-radius: 4px; width: 0%;"></div>
                    </div>
                    <p id="uploadStatus" style="font-size: 0.8em; margin-top: 5px;"></p>
                </div>
                
                <div class="file-list" id="fileList">
                    <p style="color: #888; text-align: center; padding: 20px;">Loading files...</p>
                </div>
                
                <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid rgba(255,255,255,0.1);">
                    <h2>📊 File Info</h2>
                    <div id="fileInfo" style="font-size: 0.85em; color: #888;">
                        <p>Select a file to view info</p>
                    </div>
                </div>
            </div>
            
            <!-- Center: Viewer -->
            <div class="viewer-container">
                <div class="view-mode-btns">
                    <button id="viewCombined" class="active" onclick="setViewMode('combined')">📐 Combined</button>
                    <button id="viewLeft" onclick="setViewMode('left')">👁️ Left</button>
                    <button id="viewRight" onclick="setViewMode('right')">👁️ Right</button>
                    <button id="viewDepth" onclick="setViewMode('depth')">🌊 Depth</button>
                </div>
                
                <div class="viewer-frame" id="viewerFrame">
                    <div class="viewer-placeholder" id="placeholder">
                        <p>🎥</p>
                        <p>Load a file to view</p>
                        <p style="font-size: 0.9em; color: #666;">Supports SVO, SVO2, BAG, DB3, MCAP</p>
                    </div>
                    <div id="viewLayout" class="view-layout combined hidden">
                        <div class="view-panel" id="panelLeft">
                            <span class="view-label">📷 Left Camera</span>
                            <img id="imgLeft" onclick="handleFrameClick(event)">
                        </div>
                        <div class="view-panel" id="panelRight">
                            <span class="view-label">📷 Right Camera</span>
                            <img id="imgRight" onclick="handleFrameClick(event)">
                        </div>
                        <div class="view-panel" id="panelDepth" style="grid-column: span 2;">
                            <span class="view-label">🌊 Depth Map</span>
                            <img id="imgDepth" onclick="handleFrameClick(event)">
                        </div>
                    </div>
                    <img id="frameImage" class="hidden" onclick="handleFrameClick(event)">
                    <div id="loadingOverlay" class="loading-overlay hidden">
                        <div style="font-size: 2em;">🎬</div>
                        <div id="loadingTitle">Loading Video...</div>
                        <div class="progress-container">
                            <div id="loadingProgress" class="progress-bar" style="width: 0%;"></div>
                        </div>
                        <div id="loadingText" class="loading-text">Initializing...</div>
                    </div>
                </div>
                
                <div class="zoom-controls">
                    <button onclick="zoomOut()">➖</button>
                    <input type="range" id="zoomSlider" min="50" max="300" value="100" oninput="setZoom(this.value)">
                    <button onclick="zoomIn()">➕</button>
                    <div class="zoom-display">
                        <span id="zoomLevel">1.00x</span> | <span id="depthDisplay">12.0m</span>
                    </div>
                </div>
                
                <div class="controls">
                    <button class="btn btn-secondary" onclick="prevFrame()" style="width: auto; padding: 8px 15px;">⏮️</button>
                    <button id="playBtn" class="btn" onclick="togglePlay()" style="width: auto; padding: 8px 20px;">▶️</button>
                    <button class="btn btn-secondary" onclick="nextFrame()" style="width: auto; padding: 8px 15px;">⏭️</button>
                    <input type="range" id="frameSlider" min="0" max="100" value="0" oninput="seekFrame(this.value)">
                    <div class="frame-info">
                        <span id="currentFrame">0</span> / <span id="totalFrames">0</span>
                    </div>
                </div>
                
                <div class="status-bar">
                    <div class="status-indicator">
                        <span class="status-dot" id="depthStatus"></span>
                        <span>Depth</span>
                    </div>
                    <div class="status-indicator">
                        <span class="status-dot" id="colmapStatus"></span>
                        <span>cuVSLAM</span>
                    </div>
                    <div class="status-indicator">
                        <span class="status-dot" id="trainingStatus"></span>
                        <span>Training</span>
                    </div>
                    <div id="viewerStatus">Ready</div>
                </div>
            </div>
            
            <!-- Right Panel: Workflow -->
            <div class="panel">
                <h2>� SVO to Gaussian Splat</h2>
                <p style="font-size: 0.8em; color: #888; margin-bottom: 10px;">Extract frames, run cuVSLAM, train splat</p>
                
                <div style="background: rgba(0,0,0,0.2); padding: 10px; border-radius: 6px; margin-bottom: 12px;">
                    <div class="workflow-param">
                        <label>Dataset Name</label>
                        <input type="text" id="wfDatasetName" placeholder="my_scene">
                    </div>
                    <div class="workflow-param">
                        <label>FPS</label>
                        <input type="number" id="wfFps" value="2" min="0.5" max="30" step="0.5">
                    </div>
                    <div class="workflow-param">
                        <label>Training Steps</label>
                        <select id="wfSteps">
                            <option value="7000">7K (Quick)</option>
                            <option value="30000" selected>30K (Good)</option>
                            <option value="62200">62K (Best)</option>
                        </select>
                    </div>
                    <div class="workflow-param">
                        <label>Include Depth</label>
                        <input type="checkbox" id="wfIncludeDepth" checked style="width: auto;">
                    </div>
                    <div class="workflow-param">
                        <label title="Monte Carlo Markov Chain optimizer - better quality for complex scenes">MCMC Training</label>
                        <input type="checkbox" id="wfUseMcmc" style="width: auto;">
                    </div>
                    <div class="workflow-param">
                        <label title="Use sharp-frames to remove blurry frames before reconstruction">Filter Blurry Frames</label>
                        <input type="checkbox" id="wfFilterBlur" checked style="width: auto;">
                    </div>
                </div>
                
                <button class="btn" id="wfSuggestBtn" onclick="suggestSettings()" style="background: #2196F3; margin-bottom: 6px;">🔍 Analyze &amp; Suggest Settings</button>
                <div id="wfSuggestionInfo" style="display:none; background: rgba(33,150,243,0.15); border: 1px solid #2196F3; border-radius: 6px; padding: 8px; margin-bottom: 10px; font-size: 0.78em; color: #ccc;">
                    <div id="wfAnalysisText"></div>
                </div>
                <button class="btn" id="wfStartBtn" onclick="startWorkflow()">🚀 Start Full Pipeline</button>
                <button class="btn btn-secondary" id="wfExtractBtn" onclick="extractFramesOnly()">� Extract Frames Only</button>
                
                <p id="wfStatus" style="font-size: 0.8em; color: #888; margin: 10px 0;">Load an SVO file to begin</p>
                
                <h2 style="margin-top: 15px;">� Pipeline Status</h2>
                <div id="wfSteps">
                    <div class="workflow-step" id="stepExtract">
                        <div class="step-title">1. Extract Frames</div>
                        <div class="step-detail" id="stepExtractDetail">Waiting...</div>
                        <div class="workflow-progress"><div class="fill" id="stepExtractProgress" style="width: 0%;"></div></div>
                    </div>
                    <div class="workflow-step" id="stepColmap">
                        <div class="step-title">2. cuVSLAM (Visual SLAM)</div>
                        <div class="step-detail" id="stepColmapDetail">Waiting...</div>
                        <div class="workflow-progress"><div class="fill" id="stepColmapProgress" style="width: 0%;"></div></div>
                    </div>
                    <div class="workflow-step" id="stepTrain">
                        <div class="step-title">3. Train Gaussian Splat</div>
                        <div class="step-detail" id="stepTrainDetail">Waiting...</div>
                        <div class="workflow-progress"><div class="fill" id="stepTrainProgress" style="width: 0%;"></div></div>
                    </div>
                    <div class="workflow-step" id="stepView">
                        <div class="step-title">4. View in fVDB</div>
                        <div class="step-detail" id="stepViewDetail">Waiting...</div>
                    </div>
                </div>
                
                <h2 style="margin-top: 15px;">� Log</h2>
                <div class="workflow-log" id="wfLog">
                    <div class="log-line">Ready. Load an SVO file and start the pipeline.</div>
                </div>
                
                <div style="margin-top: 15px;">
                    <a href="" id="viewSplatLink" class="btn btn-secondary" target="_blank" style="display: none; text-align: center;">🔮 View Splat in fVDB Viewer</a>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // State
        let currentFile = null;
        let currentFrame = 0;
        let totalFrames = 100;
        let isPlaying = false;
        let playInterval = null;
        let viewMode = 'combined';
        let zoomLevel = 1.0;
        let activeWorkflowId = null;
        let workflowPollInterval = null;
        
        // Dynamic host for remote access
        const API_HOST = window.location.origin;
        const COLMAP_API = `http://${window.location.hostname}:8003`;
        const TRAINING_API = `http://${window.location.hostname}:8000`;
        const VIEWER_URL = `http://${window.location.hostname}:8085`;
        
        // Initialize
        document.addEventListener('DOMContentLoaded', () => {
            // Set nav links dynamically for remote access
            document.getElementById('navConverter').href = `http://${window.location.hostname}:8009`;
            document.getElementById('navViewer').href = VIEWER_URL;
            loadFiles();
            checkServiceStatus();
            updateZoomDisplay();
        });
        
        async function checkServiceStatus() {
            // Check Depth service
            try {
                const resp = await fetch('/health');
                const data = await resp.json();
                document.getElementById('depthStatus').className = 'status-dot';
            } catch(e) {
                document.getElementById('depthStatus').className = 'status-dot warning';
            }
            // Check cuVSLAM
            try {
                const resp = await fetch(COLMAP_API + '/health');
                const data2 = await resp.json();
                document.getElementById('colmapStatus').className = (resp.ok && (data2.colmap_available || data2.cuvslam_available)) ? 'status-dot' : 'status-dot warning';
            } catch(e) {
                document.getElementById('colmapStatus').className = 'status-dot error';
            }
            // Check Training
            try {
                const resp = await fetch(TRAINING_API + '/health');
                document.getElementById('trainingStatus').className = resp.ok ? 'status-dot' : 'status-dot warning';
            } catch(e) {
                document.getElementById('trainingStatus').className = 'status-dot error';
            }
        }
        
        async function loadFiles() {
            try {
                const response = await fetch('/files');
                const files = await response.json();
                const list = document.getElementById('fileList');
                
                if (files.length === 0) {
                    list.innerHTML = '<p style="color: #888; text-align: center; padding: 20px;">No files found</p>';
                    return;
                }
                
                list.innerHTML = '';
                files.forEach(file => {
                    const item = document.createElement('div');
                    item.className = 'file-item';
                    item.dataset.name = file.name;
                    
                    const typeClass = file.type === 'svo' ? '' : 'rosbag';
                    
                    const info = document.createElement('div');
                    info.style.flex = '1';
                    info.style.cursor = 'pointer';
                    info.innerHTML = `<div class="name">📦 ${file.name}</div><div class="meta">${formatBytes(file.size)}</div>`;
                    info.onclick = () => loadFile(file.name, item);
                    
                    const badge = document.createElement('span');
                    badge.className = 'type-badge ' + typeClass;
                    badge.textContent = file.type.toUpperCase();
                    
                    const deleteBtn = document.createElement('button');
                    deleteBtn.className = 'delete-btn';
                    deleteBtn.textContent = '🗑️';
                    deleteBtn.onclick = (e) => {
                        e.stopPropagation();
                        deleteFile(file.name);
                    };
                    
                    item.appendChild(info);
                    item.appendChild(badge);
                    item.appendChild(deleteBtn);
                    list.appendChild(item);
                });
            } catch(e) {
                console.error('Failed to load files:', e);
            }
        }
        
        async function deleteFile(filename) {
            if (!confirm('Delete ' + filename + '?')) return;
            
            try {
                const response = await fetch('/file/' + encodeURIComponent(filename), { method: 'DELETE' });
                if (response.ok) {
                    loadFiles();
                    if (currentFile === filename) {
                        currentFile = null;
                        document.getElementById('placeholder').style.display = 'block';
                        document.getElementById('viewLayout').classList.add('hidden');
                        document.getElementById('frameImage').classList.add('hidden');
                    }
                }
            } catch(e) {
                console.error('Failed to delete file:', e);
            }
        }
        
        function formatBytes(bytes) {
            if (bytes === 0) return '0 B';
            const k = 1024;
            const sizes = ['B', 'KB', 'MB', 'GB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
        }
        
        function showLoading(title, text, progress) {
            document.getElementById('loadingOverlay').classList.remove('hidden');
            document.getElementById('loadingTitle').textContent = title || 'Loading...';
            document.getElementById('loadingText').textContent = text || '';
            document.getElementById('loadingProgress').style.width = (progress || 0) + '%';
        }
        
        function hideLoading() {
            document.getElementById('loadingOverlay').classList.add('hidden');
        }
        
        async function loadFile(name, element) {
            // Stop playback
            if (playInterval) {
                clearInterval(playInterval);
                playInterval = null;
                isPlaying = false;
                document.getElementById('playBtn').textContent = '▶️';
            }
            
            // Update UI
            document.querySelectorAll('.file-item').forEach(el => el.classList.remove('active'));
            if (element) element.classList.add('active');
            
            currentFile = name;
            currentFrame = 0;
            
            document.getElementById('viewerStatus').textContent = 'Loading: ' + name;
            showLoading('Loading Video...', 'Opening file: ' + name, 10);
            
            try {
                showLoading('Loading Video...', 'Fetching file metadata...', 30);
                const response = await fetch('/load/' + encodeURIComponent(name));
                const data = await response.json();
                
                if (data.status === 'ok') {
                    totalFrames = data.total_frames || 100;
                    document.getElementById('totalFrames').textContent = totalFrames;
                    document.getElementById('frameSlider').max = Math.max(0, totalFrames - 1);
                    document.getElementById('frameSlider').value = 0;
                    document.getElementById('currentFrame').textContent = 0;
                    
                    document.getElementById('fileInfo').innerHTML = `
                        <p><strong>File:</strong> ${name}</p>
                        <p><strong>Type:</strong> ${data.type || 'Unknown'}</p>
                        <p><strong>Frames:</strong> ${totalFrames}</p>
                    `;
                    
                    showLoading('Loading Video...', 'Loading first frame...', 60);
                    
                    // Load first frame with timeout
                    await renderFrameWithProgress(0);
                    
                    showLoading('Loading Video...', 'Preparing viewer...', 90);
                    
                    document.getElementById('placeholder').style.display = 'none';
                    document.getElementById('viewLayout').classList.remove('hidden');
                    updateViewLayout();
                    
                    showLoading('Loading Video...', 'Ready!', 100);
                    await new Promise(r => setTimeout(r, 300));
                    
                    document.getElementById('viewerStatus').textContent = 'Loaded: ' + name;
                } else {
                    document.getElementById('viewerStatus').textContent = 'Error: ' + (data.detail || 'Unknown error');
                }
            } catch(e) {
                console.error('Failed to load file:', e);
                document.getElementById('viewerStatus').textContent = 'Error loading file: ' + e.message;
            }
            
            hideLoading();
        }
        
        async function renderFrameWithProgress(frameIdx) {
            return new Promise((resolve) => {
                const timestamp = Date.now();
                const baseUrl = '/frame/' + frameIdx + '?t=' + timestamp;
                
                let loadedCount = 0;
                const totalImages = 3;
                
                function onLoad() {
                    loadedCount++;
                    const progress = 60 + Math.round((loadedCount / totalImages) * 30);
                    showLoading('Loading Video...', 'Loading frames... (' + loadedCount + '/' + totalImages + ')', progress);
                    if (loadedCount >= totalImages) {
                        resolve();
                    }
                }
                
                function onError() {
                    loadedCount++;
                    if (loadedCount >= totalImages) {
                        resolve();
                    }
                }
                
                const imgLeft = document.getElementById('imgLeft');
                const imgRight = document.getElementById('imgRight');
                const imgDepth = document.getElementById('imgDepth');
                
                imgLeft.onload = onLoad;
                imgLeft.onerror = onError;
                imgRight.onload = onLoad;
                imgRight.onerror = onError;
                imgDepth.onload = onLoad;
                imgDepth.onerror = onError;
                
                imgLeft.src = baseUrl + '&view=left';
                imgRight.src = baseUrl + '&view=right';
                imgDepth.src = baseUrl + '&view=depth';
                document.getElementById('frameImage').src = baseUrl;
                
                document.getElementById('currentFrame').textContent = frameIdx;
                document.getElementById('frameSlider').value = frameIdx;
                currentFrame = frameIdx;
                
                // Timeout after 10 seconds
                setTimeout(() => {
                    if (loadedCount < totalImages) {
                        resolve();
                    }
                }, 10000);
            });
        }
        
        function setViewMode(mode) {
            viewMode = mode;
            document.querySelectorAll('.view-mode-btns button').forEach(btn => btn.classList.remove('active'));
            document.getElementById('view' + mode.charAt(0).toUpperCase() + mode.slice(1)).classList.add('active');
            updateViewLayout();
        }
        
        function updateViewLayout() {
            const layout = document.getElementById('viewLayout');
            const panelLeft = document.getElementById('panelLeft');
            const panelRight = document.getElementById('panelRight');
            const panelDepth = document.getElementById('panelDepth');
            
            if (viewMode === 'combined') {
                layout.className = 'view-layout combined';
                panelLeft.style.display = 'block';
                panelRight.style.display = 'block';
                panelDepth.style.display = 'block';
                panelDepth.style.gridColumn = 'span 2';
            } else if (viewMode === 'left') {
                layout.className = 'view-layout single';
                panelLeft.style.display = 'block';
                panelRight.style.display = 'none';
                panelDepth.style.display = 'none';
            } else if (viewMode === 'right') {
                layout.className = 'view-layout single';
                panelLeft.style.display = 'none';
                panelRight.style.display = 'block';
                panelDepth.style.display = 'none';
            } else if (viewMode === 'depth') {
                layout.className = 'view-layout single';
                panelLeft.style.display = 'none';
                panelRight.style.display = 'none';
                panelDepth.style.display = 'block';
                panelDepth.style.gridColumn = '1';
            }
        }
        
        function zoomIn() {
            zoomLevel = Math.min(3.0, zoomLevel + 0.1);
            document.getElementById('zoomSlider').value = zoomLevel * 100;
            updateZoomDisplay();
            applyZoom();
        }
        
        function zoomOut() {
            zoomLevel = Math.max(0.5, zoomLevel - 0.1);
            document.getElementById('zoomSlider').value = zoomLevel * 100;
            updateZoomDisplay();
            applyZoom();
        }
        
        function setZoom(value) {
            zoomLevel = value / 100;
            updateZoomDisplay();
            applyZoom();
        }
        
        function updateZoomDisplay() {
            const depthMeters = (0.3 + (3.0 - zoomLevel) * 6).toFixed(1);
            document.getElementById('zoomLevel').textContent = zoomLevel.toFixed(2) + 'x';
            document.getElementById('depthDisplay').textContent = depthMeters + 'm';
        }
        
        function applyZoom() {
            const imgs = document.querySelectorAll('.view-panel img, #frameImage');
            imgs.forEach(img => {
                img.style.transform = 'scale(' + zoomLevel + ')';
                img.style.transformOrigin = 'center center';
            });
        }
        
        async function renderFrame(frameIdx, skipOverlay) {
            try {
                const timestamp = Date.now();
                const ov = skipOverlay ? 0 : 1;
                const baseUrl = '/frame/' + frameIdx + '?t=' + timestamp + '&overlay=' + ov;
                
                // During playback, only update the active view for speed
                if (isPlaying) {
                    if (viewMode === 'combined') {
                        document.getElementById('imgLeft').src = baseUrl + '&view=left';
                        document.getElementById('imgRight').src = baseUrl + '&view=right';
                        document.getElementById('imgDepth').src = baseUrl + '&view=depth';
                    } else {
                        document.getElementById('img' + viewMode.charAt(0).toUpperCase() + viewMode.slice(1)).src = baseUrl + '&view=' + viewMode;
                    }
                } else {
                    document.getElementById('imgLeft').src = baseUrl + '&view=left';
                    document.getElementById('imgRight').src = baseUrl + '&view=right';
                    document.getElementById('imgDepth').src = baseUrl + '&view=depth';
                    document.getElementById('frameImage').src = baseUrl;
                }
                
                document.getElementById('currentFrame').textContent = frameIdx;
                document.getElementById('frameSlider').value = frameIdx;
                currentFrame = frameIdx;
            } catch(e) {
                console.error('Failed to render frame:', e);
            }
        }
        
        function togglePlay() {
            if (isPlaying) {
                clearInterval(playInterval);
                playInterval = null;
                isPlaying = false;
                document.getElementById('playBtn').textContent = '▶️';
                // Re-render with overlays when pausing
                renderFrame(currentFrame, false);
            } else {
                isPlaying = true;
                document.getElementById('playBtn').textContent = '⏸️';
                playInterval = setInterval(() => {
                    currentFrame = (currentFrame + 1) % totalFrames;
                    renderFrame(currentFrame, true);
                }, 250);
            }
        }
        
        function prevFrame() {
            currentFrame = Math.max(0, currentFrame - 1);
            renderFrame(currentFrame);
        }
        
        function nextFrame() {
            currentFrame = Math.min(totalFrames - 1, currentFrame + 1);
            renderFrame(currentFrame);
        }
        
        function seekFrame(value) {
            currentFrame = parseInt(value);
            renderFrame(currentFrame);
        }
        
        async function uploadFile(input) {
            const file = input.files[0];
            if (!file) return;
            
            const formData = new FormData();
            formData.append('file', file);
            
            document.getElementById('uploadProgress').classList.remove('hidden');
            document.getElementById('uploadBar').style.width = '0%';
            document.getElementById('uploadStatus').textContent = 'Uploading ' + file.name + '...';
            
            try {
                const xhr = new XMLHttpRequest();
                xhr.upload.addEventListener('progress', (e) => {
                    if (e.lengthComputable) {
                        const pct = (e.loaded / e.total) * 100;
                        document.getElementById('uploadBar').style.width = pct + '%';
                    }
                });
                
                xhr.onload = function() {
                    if (xhr.status === 200) {
                        document.getElementById('uploadStatus').textContent = '✓ Upload complete!';
                        setTimeout(() => {
                            document.getElementById('uploadProgress').classList.add('hidden');
                            loadFiles();
                        }, 1500);
                    } else {
                        document.getElementById('uploadStatus').textContent = '✗ Upload failed';
                    }
                };
                
                xhr.onerror = function() {
                    document.getElementById('uploadStatus').textContent = '✗ Upload failed';
                };
                
                xhr.open('POST', '/upload');
                xhr.send(formData);
            } catch(e) {
                document.getElementById('uploadStatus').textContent = '✗ Error: ' + e.message;
            }
            
            input.value = '';
        }
        
        function handleFrameClick(event) {
            // No-op: click segmentation removed
        }
        
        // === Workflow Functions ===
        
        function wfLog(msg) {
            const log = document.getElementById('wfLog');
            const line = document.createElement('div');
            line.className = 'log-line';
            line.textContent = '[' + new Date().toLocaleTimeString() + '] ' + msg;
            log.appendChild(line);
            log.scrollTop = log.scrollHeight;
        }
        
        function setStepState(stepId, state, detail, progress) {
            const el = document.getElementById(stepId);
            el.className = 'workflow-step' + (state ? ' ' + state : '');
            if (detail !== undefined) {
                document.getElementById(stepId + 'Detail').textContent = detail;
            }
            if (progress !== undefined) {
                const bar = document.getElementById(stepId + 'Progress');
                if (bar) bar.style.width = progress + '%';
            }
        }
        
        function resetWorkflowUI() {
            ['stepExtract', 'stepColmap', 'stepTrain', 'stepView'].forEach(id => {
                setStepState(id, '', 'Waiting...', 0);
            });
            document.getElementById('viewSplatLink').style.display = 'none';
        }
        
        async function extractFramesOnly() {
            if (!currentFile) {
                document.getElementById('wfStatus').textContent = 'Load an SVO file first';
                return;
            }
            
            const datasetName = document.getElementById('wfDatasetName').value || currentFile.replace(/\.[^.]+$/, '');
            const fps = document.getElementById('wfFps').value;
            const includeDepth = document.getElementById('wfIncludeDepth').checked;
            const filterBlur = document.getElementById('wfFilterBlur').checked;
            
            resetWorkflowUI();
            setStepState('stepExtract', 'active', 'Extracting frames...', 10);
            document.getElementById('wfStartBtn').disabled = true;
            document.getElementById('wfExtractBtn').disabled = true;
            wfLog('Starting frame extraction: ' + datasetName);
            
            try {
                const response = await fetch('/workflow/extract-frames', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        dataset_name: datasetName,
                        fps: parseFloat(fps),
                        include_depth: includeDepth,
                        filter_blur: filterBlur
                    })
                });
                const data = await response.json();
                
                if (data.status === 'ok') {
                    setStepState('stepExtract', 'completed', 'Extracted ' + data.num_frames + ' frames', 100);
                    wfLog('Extraction complete: ' + data.num_frames + ' frames');
                    document.getElementById('wfStatus').textContent = 'Frames extracted. Ready for cuSFM.';
                } else {
                    setStepState('stepExtract', 'failed', data.error || 'Failed', 0);
                    wfLog('ERROR: ' + (data.error || 'Extraction failed'));
                }
            } catch(e) {
                setStepState('stepExtract', 'failed', e.message, 0);
                wfLog('ERROR: ' + e.message);
            }
            
            document.getElementById('wfStartBtn').disabled = false;
            document.getElementById('wfExtractBtn').disabled = false;
        }
        
        async function suggestSettings() {
            if (!currentFile) {
                document.getElementById('wfStatus').textContent = 'Load an SVO file first';
                return;
            }
            
            const btn = document.getElementById('wfSuggestBtn');
            btn.disabled = true;
            btn.textContent = '🔍 Analyzing...';
            const infoDiv = document.getElementById('wfSuggestionInfo');
            const textDiv = document.getElementById('wfAnalysisText');
            infoDiv.style.display = 'none';
            
            try {
                const resp = await fetch('/analyze_file');
                if (!resp.ok) throw new Error('Analysis failed');
                const data = await resp.json();
                const s = data.suggestions;
                const a = data.analysis;
                const r = data.reasons;
                
                // Apply suggested values to form
                document.getElementById('wfFps').value = s.fps;
                
                const stepsSelect = document.getElementById('wfSteps');
                for (let opt of stepsSelect.options) {
                    if (parseInt(opt.value) === s.training_steps) {
                        opt.selected = true;
                        break;
                    }
                }
                
                document.getElementById('wfIncludeDepth').checked = s.include_depth;
                document.getElementById('wfUseMcmc').checked = s.use_mcmc;
                document.getElementById('wfFilterBlur').checked = s.filter_blur;
                
                // Show analysis info
                textDiv.innerHTML = 
                    '<div style="margin-bottom:6px;"><b style="color:#2196F3;">File Analysis</b></div>' +
                    '<div>' + a.resolution_class + ' ' + a.single_cam_dimensions + 
                    ' | ' + a.stereo_layout + ' stereo' +
                    ' | ' + a.total_frames + ' frames @ ' + a.video_fps + ' FPS' +
                    ' | ' + a.duration_sec + 's</div>' +
                    '<div style="margin-top:4px;">Sharpness: avg=' + a.sharpness_avg + 
                    ', min=' + a.sharpness_min + ', max=' + a.sharpness_max + '</div>' +
                    '<div style="margin-top:8px;"><b style="color:#2196F3;">Recommendations</b></div>' +
                    '<div style="margin-top:2px;">&#8226; <b>FPS ' + s.fps + '</b>: ' + r.fps + '</div>' +
                    '<div>&#8226; <b>' + (s.training_steps/1000) + 'K steps</b>: ' + r.steps + '</div>' +
                    '<div>&#8226; <b>Depth ' + (s.include_depth ? 'ON' : 'OFF') + '</b>: ' + r.depth + '</div>' +
                    '<div>&#8226; <b>MCMC ' + (s.use_mcmc ? 'ON' : 'OFF') + '</b>: ' + r.mcmc + '</div>' +
                    '<div>&#8226; <b>Blur filter ' + (s.filter_blur ? 'ON' : 'OFF') + '</b>: ' + r.filter_blur + '</div>' +
                    '<div style="margin-top:6px; color:#4CAF50;">~' + s.estimated_frames + ' frames will be extracted</div>';
                
                infoDiv.style.display = 'block';
                wfLog('Analysis complete: ' + a.resolution_class + ' ' + a.stereo_layout + 
                      ' stereo, suggested FPS=' + s.fps + ', steps=' + s.training_steps);
            } catch(e) {
                wfLog('ERROR: Analysis failed - ' + e.message);
            }
            
            btn.disabled = false;
            btn.textContent = '🔍 Analyze & Suggest Settings';
        }
        
        async function startWorkflow() {
            if (!currentFile) {
                document.getElementById('wfStatus').textContent = 'Load an SVO file first';
                return;
            }
            
            const datasetName = document.getElementById('wfDatasetName').value || currentFile.replace(/\.[^.]+$/, '');
            const fps = document.getElementById('wfFps').value;
            const steps = document.getElementById('wfSteps').value;
            const includeDepth = document.getElementById('wfIncludeDepth').checked;
            const useMcmc = document.getElementById('wfUseMcmc').checked;
            const filterBlur = document.getElementById('wfFilterBlur').checked;
            
            resetWorkflowUI();
            document.getElementById('wfStartBtn').disabled = true;
            document.getElementById('wfExtractBtn').disabled = true;
            document.getElementById('wfStatus').textContent = 'Pipeline running...';
            wfLog('Starting full pipeline: ' + datasetName + (useMcmc ? ' (MCMC)' : ''));
            
            try {
                const response = await fetch('/workflow/start', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        dataset_name: datasetName,
                        fps: parseFloat(fps),
                        num_training_steps: parseInt(steps),
                        include_depth: includeDepth,
                        use_mcmc: useMcmc,
                        filter_blur: filterBlur
                    })
                });
                const data = await response.json();
                
                if (data.workflow_id) {
                    activeWorkflowId = data.workflow_id;
                    wfLog('Workflow started: ' + data.workflow_id);
                    setStepState('stepExtract', 'active', 'Starting...', 5);
                    
                    // Start polling for status
                    if (workflowPollInterval) clearInterval(workflowPollInterval);
                    workflowPollInterval = setInterval(pollWorkflowStatus, 3000);
                } else {
                    wfLog('ERROR: ' + (data.error || 'Failed to start'));
                    document.getElementById('wfStatus').textContent = 'Failed to start pipeline';
                    document.getElementById('wfStartBtn').disabled = false;
                    document.getElementById('wfExtractBtn').disabled = false;
                }
            } catch(e) {
                wfLog('ERROR: ' + e.message);
                document.getElementById('wfStatus').textContent = 'Error: ' + e.message;
                document.getElementById('wfStartBtn').disabled = false;
                document.getElementById('wfExtractBtn').disabled = false;
            }
        }
        
        async function pollWorkflowStatus() {
            if (!activeWorkflowId) return;
            
            try {
                const response = await fetch('/workflow/status/' + activeWorkflowId);
                const status = await response.json();
                
                // Update step states based on progress
                const progress = status.progress || 0;
                const step = status.current_step || '';
                
                // Step 1: Extract frames (0-0.2)
                if (progress < 0.2) {
                    setStepState('stepExtract', 'active', step, Math.round(progress / 0.2 * 100));
                } else {
                    setStepState('stepExtract', 'completed', status.num_frames ? status.num_frames + ' frames' : 'Done', 100);
                }
                
                // Step 2: cuVSLAM (0.2-0.7)
                if (progress >= 0.2 && progress < 0.7) {
                    setStepState('stepColmap', 'active', step, Math.round((progress - 0.2) / 0.5 * 100));
                } else if (progress >= 0.7) {
                    setStepState('stepColmap', 'completed', 'Done', 100);
                }
                
                // Step 3: Training (0.7-0.95)
                if (progress >= 0.7 && progress < 0.95) {
                    setStepState('stepTrain', 'active', step, Math.round((progress - 0.7) / 0.25 * 100));
                } else if (progress >= 0.95) {
                    setStepState('stepTrain', 'completed', 'Done', 100);
                }
                
                // Step 4: View
                if (status.status === 'completed') {
                    setStepState('stepView', 'completed', 'Ready to view!');
                    const link = document.getElementById('viewSplatLink');
                    link.href = VIEWER_URL;
                    link.style.display = 'block';
                    document.getElementById('wfStatus').textContent = 'Pipeline complete!';
                    wfLog('Pipeline complete! View splat at ' + VIEWER_URL);
                    clearInterval(workflowPollInterval);
                    workflowPollInterval = null;
                    document.getElementById('wfStartBtn').disabled = false;
                    document.getElementById('wfExtractBtn').disabled = false;
                } else if (status.status === 'failed') {
                    wfLog('ERROR: ' + (status.error || 'Pipeline failed'));
                    document.getElementById('wfStatus').textContent = 'Pipeline failed';
                    clearInterval(workflowPollInterval);
                    workflowPollInterval = null;
                    document.getElementById('wfStartBtn').disabled = false;
                    document.getElementById('wfExtractBtn').disabled = false;
                }
                
            } catch(e) {
                console.error('Failed to poll workflow:', e);
            }
        }
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the viewer UI"""
    return HTMLResponse(content=get_ui_html())


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "isaac-viewer",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat(),
        "cv2_available": CV2_AVAILABLE,
        "current_file": current_file
    }


@app.get("/files")
async def list_files():
    """List available SVO and ROSBAG files"""
    return get_available_files()


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload SVO or ROSBAG file"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename")
    
    ext = Path(file.filename).suffix.lower()
    if ext in ['.svo', '.svo2']:
        dest_dir = SVO_DIR
    elif ext in ['.bag', '.db3', '.mcap']:
        dest_dir = ROSBAG_DIR
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    
    dest_path = dest_dir / file.filename
    
    with open(dest_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    logger.info(f"Uploaded file: {file.filename} ({len(content)} bytes)")
    return {"status": "ok", "filename": file.filename, "size": len(content)}


@app.get("/load/{filename}")
async def load_file_endpoint(filename: str):
    """Load a file for viewing"""
    global current_file, current_frame_idx, total_frames, current_image
    
    # Find file
    file_path = None
    file_type = None
    
    for d, t in [(SVO_DIR, "svo"), (ROSBAG_DIR, "rosbag")]:
        p = d / filename
        if p.exists():
            file_path = str(p)
            file_type = t
            break
    
    if not file_path:
        raise HTTPException(status_code=404, detail="File not found")
    
    current_file = filename
    current_frame_idx = 0
    current_image = None
    
    # Get frame count
    total_frames = 100  # Default
    if CV2_AVAILABLE:
        try:
            cap = cv2.VideoCapture(file_path)
            if cap.isOpened():
                fc = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                # Validate frame count (SVO files may return invalid values)
                if fc > 0 and fc < 1000000:
                    total_frames = int(fc)
                else:
                    # Try to estimate by seeking
                    cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
                    total_frames = max(100, int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
                cap.release()
        except Exception as e:
            logger.warning(f"Could not get frame count: {e}")
            total_frames = 100
    
    return {
        "status": "ok",
        "filename": filename,
        "type": file_type,
        "total_frames": total_frames
    }


def read_raw_frame(file_path: str, frame_idx: int) -> Optional[np.ndarray]:
    """Read a raw frame from video, using cache to avoid re-reading for multiple views"""
    global video_capture, video_capture_file, raw_frame_cache, raw_frame_cache_file
    
    # Clear cache if file changed
    if raw_frame_cache_file != file_path:
        raw_frame_cache = {}
        raw_frame_cache_file = file_path
    
    # Return cached frame if available
    if frame_idx in raw_frame_cache:
        return raw_frame_cache[frame_idx]
    
    # Read from video
    try:
        if video_capture_file != file_path or video_capture is None or not video_capture.isOpened():
            if video_capture is not None:
                video_capture.release()
            video_capture = cv2.VideoCapture(file_path)
            video_capture_file = file_path
        
        if video_capture.isOpened():
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, raw_frame = video_capture.read()
            
            if ret and raw_frame is not None:
                rgb_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
                # Keep only last 5 frames in cache to limit memory
                if len(raw_frame_cache) > 5:
                    oldest = min(raw_frame_cache.keys())
                    del raw_frame_cache[oldest]
                raw_frame_cache[frame_idx] = rgb_frame
                return rgb_frame
            else:
                # H264 seek can fail - try sequential read from nearest cached frame
                logger.warning(f"Seek to frame {frame_idx} failed, trying sequential read")
                nearest = max([k for k in raw_frame_cache.keys() if k < frame_idx], default=-1)
                if nearest >= 0:
                    video_capture.set(cv2.CAP_PROP_POS_FRAMES, nearest)
                    video_capture.read()  # skip cached frame
                    for skip_idx in range(nearest + 1, frame_idx + 1):
                        ret, raw_frame = video_capture.read()
                        if ret and raw_frame is not None and skip_idx == frame_idx:
                            rgb_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
                            raw_frame_cache[frame_idx] = rgb_frame
                            return rgb_frame
    except Exception as e:
        logger.warning(f"Could not read frame {frame_idx}: {e}")
    
    return None


def extract_view_from_frame(full_frame: np.ndarray, view: str) -> np.ndarray:
    """Extract a specific view (left/right/depth) from a full stereo frame.
    Supports both side-by-side and top-bottom stereo layouts."""
    h, w = full_frame.shape[:2]
    is_side_by_side = w > h * 1.5
    is_top_bottom = not is_side_by_side and h > w  # Portrait = top-bottom stereo (ZED never produces portrait natively)
    
    if view == 'left':
        if is_side_by_side:
            return full_frame[:, :w//2, :]
        elif is_top_bottom:
            return full_frame[:h//2, :, :]
        return full_frame
    elif view == 'right':
        if is_side_by_side:
            return full_frame[:, w//2:, :]
        elif is_top_bottom:
            return full_frame[h//2:, :, :]
        return full_frame
    elif view == 'depth':
        if is_side_by_side:
            left_img = full_frame[:, :w//2, :]
        elif is_top_bottom:
            left_img = full_frame[:h//2, :, :]
        else:
            left_img = full_frame
        return estimate_depth_from_service(left_img)
    
    return full_frame


def estimate_depth_from_service(image_rgb: np.ndarray) -> np.ndarray:
    """Call Depth Anything v2 service for monocular depth estimation.
    Falls back to grayscale colormap if service is unavailable."""
    try:
        # Encode image as PNG for upload
        img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        _, png_data = cv2.imencode(".png", img_bgr)
        
        response = requests.post(
            f"{DEPTH_SERVICE_URL}/estimate",
            files={"file": ("frame.png", png_data.tobytes(), "image/png")},
            params={"colormap": "inferno"},
            timeout=30,
        )
        
        if response.status_code == 200:
            # Decode the returned depth colormap PNG
            depth_arr = np.frombuffer(response.content, np.uint8)
            depth_bgr = cv2.imdecode(depth_arr, cv2.IMREAD_COLOR)
            if depth_bgr is not None:
                return cv2.cvtColor(depth_bgr, cv2.COLOR_BGR2RGB)
        
        logger.warning(f"Depth service returned {response.status_code}")
    except requests.exceptions.ConnectionError:
        logger.warning(f"Depth service unavailable at {DEPTH_SERVICE_URL}")
    except Exception as e:
        logger.error(f"Depth estimation failed: {e}")
    
    # Fallback: grayscale colormap
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    depth_colored = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)


def compute_stereo_depth(left_rgb: np.ndarray, right_rgb: np.ndarray) -> np.ndarray:
    """Compute depth map from stereo pair using downscaled SGBM + bilateral filtering"""
    orig_h, orig_w = left_rgb.shape[:2]
    
    # Downscale for smoother disparity and faster computation
    scale = 0.5 if orig_w > 800 else 1.0
    if scale < 1.0:
        sw, sh = int(orig_w * scale), int(orig_h * scale)
        left_s = cv2.resize(left_rgb, (sw, sh), interpolation=cv2.INTER_AREA)
        right_s = cv2.resize(right_rgb, (sw, sh), interpolation=cv2.INTER_AREA)
    else:
        left_s, right_s = left_rgb, right_rgb
        sw, sh = orig_w, orig_h
    
    gray_l = cv2.cvtColor(left_s, cv2.COLOR_RGB2GRAY)
    gray_r = cv2.cvtColor(right_s, cv2.COLOR_RGB2GRAY)
    
    # Histogram equalization to normalize brightness between cameras
    gray_l = cv2.equalizeHist(gray_l)
    gray_r = cv2.equalizeHist(gray_r)
    
    # StereoSGBM with tuned parameters for ZED-like stereo
    num_disp = 96  # Multiple of 16, covers typical indoor range
    block_size = 7
    stereo_left = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * 3 * block_size ** 2,
        P2=32 * 3 * block_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=5,
        speckleWindowSize=200,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    
    # Compute left disparity
    disp_left = stereo_left.compute(gray_l, gray_r).astype(np.float32) / 16.0
    
    # Also compute right disparity for left-right consistency check
    stereo_right = cv2.StereoSGBM_create(
        minDisparity=-num_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * 3 * block_size ** 2,
        P2=32 * 3 * block_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=5,
        speckleWindowSize=200,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    disp_right = stereo_right.compute(gray_r, gray_l).astype(np.float32) / 16.0
    
    # Left-right consistency check (vectorized): reject where left/right disagree
    valid = disp_left > 0
    x_coords = np.arange(sw)
    for y in range(sh):
        row_valid = valid[y]
        if not row_valid.any():
            continue
        d_vals = disp_left[y].astype(int)
        rx = x_coords - d_vals
        in_bounds = (rx >= 0) & (rx < sw) & row_valid
        check_x = np.clip(rx, 0, sw - 1)
        lr_diff = np.abs(disp_left[y] + disp_right[y, check_x])
        valid[y] = in_bounds & (lr_diff <= 1.5)
    
    # Normalize to 0-255
    disp_norm = np.zeros((sh, sw), dtype=np.uint8)
    if valid.any():
        d_min = np.percentile(disp_left[valid], 5)
        d_max = np.percentile(disp_left[valid], 95)
        if d_max > d_min:
            clipped = np.clip(disp_left, d_min, d_max)
            disp_norm = ((clipped - d_min) / (d_max - d_min) * 255).astype(np.uint8)
            disp_norm[~valid] = 0
    
    # Post-processing: bilateral filter preserves edges while smoothing
    disp_norm = cv2.bilateralFilter(disp_norm, 9, 75, 75)
    disp_norm = cv2.medianBlur(disp_norm, 5)
    
    # Fill small holes with morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    disp_norm = cv2.morphologyEx(disp_norm, cv2.MORPH_CLOSE, kernel)
    
    # Upscale back to original resolution if downscaled
    if scale < 1.0:
        disp_norm = cv2.resize(disp_norm, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        # Extra smooth at full resolution
        disp_norm = cv2.bilateralFilter(disp_norm, 5, 50, 50)
    
    # Apply colormap
    depth_colored = cv2.applyColorMap(disp_norm, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)


@app.get("/frame/{frame_idx}")
async def get_frame(frame_idx: int, view: str = Query("left")):
    """Get a specific frame as PNG image"""
    global current_frame_idx, current_image
    
    current_frame_idx = frame_idx
    
    # Get the frame from file
    frame = None
    if current_file:
        file_path = None
        for d in [SVO_DIR, ROSBAG_DIR]:
            p = d / current_file
            if p.exists():
                file_path = str(p)
                break
        
        if file_path and CV2_AVAILABLE:
            if file_path.endswith(('.svo', '.svo2', '.mp4', '.avi', '.mov')):
                raw = read_raw_frame(file_path, frame_idx)
                if raw is not None:
                    frame = extract_view_from_frame(raw, view)
    
    # Fall back to simulated frame if needed
    if frame is None:
        frame = generate_simulated_frame(frame_idx, view=view)
    
    current_image = frame
    
    png_data = frame_to_png(frame)
    return Response(content=png_data, media_type="image/png")


@app.post("/workflow/extract-frames")
async def workflow_extract_frames(request: dict = None):
    """Extract left stereo frames and depth maps from loaded SVO file.
    Saves images to FRAME_DIR/<dataset_name>/images/ and depth/ directories."""
    if not current_file:
        return JSONResponse({"error": "No file loaded"}, status_code=400)
    
    if request is None:
        request = {}
    
    dataset_name = request.get("dataset_name", current_file.rsplit(".", 1)[0])
    fps = request.get("fps", 2.0)
    include_depth = request.get("include_depth", True)
    filter_blur = request.get("filter_blur", False)
    
    # Find file path
    file_path = None
    for d in [SVO_DIR, ROSBAG_DIR]:
        p = d / current_file
        if p.exists():
            file_path = str(p)
            break
    
    if not file_path or not CV2_AVAILABLE:
        return JSONResponse({"error": "File not found or OpenCV unavailable"}, status_code=400)
    
    # Create output directories
    output_dir = FRAME_DIR / dataset_name
    images_dir = output_dir / "images"
    images_right_dir = output_dir / "images_right"
    images_dir.mkdir(parents=True, exist_ok=True)
    images_right_dir.mkdir(parents=True, exist_ok=True)
    
    if include_depth:
        depth_dir = output_dir / "depth"
        depth_dir.mkdir(parents=True, exist_ok=True)
    
    # Open video and extract frames at target FPS
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        return JSONResponse({"error": "Cannot open video file"}, status_code=500)
    
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, int(video_fps / fps))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    num_extracted = 0
    frame_idx = 0
    has_stereo = False
    
    while True:
        ret, raw_frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_interval == 0:
            rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            is_stereo = w > h * 1.5
            
            # Extract left image
            left = rgb[:, :w//2, :] if is_stereo else rgb
            left_bgr = cv2.cvtColor(left, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(images_dir / f"frame_{num_extracted:04d}.jpg"), left_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            # Extract right image for cuVSLAM stereo processing
            if is_stereo:
                has_stereo = True
                right = rgb[:, w//2:, :]
                right_bgr = cv2.cvtColor(right, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(images_right_dir / f"frame_{num_extracted:04d}.jpg"), right_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            # Extract depth if requested
            if include_depth:
                depth_img = estimate_depth_from_service(left)
                depth_bgr = cv2.cvtColor(depth_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(depth_dir / f"depth_{num_extracted:04d}.png"), depth_bgr)
            
            num_extracted += 1
        
        frame_idx += 1
    
    cap.release()
    
    logger.info(f"Extracted {num_extracted} frames (stereo={has_stereo}) from {current_file} to {output_dir}")
    
    # Filter blurry frames if enabled
    if filter_blur and num_extracted > 3:
        right_dir = images_right_dir if has_stereo else None
        num_extracted = filter_sharp_frames(images_dir, right_dir)
        logger.info(f"After blur filter: {num_extracted} frames kept")
    
    return {
        "status": "ok",
        "dataset_name": dataset_name,
        "num_frames": num_extracted,
        "output_dir": str(output_dir),
        "include_depth": include_depth,
        "has_stereo": has_stereo
    }


@app.get("/analyze_file")
async def analyze_file():
    """Analyze the currently loaded SVO/video file and suggest optimal pipeline parameters
    for producing a crisp Gaussian Splat."""
    if not current_file:
        return JSONResponse({"error": "No file loaded"}, status_code=400)

    file_path = None
    for d in [SVO_DIR, ROSBAG_DIR]:
        p = d / current_file
        if p.exists():
            file_path = str(p)
            break

    if not file_path:
        return JSONResponse({"error": "File not found on disk"}, status_code=404)

    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return JSONResponse({"error": "Cannot open file"}, status_code=500)

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        # Count readable frames
        all_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            all_frames.append(frame)
        cap.release()

        total_frames = len(all_frames)
        if total_frames == 0:
            return JSONResponse({"error": "No readable frames"}, status_code=400)

        h, w = all_frames[0].shape[:2]
        duration = total_frames / max(video_fps, 1)

        # Detect stereo layout
        is_sbs = w > h * 1.5
        is_tb = not is_sbs and h > w
        is_stereo = is_sbs or is_tb
        single_w = w // 2 if is_sbs else w
        single_h = h // 2 if is_tb else h

        # Resolution class
        if single_w >= 2000:
            res_class = "HD2K"
        elif single_w >= 1200:
            res_class = "HD1080"
        elif single_w >= 800:
            res_class = "HD720"
        else:
            res_class = "VGA"

        # Sample sharpness (Laplacian variance) across ~20 frames
        blur_scores = []
        sample_indices = list(range(0, total_frames, max(1, total_frames // 20)))
        for i in sample_indices:
            f = all_frames[i]
            if is_tb:
                f = f[:h // 2, :, :]
            elif is_sbs:
                f = f[:, :w // 2, :]
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            blur_scores.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))

        del all_frames  # Free memory

        avg_sharpness = float(np.mean(blur_scores))
        min_sharpness = float(np.min(blur_scores))
        max_sharpness = float(np.max(blur_scores))
        sharpness_range = max_sharpness - min_sharpness

        # --- Determine optimal parameters ---

        # FPS: target 100-150 output frames for crisp Gaussian Splats
        # Research shows minimum ~100 frames needed for good reconstruction
        target_frames = 120
        suggested_fps = round(target_frames / max(duration, 1), 1)
        suggested_fps = max(0.5, min(suggested_fps, video_fps))
        estimated_frames = int(duration * suggested_fps)

        # Training steps: based on resolution and frame count
        if res_class == "HD2K" and estimated_frames >= 100:
            suggested_steps = 62200
            steps_label = "62K (Best)"
        elif estimated_frames > 80 or res_class in ("HD2K", "HD1080"):
            suggested_steps = 30000
            steps_label = "30K (Good)"
        elif estimated_frames < 30:
            suggested_steps = 7000
            steps_label = "7K (Quick)"
        else:
            suggested_steps = 30000
            steps_label = "30K (Good)"

        # Include depth: always yes for stereo (ZED provides real depth)
        suggested_depth = is_stereo

        # MCMC: default OFF - 1M Gaussian cap can reduce crispness for HD scenes
        suggested_mcmc = False

        # Filter blurry frames: recommend if sharpness variance is high
        blur_ratio = sharpness_range / max(avg_sharpness, 1)
        suggested_filter_blur = blur_ratio > 0.5 or min_sharpness < 50

        # Build reasoning strings
        reasons = {}
        reasons["fps"] = (
            f"At {video_fps:.0f} FPS source over {duration:.1f}s, "
            f"extracting at {suggested_fps} FPS yields ~{estimated_frames} frames "
            f"(min ~100 frames recommended for crisp reconstruction)"
        )
        reasons["steps"] = (
            f"{steps_label} recommended for {res_class} resolution with ~{estimated_frames} frames"
        )
        reasons["depth"] = (
            "Stereo depth from ZED improves Gaussian placement accuracy"
            if suggested_depth else
            "No stereo depth available; monocular depth adds moderate benefit"
        )
        reasons["mcmc"] = (
            "Standard training recommended - allows Gaussians to grow freely for maximum crispness. "
            "MCMC's 1M cap can cause blurriness on HD scenes."
        )
        reasons["filter_blur"] = (
            f"Sharpness varies significantly (min={min_sharpness:.0f}, max={max_sharpness:.0f}); "
            f"filtering removes blurry frames for crisper results"
            if suggested_filter_blur else
            f"Frames are consistently sharp (avg={avg_sharpness:.0f}); filtering not critical"
        )

        return {
            "file": current_file,
            "analysis": {
                "video_fps": video_fps,
                "total_frames": total_frames,
                "duration_sec": round(duration, 1),
                "raw_dimensions": f"{w}x{h}",
                "single_cam_dimensions": f"{single_w}x{single_h}",
                "resolution_class": res_class,
                "is_stereo": is_stereo,
                "stereo_layout": "side-by-side" if is_sbs else "top-bottom" if is_tb else "mono",
                "sharpness_avg": round(avg_sharpness, 1),
                "sharpness_min": round(min_sharpness, 1),
                "sharpness_max": round(max_sharpness, 1),
            },
            "suggestions": {
                "fps": suggested_fps,
                "training_steps": suggested_steps,
                "include_depth": suggested_depth,
                "use_mcmc": suggested_mcmc,
                "filter_blur": suggested_filter_blur,
                "estimated_frames": estimated_frames,
            },
            "reasons": reasons,
        }
    except Exception as e:
        logger.error(f"File analysis failed: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/workflow/start")
async def workflow_start(request: dict, background_tasks: BackgroundTasks = None):
    """Start the full SVO to Gaussian Splat pipeline.
    Steps: Extract frames -> cuVSLAM -> fVDB Training -> View in :8085"""
    if not current_file:
        return JSONResponse({"error": "No file loaded"}, status_code=400)
    
    dataset_name = request.get("dataset_name", current_file.rsplit(".", 1)[0])
    fps = request.get("fps", 2.0)
    num_training_steps = request.get("num_training_steps", 30000)
    include_depth = request.get("include_depth", True)
    use_mcmc = request.get("use_mcmc", False)
    filter_blur = request.get("filter_blur", False)
    
    workflow_id = f"wf_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    
    active_workflows[workflow_id] = {
        "workflow_id": workflow_id,
        "status": "running",
        "progress": 0.0,
        "current_step": "Starting frame extraction",
        "dataset_name": dataset_name,
        "num_frames": 0,
        "use_mcmc": use_mcmc,
        "error": None,
        "started_at": datetime.now().isoformat()
    }
    
    if background_tasks:
        background_tasks.add_task(
            run_full_pipeline,
            workflow_id=workflow_id,
            dataset_name=dataset_name,
            fps=fps,
            num_training_steps=num_training_steps,
            include_depth=include_depth,
            use_mcmc=use_mcmc,
            filter_blur=filter_blur
        )
    
    return {
        "workflow_id": workflow_id,
        "status": "started",
        "message": f"Pipeline started. Monitor at /workflow/status/{workflow_id}"
    }


@app.get("/workflow/status/{workflow_id}")
async def workflow_status(workflow_id: str):
    """Get workflow status by ID"""
    if workflow_id not in active_workflows:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return active_workflows[workflow_id]


@app.get("/workflow/list")
async def workflow_list():
    """List all workflows"""
    return {"workflows": list(active_workflows.values())}


async def run_full_pipeline(workflow_id: str, dataset_name: str, fps: float,
                            num_training_steps: int, include_depth: bool,
                            use_mcmc: bool = False, filter_blur: bool = False):
    """Background task: full SVO -> cuVSLAM -> Train -> View pipeline"""
    wf = active_workflows[workflow_id]
    
    try:
        # === Step 1: Extract frames ===
        wf["current_step"] = "Extracting frames from SVO"
        wf["progress"] = 0.05
        
        # Find file
        file_path = None
        for d in [SVO_DIR, ROSBAG_DIR]:
            p = d / current_file
            if p.exists():
                file_path = str(p)
                break
        
        if not file_path:
            raise Exception("Source file not found")
        
        output_dir = FRAME_DIR / dataset_name
        images_dir = output_dir / "images"
        images_right_dir = output_dir / "images_right"
        images_dir.mkdir(parents=True, exist_ok=True)
        images_right_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise Exception("Cannot open video file")
        
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        reported_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        logger.info(f"[{workflow_id}] Video: fps={video_fps}, reported_count={reported_count}")
        
        # Read all frames first (CAP_PROP_FRAME_COUNT is unreliable for SVO files)
        all_frames = []
        while True:
            ret, raw_frame = cap.read()
            if not ret:
                break
            all_frames.append(raw_frame)
        cap.release()
        
        total_readable = len(all_frames)
        logger.info(f"[{workflow_id}] Total readable frames: {total_readable}")
        
        if total_readable == 0:
            raise Exception("No frames could be read from file")
        
        # Calculate frame interval based on desired output fps
        frame_interval = max(1, int(video_fps / fps))
        expected_output = total_readable // frame_interval
        
        # Ensure we extract at least 10 frames (or all if fewer available)
        min_frames = min(10, total_readable)
        if expected_output < min_frames:
            frame_interval = max(1, total_readable // min_frames)
        
        logger.info(f"[{workflow_id}] Sampling: interval={frame_interval}, expected={total_readable // frame_interval} frames")
        
        num_extracted = 0
        has_stereo = False
        for frame_idx, raw_frame in enumerate(all_frames):
            if frame_idx % frame_interval == 0:
                rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]
                is_side_by_side = w > h * 1.5
                is_top_bottom = not is_side_by_side and h > w  # Portrait = top-bottom stereo
                is_stereo = is_side_by_side or is_top_bottom
                # Extract left frame
                if is_side_by_side:
                    left = rgb[:, :w//2, :]
                elif is_top_bottom:
                    left = rgb[:h//2, :, :]
                else:
                    left = rgb
                left_bgr = cv2.cvtColor(left, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(images_dir / f"frame_{num_extracted:04d}.jpg"), left_bgr,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                # Extract right stereo image for cuVSLAM
                if is_stereo:
                    has_stereo = True
                    if is_side_by_side:
                        right = rgb[:, w//2:, :]
                    else:
                        right = rgb[h//2:, :, :]
                    right_bgr = cv2.cvtColor(right, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(images_right_dir / f"frame_{num_extracted:04d}.jpg"), right_bgr,
                                [cv2.IMWRITE_JPEG_QUALITY, 95])
                num_extracted += 1
                wf["progress"] = min(0.18, 0.05 + (frame_idx / total_readable) * 0.13)
        
        del all_frames  # Free memory
        wf["num_frames"] = num_extracted
        wf["progress"] = 0.18
        wf["current_step"] = f"Extracted {num_extracted} frames"
        logger.info(f"[{workflow_id}] Extracted {num_extracted} frames")
        
        # Filter blurry frames using sharp-frames if enabled
        if filter_blur and num_extracted > 3:
            wf["current_step"] = "Filtering blurry frames..."
            wf["progress"] = 0.19
            right_dir = images_right_dir if has_stereo else None
            num_extracted = filter_sharp_frames(images_dir, right_dir, workflow_id)
            wf["num_frames"] = num_extracted
            wf["progress"] = 0.2
            wf["current_step"] = f"Sharp filter: kept {num_extracted} frames"
            logger.info(f"[{workflow_id}] After blur filter: {num_extracted} frames")
        
        if num_extracted < 3:
            raise Exception(f"Only {num_extracted} frames extracted, need at least 3")
        
        # === Step 2: Send to cuVSLAM service (Visual SLAM) ===
        wf["current_step"] = "Sending frames to cuVSLAM"
        wf["progress"] = 0.22
        
        # Create a ZIP of the images (left + right stereo) for upload
        zip_path = output_dir / "images.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for img_file in sorted(images_dir.glob("*.jpg")):
                zf.write(img_file, f"images/{img_file.name}")
            # Include right stereo frames if available
            if has_stereo and images_right_dir.exists():
                for img_file in sorted(images_right_dir.glob("*.jpg")):
                    zf.write(img_file, f"images_right/{img_file.name}")
        
        # Upload ZIP + camera params to cuVSLAM service
        # Include camera intrinsics JSON for accurate pose estimation
        camera_params = json.dumps({
            "fx": 527.6, "fy": 527.6,
            "cx": 636.4, "cy": 361.5,
            "baseline": 0.12
        })
        with open(zip_path, 'rb') as f:
            files = {
                'files': ('images.zip', f, 'application/zip'),
                'camera_params.json': ('camera_params.json', camera_params, 'application/json'),
            }
            data = {
                'dataset_id': dataset_name,
                'camera_model': 'PINHOLE',
                'matcher': 'sequential',
                'num_training_steps': str(num_training_steps),
                'use_mcmc': 'true' if use_mcmc else 'false',
            }
            
            colmap_resp = requests.post(
                f"{COLMAP_SERVICE_URL}/workflow/photos-to-model",
                files=files,
                data=data,
                timeout=60
            )
        
        zip_path.unlink(missing_ok=True)
        
        if colmap_resp.status_code != 200:
            raise Exception(f"cuVSLAM service returned {colmap_resp.status_code}: {colmap_resp.text[:200]}")
        
        colmap_data = colmap_resp.json()
        colmap_workflow_id = colmap_data.get("workflow_id", "")
        wf["colmap_workflow_id"] = colmap_workflow_id
        wf["progress"] = 0.25
        wf["current_step"] = "cuVSLAM processing started"
        logger.info(f"[{workflow_id}] cuVSLAM workflow: {colmap_workflow_id}")
        
        # Poll cuVSLAM status until complete
        max_wait = 3600  # 1 hour max
        elapsed = 0
        while elapsed < max_wait:
            await asyncio.sleep(10)
            elapsed += 10
            
            try:
                status_resp = requests.get(
                    f"{COLMAP_SERVICE_URL}/workflow/status/{colmap_workflow_id}",
                    timeout=10
                )
                if status_resp.status_code == 200:
                    colmap_status = status_resp.json()
                    colmap_progress = colmap_status.get("progress", 0)
                    colmap_step = colmap_status.get("current_step", "")
                    
                    # Map cuVSLAM progress (0-1) to our range (0.25-0.7)
                    wf["progress"] = 0.25 + colmap_progress * 0.45
                    wf["current_step"] = f"cuVSLAM: {colmap_step}"
                    
                    if colmap_status.get("status") in ("completed", "training", "completed_colmap_only"):
                        wf["progress"] = 0.7
                        wf["current_step"] = "cuVSLAM complete - sparse output ready"
                        logger.info(f"[{workflow_id}] cuVSLAM complete")
                        break
                    elif colmap_status.get("status") == "failed":
                        raise Exception(f"cuVSLAM failed: {colmap_status.get('error', 'Unknown')}")
            except requests.exceptions.ConnectionError:
                logger.warning(f"[{workflow_id}] cuVSLAM status check failed, retrying...")
        
        # === Step 3: Training (may have been started by cuVSLAM workflow) ===
        # Check if cuVSLAM workflow already triggered training
        training_job_id = None
        try:
            status_resp = requests.get(
                f"{COLMAP_SERVICE_URL}/workflow/status/{colmap_workflow_id}",
                timeout=10
            )
            if status_resp.status_code == 200:
                training_job_id = status_resp.json().get("training_job_id")
        except Exception:
            pass
        if not training_job_id:
            # Trigger training manually
            wf["current_step"] = "Starting Gaussian Splat training"
            wf["progress"] = 0.72
            
            train_resp = requests.post(
                f"{TRAINING_SERVICE_URL}/train",
                json={
                    "dataset_id": dataset_name,
                    "num_training_steps": num_training_steps,
                    "output_name": f"{dataset_name}_model",
                    "use_mcmc": use_mcmc
                },
                timeout=30
            )
            
            if train_resp.status_code == 200:
                training_job_id = train_resp.json().get("job_id")
            else:
                raise Exception(f"Training service returned {train_resp.status_code}")
        
        wf["training_job_id"] = training_job_id
        wf["current_step"] = "Training Gaussian Splat"
        logger.info(f"[{workflow_id}] Training job: {training_job_id}")
        
        # Poll training status
        elapsed = 0
        while elapsed < max_wait:
            await asyncio.sleep(15)
            elapsed += 15
            
            try:
                train_status = requests.get(
                    f"{TRAINING_SERVICE_URL}/jobs/{training_job_id}",
                    timeout=30
                )
                if train_status.status_code == 200:
                    tdata = train_status.json()
                    train_progress = tdata.get("progress", 0)
                    
                    # Map training progress to our range (0.70-0.95)
                    tstatus = tdata.get("status", "")
                    
                    wf["progress"] = 0.70 + train_progress * 0.25
                    wf["current_step"] = f"Training: {tdata.get('message', '')}"
                    
                    if tstatus == "completed":
                        wf["progress"] = 0.95
                        wf["current_step"] = "Training complete"
                        logger.info(f"[{workflow_id}] Training complete")
                        break
                    elif tstatus == "failed":
                        raise Exception(f"Training failed: {tdata.get('message', 'Unknown')}")
            except requests.exceptions.ConnectionError:
                logger.warning(f"[{workflow_id}] Training status check failed, retrying...")
        
        # === Step 5: Complete ===
        wf["status"] = "completed"
        wf["progress"] = 1.0
        wf["current_step"] = "Pipeline complete! View splat at :8085"
        wf["completed_at"] = datetime.now().isoformat()
        logger.info(f"[{workflow_id}] Pipeline complete!")
        
    except Exception as e:
        logger.error(f"[{workflow_id}] Pipeline failed: {e}")
        wf["status"] = "failed"
        wf["error"] = str(e)
        wf["current_step"] = f"Failed: {str(e)}"


@app.delete("/file/{filename}")
async def delete_file(filename: str):
    """Delete a file"""
    for d in [SVO_DIR, ROSBAG_DIR]:
        p = d / filename
        if p.exists():
            p.unlink()
            logger.info(f"Deleted file: {filename}")
            return {"status": "ok", "filename": filename}
    
    raise HTTPException(status_code=404, detail="File not found")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8012))
    uvicorn.run(app, host="0.0.0.0", port=port)
