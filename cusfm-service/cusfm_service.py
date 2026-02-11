"""
cuSFM Processing Service
GPU-accelerated Structure from Motion using NVIDIA cuSFM with TensorRT
Handles SVO/BAG stereo workflow for the ISAAC Viewer at :8012
"""

import os
import io
import json
import shutil
import subprocess
import logging
import asyncio
import zipfile
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="cuSFM Processing Service",
    description="GPU-accelerated Structure from Motion using NVIDIA cuSFM with TensorRT",
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
PYCUSFM_DIR = Path(os.getenv("PYCUSFM_DIR", "/app/pycusfm/pycusfm"))

for dir_path in [UPLOAD_DIR, PROCESSING_DIR, OUTPUT_DIR, TEMP_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# Active workflows (in-memory state)
active_workflows: Dict[str, Dict] = {}


# === Helper Functions ===

def generate_frames_meta(
    images_dir: Path,
    image_width: int = 1920,
    image_height: int = 1080,
    fx: float = 1000.0,
    fy: float = 1000.0,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
    fps: float = 30.0,
    camera_name: str = "zed_left"
) -> Dict:
    """Generate frames_meta.json for cuSFM from extracted images.
    Creates sequential EGO_MOTION poses (identity poses for un-posed images)."""
    if cx is None:
        cx = image_width / 2.0
    if cy is None:
        cy = image_height / 2.0

    # Find all images in the directory
    image_files = sorted(
        [f for f in images_dir.iterdir()
         if f.suffix.lower() in ('.jpg', '.jpeg', '.png')]
    )

    if not image_files:
        raise ValueError(f"No images found in {images_dir}")

    # Build keyframes metadata with identity poses
    keyframes = []
    base_timestamp = 1700000000000000  # base microseconds
    for idx, img_file in enumerate(image_files):
        timestamp = base_timestamp + int(idx * (1_000_000 / fps))
        keyframes.append({
            "id": str(idx),
            "camera_params_id": "0",
            "timestamp_microseconds": str(timestamp),
            "image_name": f"{camera_name}/{img_file.name}",
            "camera_to_world": {
                "axis_angle": {
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.0,
                    "angle_degrees": 0.0
                },
                "translation": {
                    "x": idx * 0.1,  # Simple sequential translation estimate
                    "y": 0.0,
                    "z": 0.0
                }
            },
            "synced_sample_id": str(idx)
        })

    # Build camera parameters
    camera_params = {
        "0": {
            "sensor_meta_data": {
                "sensor_id": 0,
                "sensor_type": "CAMERA",
                "sensor_name": camera_name,
                "frequency": fps,
                "sensor_to_vehicle_transform": {
                    "axis_angle": {"x": 0, "y": 0, "z": 0, "angle_degrees": 0},
                    "translation": {"x": 0, "y": 0, "z": 0}
                }
            },
            "calibration_parameters": {
                "image_width": image_width,
                "image_height": image_height,
                "camera_matrix": {
                    "data": [fx, 0, cx, 0, fy, cy, 0, 0, 1]
                },
                "distortion_coefficients": {
                    "data": [0, 0, 0, 0, 0]
                }
            }
        }
    }

    frames_meta = {
        "keyframes_metadata": keyframes,
        "initial_pose_type": "EGO_MOTION",
        "camera_params_id_to_session_name": {"0": "0"},
        "camera_params_id_to_camera_params": camera_params
    }

    return frames_meta


def prepare_cusfm_input(
    dataset_dir: Path,
    images_dir: Path,
    image_width: int = 1920,
    image_height: int = 1080,
    fx: float = 1000.0,
    fy: float = 1000.0,
    fps: float = 30.0,
    camera_name: str = "zed_left"
) -> Path:
    """Prepare input directory structure for cuSFM.
    Returns the input directory path."""
    input_dir = dataset_dir / "mapping_data"
    camera_dir = input_dir / camera_name
    camera_dir.mkdir(parents=True, exist_ok=True)

    # Copy images to camera directory
    image_files = sorted(
        [f for f in images_dir.iterdir()
         if f.suffix.lower() in ('.jpg', '.jpeg', '.png')]
    )

    for img_file in image_files:
        shutil.copy2(img_file, camera_dir / img_file.name)

    # Generate and write frames_meta.json
    frames_meta = generate_frames_meta(
        images_dir=images_dir,
        image_width=image_width,
        image_height=image_height,
        fx=fx,
        fy=fy,
        fps=fps,
        camera_name=camera_name
    )

    meta_path = input_dir / "frames_meta.json"
    with open(meta_path, 'w') as f:
        json.dump(frames_meta, f, indent=2)

    logger.info(f"Prepared cuSFM input: {len(image_files)} images in {input_dir}")
    return input_dir


async def run_cusfm_pipeline(
    workflow_id: str,
    input_dir: Path,
    output_dir: Path,
    feature_type: str = "aliked",
    export_binary: bool = True
):
    """Run the cuSFM pipeline as a background task."""
    wf = active_workflows[workflow_id]

    try:
        # Build cusfm_cli command
        cmd = [
            "cusfm_cli",
            "--input_dir", str(input_dir),
            "--cusfm_base_dir", str(output_dir),
            f"--feature_type={feature_type}",
        ]

        if export_binary:
            cmd.append("--export_binary_colmap_files")

        wf["current_step"] = "Running cuSFM feature extraction (ALIKED + TensorRT)"
        wf["progress"] = 0.1
        logger.info(f"[{workflow_id}] Running: {' '.join(cmd)}")

        # Run cusfm_cli as subprocess
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, "PYCUSFM_DIR": str(PYCUSFM_DIR)}
        )

        # Read output lines and update progress
        step_progress = {
            "feature_extractor": 0.2,
            "vocab_generator": 0.35,
            "pose_graph": 0.45,
            "matcher": 0.6,
            "mapper": 0.75,
            "map_convertor": 0.85,
        }

        while True:
            line = await process.stdout.readline()
            if not line:
                break
            decoded = line.decode().strip()
            if decoded:
                logger.info(f"[{workflow_id}] cusfm: {decoded}")
                wf["log"].append(decoded)

                # Update progress based on cuSFM pipeline steps
                for step_name, progress in step_progress.items():
                    if step_name.lower() in decoded.lower():
                        wf["progress"] = progress
                        wf["current_step"] = f"cuSFM: {step_name}"

        # Wait for completion
        await process.wait()

        if process.returncode != 0:
            stderr = await process.stderr.read()
            error_msg = stderr.decode().strip()
            raise Exception(f"cuSFM failed (exit code {process.returncode}): {error_msg[:500]}")

        # Verify output
        sparse_dir = output_dir / "sparse"
        if not sparse_dir.exists():
            raise Exception("cuSFM completed but sparse output not found")

        wf["status"] = "completed"
        wf["progress"] = 1.0
        wf["current_step"] = "cuSFM complete - COLMAP-compatible output ready"
        wf["output_dir"] = str(output_dir)
        wf["sparse_dir"] = str(sparse_dir)
        wf["completed_at"] = datetime.now().isoformat()
        logger.info(f"[{workflow_id}] cuSFM pipeline complete!")

    except Exception as e:
        logger.error(f"[{workflow_id}] cuSFM pipeline failed: {e}")
        wf["status"] = "failed"
        wf["error"] = str(e)
        wf["current_step"] = f"Failed: {str(e)}"


# === API Endpoints ===

@app.get("/health")
async def health():
    """Health check endpoint"""
    # Check if cusfm_cli is available
    cusfm_available = False
    try:
        result = subprocess.run(
            ["cusfm_cli", "--help"],
            capture_output=True, timeout=5,
            env={**os.environ, "PYCUSFM_DIR": str(PYCUSFM_DIR)}
        )
        cusfm_available = result.returncode == 0
    except Exception:
        pass

    return {
        "status": "ok",
        "service": "cusfm",
        "cusfm_available": cusfm_available,
        "gpu": True,
        "tensorrt": True
    }


@app.post("/process")
async def process_images(
    background_tasks: BackgroundTasks,
    dataset_id: str = Form("dataset"),
    image_width: int = Form(1920),
    image_height: int = Form(1080),
    fx: float = Form(1000.0),
    fy: float = Form(1000.0),
    fps: float = Form(30.0),
    feature_type: str = Form("aliked"),
    files: List[UploadFile] = File(...)
):
    """Upload images and run cuSFM processing.
    Accepts individual images or a ZIP file containing images."""
    workflow_id = f"cusfm_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    dataset_dir = PROCESSING_DIR / dataset_id
    images_dir = dataset_dir / "raw_images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Handle file uploads
    num_images = 0
    for file in files:
        content = await file.read()

        if file.filename.endswith('.zip'):
            # Extract ZIP file
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                for name in zf.namelist():
                    if name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        zf.extract(name, images_dir)
                        num_images += 1
        else:
            # Save individual image
            dest = images_dir / file.filename
            with open(dest, 'wb') as f:
                f.write(content)
            num_images += 1

    if num_images < 3:
        return JSONResponse(
            {"error": f"Need at least 3 images, got {num_images}"},
            status_code=400
        )

    # Flatten any nested directories from ZIP extraction
    for root, dirs, fnames in os.walk(images_dir):
        for fname in fnames:
            src = Path(root) / fname
            if src.parent != images_dir and fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                shutil.move(str(src), str(images_dir / fname))

    # Prepare cuSFM input structure
    input_dir = prepare_cusfm_input(
        dataset_dir=dataset_dir,
        images_dir=images_dir,
        image_width=image_width,
        image_height=image_height,
        fx=fx,
        fy=fy,
        fps=fps
    )

    output_dir = OUTPUT_DIR / dataset_id / "cusfm"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize workflow state
    active_workflows[workflow_id] = {
        "workflow_id": workflow_id,
        "status": "running",
        "progress": 0.0,
        "current_step": "Preparing cuSFM input",
        "dataset_id": dataset_id,
        "num_images": num_images,
        "feature_type": feature_type,
        "error": None,
        "log": [],
        "started_at": datetime.now().isoformat()
    }

    # Run cuSFM in background
    background_tasks.add_task(
        run_cusfm_pipeline,
        workflow_id=workflow_id,
        input_dir=input_dir,
        output_dir=output_dir,
        feature_type=feature_type
    )

    return {
        "workflow_id": workflow_id,
        "status": "started",
        "num_images": num_images,
        "message": f"cuSFM processing started with {num_images} images"
    }


@app.post("/process-directory")
async def process_directory(
    background_tasks: BackgroundTasks,
    request: Dict[str, Any] = None
):
    """Process images from an existing directory (called by isaac-viewer workflow).
    Expects images already extracted to a shared volume."""
    if request is None:
        request = {}

    dataset_id = request.get("dataset_id", "dataset")
    images_path = request.get("images_path")
    image_width = request.get("image_width", 1920)
    image_height = request.get("image_height", 1080)
    fx = request.get("fx", 1000.0)
    fy = request.get("fy", 1000.0)
    fps = request.get("fps", 30.0)
    feature_type = request.get("feature_type", "aliked")

    if not images_path:
        return JSONResponse({"error": "images_path is required"}, status_code=400)

    images_dir = Path(images_path)
    if not images_dir.exists():
        return JSONResponse(
            {"error": f"Images directory not found: {images_path}"},
            status_code=400
        )

    # Count images
    image_files = [
        f for f in images_dir.iterdir()
        if f.suffix.lower() in ('.jpg', '.jpeg', '.png')
    ]
    if len(image_files) < 3:
        return JSONResponse(
            {"error": f"Need at least 3 images, found {len(image_files)}"},
            status_code=400
        )

    workflow_id = f"cusfm_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    dataset_dir = PROCESSING_DIR / dataset_id

    # Prepare cuSFM input structure
    input_dir = prepare_cusfm_input(
        dataset_dir=dataset_dir,
        images_dir=images_dir,
        image_width=image_width,
        image_height=image_height,
        fx=fx,
        fy=fy,
        fps=fps
    )

    output_dir = OUTPUT_DIR / dataset_id / "cusfm"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize workflow state
    active_workflows[workflow_id] = {
        "workflow_id": workflow_id,
        "status": "running",
        "progress": 0.0,
        "current_step": "Starting cuSFM pipeline",
        "dataset_id": dataset_id,
        "num_images": len(image_files),
        "feature_type": feature_type,
        "error": None,
        "log": [],
        "started_at": datetime.now().isoformat()
    }

    # Run cuSFM in background
    background_tasks.add_task(
        run_cusfm_pipeline,
        workflow_id=workflow_id,
        input_dir=input_dir,
        output_dir=output_dir,
        feature_type=feature_type
    )

    return {
        "workflow_id": workflow_id,
        "status": "started",
        "num_images": len(image_files),
        "message": f"cuSFM processing started with {len(image_files)} images"
    }


@app.get("/status/{workflow_id}")
async def get_status(workflow_id: str):
    """Get workflow status by ID"""
    if workflow_id not in active_workflows:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return active_workflows[workflow_id]


@app.get("/workflows")
async def list_workflows():
    """List all workflows"""
    return {"workflows": list(active_workflows.values())}


@app.get("/output/{dataset_id}/sparse")
async def get_sparse_output(dataset_id: str):
    """Get the COLMAP-compatible sparse output as a ZIP"""
    sparse_dir = OUTPUT_DIR / dataset_id / "cusfm" / "sparse"
    if not sparse_dir.exists():
        raise HTTPException(status_code=404, detail="Sparse output not found")

    # Create ZIP of sparse output
    zip_path = TEMP_DIR / f"{dataset_id}_sparse.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in sparse_dir.iterdir():
            zf.write(f, f"sparse/{f.name}")

    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"{dataset_id}_sparse.zip"
    )


@app.get("/output/{dataset_id}/poses")
async def get_poses(dataset_id: str):
    """Get the output poses in TUM format"""
    poses_dir = OUTPUT_DIR / dataset_id / "cusfm" / "output_poses"
    if not poses_dir.exists():
        raise HTTPException(status_code=404, detail="Poses output not found")

    # Return merged pose file if available
    merged = poses_dir / "merged_pose_file.tum"
    if merged.exists():
        return FileResponse(merged, media_type="text/plain")

    raise HTTPException(status_code=404, detail="Merged pose file not found")


@app.delete("/dataset/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """Delete dataset and its outputs"""
    deleted = False
    for base_dir in [PROCESSING_DIR, OUTPUT_DIR]:
        dataset_path = base_dir / dataset_id
        if dataset_path.exists():
            shutil.rmtree(dataset_path)
            deleted = True

    if deleted:
        return {"status": "ok", "dataset_id": dataset_id}
    raise HTTPException(status_code=404, detail="Dataset not found")
