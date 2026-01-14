"""
SAM-2 Segmentation Service
GPU-accelerated object segmentation using Segment Anything Model 2

Features:
- Zero-shot object segmentation
- Video consistency tracking (SAM-2)
- Point/box prompt-based segmentation
- Automatic mask generation
- Integration with fvdb-viewer for 3D splat labeling
"""

import os
import uuid
import json
import asyncio
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
from datetime import datetime

import numpy as np
import cv2
from PIL import Image
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import aiofiles
import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", "/app/uploads"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/app/outputs"))
MODEL_DIR = Path(os.environ.get("SAM2_CHECKPOINT_DIR", "/app/models/sam2"))
CACHE_DIR = Path(os.environ.get("TORCH_HOME", "/app/cache"))

VIEWER_SERVICE_URL = os.environ.get("VIEWER_SERVICE_URL", "http://fvdb-viewer:8085")
TRAINING_SERVICE_URL = os.environ.get("TRAINING_SERVICE_URL", "http://fvdb-training-gpu:8000")
RENDERING_SERVICE_URL = os.environ.get("RENDERING_SERVICE_URL", "http://fvdb-rendering:8001")

for d in [UPLOAD_DIR, OUTPUT_DIR, MODEL_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="SAM-2 Segmentation Service",
    description="GPU-accelerated object segmentation using Segment Anything Model 2",
    version="1.0.0",
    docs_url="/api",
    redoc_url="/api/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sam2_predictor = None
sam2_video_predictor = None
model_loaded = False
jobs: Dict[str, Dict[str, Any]] = {}


class ModelSize(str, Enum):
    TINY = "tiny"
    SMALL = "small"
    BASE_PLUS = "base_plus"
    LARGE = "large"


class SegmentationMode(str, Enum):
    POINT = "point"
    BOX = "box"
    AUTO = "auto"
    VIDEO = "video"


class SegmentationResult(BaseModel):
    job_id: str
    status: str
    masks: Optional[List[str]] = None
    labels: Optional[List[str]] = None
    scores: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None


def get_model_config(size: ModelSize) -> Tuple[str, str]:
    configs = {
        ModelSize.TINY: ("sam2_hiera_tiny.pt", "sam2_hiera_t.yaml"),
        ModelSize.SMALL: ("sam2_hiera_small.pt", "sam2_hiera_s.yaml"),
        ModelSize.BASE_PLUS: ("sam2_hiera_base_plus.pt", "sam2_hiera_b+.yaml"),
        ModelSize.LARGE: ("sam2_hiera_large.pt", "sam2_hiera_l.yaml"),
    }
    return configs.get(size, configs[ModelSize.BASE_PLUS])


async def download_checkpoint(size: ModelSize):
    checkpoint, _ = get_model_config(size)
    base_url = "https://dl.fbaipublicfiles.com/segment_anything_2/072824"
    url = f"{base_url}/{checkpoint}"
    checkpoint_path = MODEL_DIR / checkpoint
    
    logger.info(f"Downloading {checkpoint} from {url}")
    async with httpx.AsyncClient(timeout=600.0) as client:
        response = await client.get(url, follow_redirects=True)
        response.raise_for_status()
        async with aiofiles.open(checkpoint_path, 'wb') as f:
            await f.write(response.content)
    logger.info(f"Checkpoint downloaded to {checkpoint_path}")


async def load_sam2_model(size: ModelSize = ModelSize.BASE_PLUS):
    global sam2_predictor, sam2_video_predictor, model_loaded
    
    if model_loaded:
        return
    
    logger.info(f"Loading SAM-2 model (size: {size})...")
    
    try:
        from sam2.build_sam import build_sam2, build_sam2_video_predictor
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        
        checkpoint, config = get_model_config(size)
        checkpoint_path = MODEL_DIR / checkpoint
        
        if not checkpoint_path.exists():
            logger.info(f"Downloading SAM-2 checkpoint: {checkpoint}")
            await download_checkpoint(size)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        sam2_model = build_sam2(config, str(checkpoint_path), device=device)
        sam2_predictor = SAM2ImagePredictor(sam2_model)
        sam2_video_predictor = build_sam2_video_predictor(config, str(checkpoint_path), device=device)
        
        model_loaded = True
        logger.info("SAM-2 model loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load SAM-2 model: {e}")
        raise


def create_mask_overlay(image: np.ndarray, mask_paths: List[str]) -> np.ndarray:
    overlay = image.copy()
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    
    for i, mask_path in enumerate(mask_paths):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        color = colors[i % len(colors)]
        colored_mask = np.zeros_like(overlay)
        colored_mask[mask > 127] = color
        overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 0.4, 0)
    
    return overlay


@app.on_event("startup")
async def startup_event():
    try:
        await load_sam2_model()
    except Exception as e:
        logger.warning(f"Model not loaded on startup: {e}")


@app.get("/health")
async def health_check():
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else None
    return {
        "status": "healthy",
        "service": "SAM-2 Segmentation Service",
        "model_loaded": model_loaded,
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/api/models")
async def list_available_models():
    return {
        "models": [
            {"id": "tiny", "name": "SAM-2 Tiny", "description": "Fastest, lower quality"},
            {"id": "small", "name": "SAM-2 Small", "description": "Fast, good quality"},
            {"id": "base_plus", "name": "SAM-2 Base+", "description": "Balanced (recommended)"},
            {"id": "large", "name": "SAM-2 Large", "description": "Best quality, slower"},
        ],
        "current": "base_plus" if model_loaded else None
    }


@app.post("/api/segment/image", response_model=SegmentationResult)
async def segment_image(
    file: UploadFile = File(...),
    mode: SegmentationMode = Form(SegmentationMode.AUTO),
    points_json: Optional[str] = Form(None),
    boxes_json: Optional[str] = Form(None),
    multimask_output: bool = Form(False)
):
    """Segment objects in a single image"""
    if not model_loaded:
        await load_sam2_model()
    
    job_id = str(uuid.uuid4())
    upload_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
    async with aiofiles.open(upload_path, 'wb') as f:
        content = await file.read()
        await f.write(content)
    
    points = json.loads(points_json) if points_json else None
    boxes = json.loads(boxes_json) if boxes_json else None
    
    jobs[job_id] = {
        "job_id": job_id,
        "status": "processing",
        "mode": mode,
        "image_path": str(upload_path),
        "created_at": datetime.utcnow().isoformat()
    }
    
    try:
        image = cv2.imread(str(upload_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        sam2_predictor.set_image(image)
        
        output_dir = OUTPUT_DIR / job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        masks_data, scores_data, labels_data = [], [], []
        
        if mode == SegmentationMode.AUTO:
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            mask_generator = SAM2AutomaticMaskGenerator(sam2_predictor.model)
            masks = mask_generator.generate(image)
            
            for i, mask_data in enumerate(masks):
                mask = mask_data['segmentation']
                score = mask_data['stability_score']
                mask_path = output_dir / f"mask_{i:03d}.png"
                cv2.imwrite(str(mask_path), (mask * 255).astype(np.uint8))
                masks_data.append(str(mask_path))
                scores_data.append(float(score))
                labels_data.append(f"object_{i}")
        
        elif mode == SegmentationMode.POINT and points:
            point_coords = np.array([[p['x'], p['y']] for p in points])
            point_labels = np.array([p.get('label', 1) for p in points])
            masks, scores, _ = sam2_predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=multimask_output
            )
            for i, (mask, score) in enumerate(zip(masks, scores)):
                mask_path = output_dir / f"mask_{i:03d}.png"
                cv2.imwrite(str(mask_path), (mask * 255).astype(np.uint8))
                masks_data.append(str(mask_path))
                scores_data.append(float(score))
                labels_data.append(f"segment_{i}")
        
        elif mode == SegmentationMode.BOX and boxes:
            for i, box in enumerate(boxes):
                box_array = np.array([box['x1'], box['y1'], box['x2'], box['y2']])
                masks, scores, _ = sam2_predictor.predict(box=box_array, multimask_output=multimask_output)
                best_idx = np.argmax(scores)
                mask_path = output_dir / f"mask_{i:03d}.png"
                cv2.imwrite(str(mask_path), (masks[best_idx] * 255).astype(np.uint8))
                masks_data.append(str(mask_path))
                scores_data.append(float(scores[best_idx]))
                labels_data.append(f"box_segment_{i}")
        
        overlay_path = output_dir / "overlay.png"
        overlay = create_mask_overlay(image, masks_data)
        cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        
        jobs[job_id].update({
            "status": "completed",
            "masks": masks_data,
            "scores": scores_data,
            "labels": labels_data,
            "overlay": str(overlay_path),
            "completed_at": datetime.utcnow().isoformat()
        })
        
        return SegmentationResult(
            job_id=job_id,
            status="completed",
            masks=masks_data,
            labels=labels_data,
            scores=scores_data,
            metadata={"overlay": str(overlay_path), "num_masks": len(masks_data)}
        )
        
    except Exception as e:
        logger.error(f"Segmentation failed: {e}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/segment/video")
async def segment_video(
    file: UploadFile = File(...),
    prompts_json: str = Form(...),
    track_objects: bool = Form(True),
    background_tasks: BackgroundTasks = None
):
    """Segment and track objects across video frames with temporal consistency"""
    if not model_loaded:
        await load_sam2_model()
    
    job_id = str(uuid.uuid4())
    video_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
    async with aiofiles.open(video_path, 'wb') as f:
        content = await file.read()
        await f.write(content)
    
    frame_prompts = json.loads(prompts_json)
    jobs[job_id] = {
        "job_id": job_id,
        "status": "processing",
        "video_path": str(video_path),
        "created_at": datetime.utcnow().isoformat()
    }
    
    background_tasks.add_task(process_video_segmentation, job_id, video_path, frame_prompts, track_objects)
    return {"job_id": job_id, "status": "processing"}


async def process_video_segmentation(job_id: str, video_path: Path, frame_prompts: Dict, track_objects: bool):
    """Background task for video segmentation"""
    try:
        output_dir = OUTPUT_DIR / job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        frames_dir = output_dir / "frames"
        masks_dir = output_dir / "masks"
        frames_dir.mkdir(exist_ok=True)
        masks_dir.mkdir(exist_ok=True)
        
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_paths = []
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = frames_dir / f"frame_{idx:05d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            frame_paths.append(str(frame_path))
            idx += 1
        cap.release()
        
        inference_state = sam2_video_predictor.init_state(video_path=str(frames_dir))
        
        for frame_idx_str, points in frame_prompts.items():
            frame_idx = int(frame_idx_str)
            point_coords = np.array([[p['x'], p['y']] for p in points])
            point_labels = np.array([p.get('label', 1) for p in points])
            
            sam2_video_predictor.add_new_points(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=1,
                points=point_coords,
                labels=point_labels
            )
        
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in sam2_video_predictor.propagate_in_video(inference_state):
            mask = (out_mask_logits[0] > 0).cpu().numpy().squeeze()
            mask_path = masks_dir / f"mask_{out_frame_idx:05d}.png"
            cv2.imwrite(str(mask_path), (mask * 255).astype(np.uint8))
            video_segments[out_frame_idx] = str(mask_path)
        
        jobs[job_id].update({
            "status": "completed",
            "frame_count": frame_count,
            "fps": fps,
            "masks_dir": str(masks_dir),
            "frames_dir": str(frames_dir),
            "segments": video_segments,
            "completed_at": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Video segmentation failed: {e}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)


@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get segmentation job status"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@app.get("/api/jobs")
async def list_jobs():
    """List all segmentation jobs"""
    return {"jobs": list(jobs.values())}


@app.get("/api/mask/{job_id}/{mask_idx}")
async def get_mask(job_id: str, mask_idx: int):
    """Get a specific mask image"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    if "masks" not in job or mask_idx >= len(job["masks"]):
        raise HTTPException(status_code=404, detail="Mask not found")
    
    return FileResponse(job["masks"][mask_idx], media_type="image/png")


@app.get("/api/overlay/{job_id}")
async def get_overlay(job_id: str):
    """Get segmentation overlay image"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    if "overlay" not in job:
        raise HTTPException(status_code=404, detail="Overlay not found")
    
    return FileResponse(job["overlay"], media_type="image/png")


@app.post("/api/segment/splat-render")
async def segment_splat_render(
    model_name: str = Form(...),
    camera_position: str = Form(None),
    mode: SegmentationMode = Form(SegmentationMode.AUTO),
    points_json: Optional[str] = Form(None)
):
    """
    Render a view from a trained splat model and segment it.
    Integrates with fvdb-viewer/rendering service.
    """
    if not model_loaded:
        await load_sam2_model()
    
    job_id = str(uuid.uuid4())
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            render_params = {"model": model_name}
            if camera_position:
                render_params["camera"] = camera_position
            
            response = await client.get(f"{RENDERING_SERVICE_URL}/render", params=render_params)
            response.raise_for_status()
            
            render_path = UPLOAD_DIR / f"{job_id}_render.png"
            async with aiofiles.open(render_path, 'wb') as f:
                await f.write(response.content)
        
        image = cv2.imread(str(render_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        sam2_predictor.set_image(image)
        
        output_dir = OUTPUT_DIR / job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        masks_data, scores_data, labels_data = [], [], []
        
        if mode == SegmentationMode.AUTO:
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            mask_generator = SAM2AutomaticMaskGenerator(sam2_predictor.model)
            masks = mask_generator.generate(image)
            
            for i, mask_data in enumerate(masks):
                mask = mask_data['segmentation']
                score = mask_data['stability_score']
                mask_path = output_dir / f"mask_{i:03d}.png"
                cv2.imwrite(str(mask_path), (mask * 255).astype(np.uint8))
                masks_data.append(str(mask_path))
                scores_data.append(float(score))
                labels_data.append(f"object_{i}")
        
        overlay_path = output_dir / "overlay.png"
        overlay = create_mask_overlay(image, masks_data)
        cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        
        jobs[job_id] = {
            "job_id": job_id,
            "status": "completed",
            "model_name": model_name,
            "masks": masks_data,
            "scores": scores_data,
            "labels": labels_data,
            "overlay": str(overlay_path),
            "render_path": str(render_path),
            "completed_at": datetime.utcnow().isoformat()
        }
        
        return SegmentationResult(
            job_id=job_id,
            status="completed",
            masks=masks_data,
            labels=labels_data,
            scores=scores_data,
            metadata={"overlay": str(overlay_path), "model": model_name}
        )
        
    except Exception as e:
        logger.error(f"Splat render segmentation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/labels/save")
async def save_labels(
    job_id: str = Form(...),
    labels_json: str = Form(...)
):
    """Save user-defined labels for segments"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    labels = json.loads(labels_json)
    jobs[job_id]["user_labels"] = labels
    
    labels_path = OUTPUT_DIR / job_id / "labels.json"
    async with aiofiles.open(labels_path, 'w') as f:
        await f.write(json.dumps(labels, indent=2))
    
    return {"status": "saved", "job_id": job_id, "labels": labels}


try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except:
    pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
