"""
TRELLIS.2 3D Reconstruction Service
Takes rendered images of GARField-extracted Gaussians and reconstructs them
as full 3D meshes with PBR materials using Microsoft TRELLIS.2.

Output: GLB files viewable in the integrated Three.js viewer with orbit controls.
"""

import os
import io
import gc
import json
import logging
import threading
import uuid
import time
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
from PIL import Image
from fastapi import (
    FastAPI, File, UploadFile, Form, Query,
    HTTPException, BackgroundTasks
)
from fastapi.responses import JSONResponse, Response, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
TRELLIS_PORT = int(os.environ.get("TRELLIS_PORT", "8013"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/app/outputs"))
CACHE_DIR = Path(os.environ.get("CACHE_DIR", "/app/cache"))
MODEL_CACHE = Path(os.environ.get("MODEL_CACHE", "/app/model_cache"))

# Cleanup policy: keep at most MAX_MODELS reconstructed models at once,
# and expire any model older than MODEL_TTL_HOURS so it must be rebuilt.
MAX_MODELS = int(os.environ.get("TRELLIS_MAX_MODELS", "3"))
MODEL_TTL_SECONDS = float(os.environ.get("TRELLIS_MODEL_TTL_HOURS", "24")) * 3600
CLEANUP_INTERVAL_SECONDS = int(os.environ.get("TRELLIS_CLEANUP_INTERVAL", "3600"))

# Memory policy: unload the pipeline after this many idle minutes so the
# GB10's unified memory is available for training and other services.
IDLE_UNLOAD_SECONDS = float(os.environ.get("TRELLIS_IDLE_UNLOAD_MINUTES", "10")) * 60
# Watchdog: fail any generation that runs longer than this (thrashing guard)
JOB_TIMEOUT_SECONDS = float(os.environ.get("TRELLIS_JOB_TIMEOUT_MINUTES", "30")) * 60

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_CACHE.mkdir(parents=True, exist_ok=True)

# Global state
pipeline = None
pipeline_loaded = False
pipeline_last_used = 0.0
_pipeline_lock = threading.Lock()   # prevents concurrent double-load
_job_semaphore = asyncio.Semaphore(1)  # serialize GPU jobs (one at a time)
reconstruction_jobs: Dict[str, Dict[str, Any]] = {}
job_metadata: Dict[str, Dict[str, Any]] = {}

app = FastAPI(
    title="TRELLIS.2 3D Reconstruction Service",
    description="Image-to-3D reconstruction using Microsoft TRELLIS.2",
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

# Mount static files for the Three.js viewer
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception:
    logger.warning("Static directory not found, viewer page will be embedded")


def _delete_job_files(job: Dict[str, Any]):
    """Delete the GLB and input image files belonging to a job."""
    for key in ("output_glb", "input_image"):
        path = job.get(key)
        if path and Path(path).exists():
            Path(path).unlink()


def cleanup_models() -> Dict[str, Any]:
    """Enforce the model retention policy on disk and in memory.

    - Keeps only the MAX_MODELS newest GLB models (by file mtime).
    - Deletes any model older than MODEL_TTL_SECONDS (24h default) so a
      fresh one must be rebuilt on the next request.
    - Also sweeps orphaned files left on disk by previous container runs.
    """
    now = time.time()
    removed: List[str] = []

    # All GLB models on disk, newest first (covers orphans from restarts too)
    glbs = sorted(
        OUTPUT_DIR.glob("*.glb"),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )

    for rank, glb in enumerate(glbs):
        age = now - glb.stat().st_mtime
        if rank < MAX_MODELS and age <= MODEL_TTL_SECONDS:
            continue  # within retention window — keep

        job_id = glb.stem
        # Skip models belonging to jobs that are still in flight
        job = reconstruction_jobs.get(job_id)
        if job and job["status"] not in ("completed", "failed"):
            continue

        # Remove GLB, its input image, and any in-memory job state
        glb.unlink(missing_ok=True)
        input_png = OUTPUT_DIR / f"{job_id}_input.png"
        input_png.unlink(missing_ok=True)
        reconstruction_jobs.pop(job_id, None)
        job_metadata.pop(job_id, None)
        removed.append(job_id)
        reason = "expired (>24h)" if age > MODEL_TTL_SECONDS else "over model limit"
        logger.info(f"[cleanup] Removed model {job_id}: {reason}")

    return {"removed": removed, "kept": min(len(glbs) - len(removed), MAX_MODELS)}


async def _periodic_cleanup():
    """Background loop enforcing the retention policy every hour."""
    while True:
        try:
            result = cleanup_models()
            if result["removed"]:
                logger.info(f"[cleanup] Periodic sweep removed {len(result['removed'])} model(s)")
        except Exception as e:
            logger.error(f"[cleanup] Periodic sweep failed: {e}")
        await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)


def _ensure_alpha_channel(image: Image.Image) -> Image.Image:
    """Ensure image sent to TRELLIS has a valid alpha channel so that
    preprocess_image finds foreground pixels and doesn't crash.
    If the image has no alpha or all-zero alpha, add a full-white alpha."""
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    alpha = np.array(image)[:, :, 3]
    if np.all(alpha == 0):
        # All transparent — make fully opaque so rembg can segment it
        arr = np.array(image)
        arr[:, :, 3] = 255
        image = Image.fromarray(arr)
    return image


def load_pipeline():
    """Load the TRELLIS.2 pipeline (thread-safe, lazy)."""
    global pipeline, pipeline_loaded, pipeline_last_used

    # Lock prevents the startup/first-request race that double-loaded
    # the 4B pipeline and doubled GPU memory usage.
    with _pipeline_lock:
        if pipeline is not None:
            pipeline_last_used = time.time()
            return True

        try:
            logger.info("Loading TRELLIS.2 pipeline...")

            os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
            os.environ.setdefault(
                'PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True'
            )

            from trellis2.pipelines import Trellis2ImageTo3DPipeline
            pipeline = Trellis2ImageTo3DPipeline.from_pretrained(
                "microsoft/TRELLIS.2-4B"
            )
            pipeline.cuda()
            pipeline_loaded = True
            pipeline_last_used = time.time()
            logger.info("TRELLIS.2 pipeline loaded successfully")
            return True

        except ImportError as e:
            logger.error(f"TRELLIS.2 not installed: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to load TRELLIS.2 pipeline: {e}")
            return False


def unload_pipeline():
    """Release the pipeline and free GPU memory (idle-unload policy)."""
    global pipeline, pipeline_loaded

    with _pipeline_lock:
        if pipeline is None:
            return False
        logger.info("Unloading TRELLIS.2 pipeline (idle) to free GPU memory")
        pipeline = None
        pipeline_loaded = False
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        return True


async def _idle_unload_loop():
    """Unload the pipeline after IDLE_UNLOAD_SECONDS without a job."""
    while True:
        await asyncio.sleep(60)
        try:
            if pipeline is None:
                continue
            # Never unload while a job is running
            if _job_semaphore.locked():
                continue
            if time.time() - pipeline_last_used > IDLE_UNLOAD_SECONDS:
                await asyncio.to_thread(unload_pipeline)
        except Exception as e:
            logger.error(f"[idle-unload] {e}")


async def run_reconstruction(job_id: str, image: Image.Image):
    """Run TRELLIS.2 reconstruction in a background thread.

    Jobs are serialized through a semaphore (one GPU job at a time) and
    guarded by a watchdog timeout so a thrashing run cannot hold the GPU
    forever.
    """
    global pipeline_last_used
    job = reconstruction_jobs[job_id]

    # Serialize: only one reconstruction may use the GPU at a time
    job["status"] = "waiting"
    job["current_step"] = "Waiting for GPU (another job is running)"
    async with _job_semaphore:
        try:
            job["status"] = "loading_model"
            job["progress"] = 0.1
            job["current_step"] = "Loading TRELLIS.2 model"

            ready = await asyncio.to_thread(load_pipeline)
            if not ready or pipeline is None:
                job["status"] = "failed"
                job["error"] = "TRELLIS.2 pipeline not available"
                return

            # Run image-to-3D generation
            job["status"] = "generating"
            job["progress"] = 0.3
            job["current_step"] = "Generating 3D mesh from image"

            def _generate():
                import o_voxel
                # Ensure valid alpha so TRELLIS preprocess_image can find foreground
                input_img = _ensure_alpha_channel(image)
                try:
                    mesh = pipeline.run(input_img)[0]
                except ValueError as e:
                    if "zero-size array" in str(e):
                        # rembg failed to segment — retry with fully opaque alpha
                        # so TRELLIS skips rembg and uses the image directly
                        logger.warning(f"Background removal failed, retrying with opaque alpha")
                        arr = np.array(input_img.convert('RGBA'))
                        arr[:, :, 3] = 255
                        opaque_img = Image.fromarray(arr)
                        mesh = pipeline.run(opaque_img)[0]
                    else:
                        raise
                mesh.simplify(16777216)  # nvdiffrast limit
                return mesh

            # Watchdog: fail jobs stuck longer than JOB_TIMEOUT_SECONDS
            # (e.g. thrashing under memory pressure) instead of holding
            # the GPU indefinitely.
            try:
                mesh = await asyncio.wait_for(
                    asyncio.to_thread(_generate),
                    timeout=JOB_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                raise Exception(
                    f"Generation exceeded {JOB_TIMEOUT_SECONDS/60:.0f} min "
                    "watchdog timeout - likely memory thrashing. Free GPU "
                    "memory and retry (service restart may be required)."
                )

            # Export to GLB
            job["status"] = "exporting"
            job["progress"] = 0.8
            job["current_step"] = "Exporting GLB with PBR materials"

            output_path = OUTPUT_DIR / f"{job_id}.glb"

            def _export():
                import o_voxel
                glb = o_voxel.postprocess.to_glb(
                    vertices=mesh.vertices,
                    faces=mesh.faces,
                    attr_volume=mesh.attrs,
                    coords=mesh.coords,
                    attr_layout=mesh.layout,
                    voxel_size=mesh.voxel_size,
                    aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                    decimation_target=1000000,
                    texture_size=4096,
                    remesh=True,
                    remesh_band=1,
                    remesh_project=0,
                    verbose=True
                )
                glb.export(str(output_path), extension_webp=True)

            await asyncio.wait_for(
                asyncio.to_thread(_export),
                timeout=JOB_TIMEOUT_SECONDS,
            )

            # Save input image for reference
            input_path = OUTPUT_DIR / f"{job_id}_input.png"
            image.save(str(input_path))

            job["status"] = "completed"
            job["progress"] = 1.0
            job["current_step"] = "Reconstruction complete"
            job["output_glb"] = str(output_path)
            job["input_image"] = str(input_path)
            job["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"[{job_id}] Reconstruction complete: {output_path}")

            # Enforce retention immediately so no more than MAX_MODELS remain
            cleanup_models()

        except asyncio.TimeoutError:
            logger.error(f"[{job_id}] GLB export exceeded watchdog timeout")
            job["status"] = "failed"
            job["error"] = "GLB export exceeded watchdog timeout"
            job["current_step"] = "Failed: export watchdog timeout"
        except Exception as e:
            logger.error(f"[{job_id}] Reconstruction failed: {e}")
            import traceback
            traceback.print_exc()
            job["status"] = "failed"
            job["error"] = str(e)
            job["current_step"] = f"Failed: {str(e)}"
        finally:
            # Release cached intermediate tensors after every job and
            # refresh the idle timer for the unload policy.
            pipeline_last_used = time.time()
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass


# ===== Health & Info =====

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "trellis2",
        "pipeline_loaded": pipeline is not None,
        "job_running": _job_semaphore.locked(),
        "idle_unload_minutes": IDLE_UNLOAD_SECONDS / 60,
        "job_timeout_minutes": JOB_TIMEOUT_SECONDS / 60,
        "jobs_count": len(reconstruction_jobs),
        "max_models": MAX_MODELS,
        "model_ttl_hours": MODEL_TTL_SECONDS / 3600,
    }


@app.post("/cleanup")
async def trigger_cleanup():
    """Manually run the model retention sweep (max models + 24h TTL)."""
    return cleanup_models()


@app.get("/", response_class=HTMLResponse)
async def root():
    """Redirect to API docs."""
    return """
    <html><head><meta http-equiv="refresh" content="0;url=/api" /></head>
    <body>Redirecting to <a href="/api">API docs</a>...</body></html>
    """


# ===== Reconstruction Endpoints =====

@app.post("/reconstruct")
async def start_reconstruction(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    source_job_id: str = Form(""),
    label: str = Form(""),
):
    """
    Start a TRELLIS.2 image-to-3D reconstruction.

    - image: PNG/JPG of the object to reconstruct (e.g. rendered GARField extraction)
    - source_job_id: Optional GARField extraction job_id for linking
    - label: Optional label for the reconstruction
    """
    job_id = str(uuid.uuid4())[:8]

    content = await image.read()
    pil_image = Image.open(io.BytesIO(content)).convert("RGB")

    reconstruction_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0.0,
        "current_step": "Queued for reconstruction",
        "source_job_id": source_job_id,
        "label": label,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "completed_at": None,
        "output_glb": None,
        "input_image": None,
        "error": None,
    }

    # Initialize metadata
    job_metadata[job_id] = {
        "label": label,
        "text": "",
        "files": [],
        "source_job_id": source_job_id,
    }

    background_tasks.add_task(run_reconstruction, job_id, pil_image)

    return {
        "job_id": job_id,
        "status": "queued",
        "viewer_url": f"/viewer/{job_id}",
        "message": "Reconstruction started",
    }


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """Get reconstruction job status."""
    if job_id not in reconstruction_jobs:
        raise HTTPException(404, "Job not found")
    return reconstruction_jobs[job_id]


@app.get("/jobs")
async def list_jobs():
    """List all reconstruction jobs."""
    return {
        "jobs": list(reconstruction_jobs.values()),
        "count": len(reconstruction_jobs),
    }


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a reconstruction job and its files."""
    if job_id not in reconstruction_jobs:
        raise HTTPException(404, "Job not found")

    job = reconstruction_jobs[job_id]
    _delete_job_files(job)

    del reconstruction_jobs[job_id]
    job_metadata.pop(job_id, None)
    return {"status": "deleted", "job_id": job_id}


# ===== Asset Download =====

@app.get("/download/{job_id}")
async def download_glb(job_id: str):
    """Download the reconstructed GLB file."""
    if job_id not in reconstruction_jobs:
        raise HTTPException(404, "Job not found")

    job = reconstruction_jobs[job_id]
    if job["status"] != "completed" or not job.get("output_glb"):
        raise HTTPException(400, "Reconstruction not complete")

    glb_path = Path(job["output_glb"])
    if not glb_path.exists():
        raise HTTPException(404, "GLB file not found")

    label = job.get("label", job_id)
    return FileResponse(
        path=glb_path,
        filename=f"{label}_{job_id}.glb",
        media_type="model/gltf-binary",
    )


@app.get("/input_image/{job_id}")
async def get_input_image(job_id: str):
    """Get the input image used for reconstruction."""
    if job_id not in reconstruction_jobs:
        raise HTTPException(404, "Job not found")

    job = reconstruction_jobs[job_id]
    img_path = job.get("input_image")
    if not img_path or not Path(img_path).exists():
        raise HTTPException(404, "Input image not found")

    return FileResponse(path=img_path, media_type="image/png")


# ===== Three.js Viewer =====

@app.get("/viewer/{job_id}", response_class=HTMLResponse)
async def viewer_page(job_id: str):
    """Serve the interactive Three.js 3D viewer for a reconstruction."""
    if job_id not in reconstruction_jobs:
        raise HTTPException(404, "Job not found")

    # Serve from static file if available, otherwise use embedded
    static_path = Path("static/viewer.html")
    if static_path.exists():
        html = static_path.read_text()
        html = html.replace("{{JOB_ID}}", job_id)
        return HTMLResponse(html)

    return HTMLResponse(f"""
    <html><body>
    <h1>Viewer for {job_id}</h1>
    <p>Static viewer.html not found. Place it in static/viewer.html</p>
    </body></html>
    """)


# ===== Metadata Endpoints (like SAM/GARField) =====

@app.get("/metadata/{job_id}")
async def get_metadata(job_id: str):
    """Get metadata for a reconstruction."""
    meta = job_metadata.get(job_id)
    if not meta:
        return {"job_id": job_id, "metadata": None}

    serializable = {
        "label": meta.get("label", ""),
        "text": meta.get("text", ""),
        "source_job_id": meta.get("source_job_id", ""),
        "files": [
            {
                "idx": i,
                "name": f.get("name"),
                "size": f.get("size"),
                "content_type": f.get("content_type"),
            }
            for i, f in enumerate(meta.get("files", []))
        ],
    }
    return {"job_id": job_id, "metadata": serializable}


@app.post("/metadata")
async def save_metadata(
    job_id: str = Form(...),
    text: str = Form(""),
    label: str = Form(""),
):
    """Save text metadata for a reconstruction."""
    if job_id not in job_metadata:
        job_metadata[job_id] = {
            "label": "", "text": "", "files": [], "source_job_id": ""
        }
    job_metadata[job_id]["text"] = text
    job_metadata[job_id]["label"] = label
    logger.info(f"Saved metadata for {job_id}: label={label}")
    return {"status": "ok", "job_id": job_id}


@app.post("/metadata/upload")
async def upload_metadata_files(
    job_id: str = Form(...),
    files: List[UploadFile] = File(...),
):
    """Upload documents associated with a reconstruction for RAG."""
    if job_id not in job_metadata:
        job_metadata[job_id] = {
            "label": "", "text": "", "files": [], "source_job_id": ""
        }

    uploaded = []
    for file in files:
        content = await file.read()
        file_info = {
            "name": file.filename,
            "content_type": file.content_type,
            "size": f"{len(content) / 1024:.1f} KB",
            "data": content,
        }
        job_metadata[job_id]["files"].append(file_info)
        uploaded.append({"name": file.filename, "size": file_info["size"]})

    logger.info(f"Uploaded {len(uploaded)} files for reconstruction {job_id}")
    return {"status": "ok", "job_id": job_id, "uploaded": uploaded}


@app.get("/metadata/{job_id}/file/{file_idx}")
async def get_metadata_file(job_id: str, file_idx: int):
    """Serve an uploaded metadata document."""
    if job_id not in job_metadata:
        raise HTTPException(404, "Job not found")

    files = job_metadata[job_id].get("files", [])
    if file_idx < 0 or file_idx >= len(files):
        raise HTTPException(404, "File not found")

    file_info = files[file_idx]
    return Response(
        content=file_info.get("data", b""),
        media_type=file_info.get("content_type", "application/octet-stream"),
        headers={
            "Content-Disposition":
                f'inline; filename="{file_info.get("name", "file")}"'
        },
    )


@app.get("/metadata_all")
async def get_all_metadata():
    """Get metadata summary for all reconstructions."""
    result = {}
    for jid, meta in job_metadata.items():
        result[jid] = {
            "label": meta.get("label", ""),
            "text": meta.get("text", ""),
            "files_count": len(meta.get("files", [])),
            "source_job_id": meta.get("source_job_id", ""),
        }
    return {"metadata": result, "count": len(result)}


@app.on_event("startup")
async def startup_tasks():
    """Start cleanup and idle-unload loops. The pipeline itself is loaded
    lazily on the first reconstruction request and unloaded when idle, so
    the GB10's unified memory stays free for training between jobs."""
    # Sweep stale models left over from previous runs, then keep sweeping hourly
    cleanup_models()
    asyncio.create_task(_periodic_cleanup())
    asyncio.create_task(_idle_unload_loop())
    logger.info(
        f"TRELLIS.2 service ready (lazy load, idle unload after "
        f"{IDLE_UNLOAD_SECONDS/60:.0f} min, job watchdog "
        f"{JOB_TIMEOUT_SECONDS/60:.0f} min)"
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=TRELLIS_PORT)
