#!/usr/bin/env python3
"""
Training subprocess - runs training in isolation so it doesn't block API
"""
import sys
import json
import logging
import ssl
import os
import time
import threading
import signal
from pathlib import Path
from datetime import datetime

# Disable SSL verification (for downloading pretrained weights)
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# State file for communication with main process
def update_state(state_file: Path, updates: dict):
    """Update job state file"""
    state = {}
    if state_file.exists():
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
        except:
            pass
    state.update(updates)
    with open(state_file, 'w') as f:
        json.dump(state, f)

def count_registered_images(sparse_dir: Path) -> int:
    """Count registered images in a COLMAP sparse model"""
    import struct
    images_bin = sparse_dir / 'images.bin'
    if images_bin.exists():
        with open(images_bin, 'rb') as f:
            return struct.unpack('<Q', f.read(8))[0]
    return 0

def find_colmap_dir(dataset_path: Path) -> Path:
    """Find COLMAP output directory with most registered images"""
    candidates = []
    
    # Check numbered sparse models (0, 1, 2, etc.)
    sparse_base = dataset_path / 'sparse'
    if sparse_base.exists():
        for subdir in sorted(sparse_base.iterdir()):
            if subdir.is_dir() and (subdir / 'cameras.bin').exists():
                count = count_registered_images(subdir)
                candidates.append((count, subdir))
                logger.info(f"Found sparse model {subdir.name}: {count} images")
    
    # If we have candidates, use the one with most images
    if candidates:
        candidates.sort(reverse=True)  # Sort by count descending
        best = candidates[0]
        logger.info(f"Using sparse model with {best[0]} registered images: {best[1]}")
        return best[1]
    
    # Fallback to old logic
    for sparse_path in ['sparse/0', 'sparse', 'colmap/sparse/0', 'output/sparse/0']:
        full_path = dataset_path / sparse_path
        if full_path.exists() and (full_path / 'cameras.bin').exists():
            return full_path
        if full_path.exists() and (full_path / 'cameras.txt').exists():
            return full_path
    
    # Search recursively
    for root_path in dataset_path.rglob('cameras.bin'):
        return root_path.parent
    
    return None

def run_training(job_id: str, dataset_path: str, num_steps: int, output_name: str, output_dir: str, use_mcmc: bool = False):
    """Run the actual training"""
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    state_file = output_dir / f"{job_id}_state.json"
    
    try:
        # Initialize state
        update_state(state_file, {
            "job_id": job_id,
            "status": "loading_data",
            "progress": 0.1,
            "message": "Loading COLMAP scene...",
            "started_at": datetime.now().isoformat()
        })
        
        # Import fVDB
        import fvdb
        try:
            import fvdb_reality_capture as frc
        except ImportError as e:
            raise Exception(f"fVDB Reality Capture not available: {e}")
        
        # Find COLMAP directory
        colmap_dir = find_colmap_dir(dataset_path)
        if not colmap_dir:
            raise Exception("Could not find COLMAP data in dataset")
        
        # Load scene
        scene_path = dataset_path if (dataset_path / "sparse").exists() else colmap_dir.parent
        logger.info(f"Loading scene from {scene_path}")
        scene = frc.sfm_scene.SfmScene.from_colmap(str(scene_path))
        
        optimizer_label = "MCMC" if use_mcmc else "Standard"
        update_state(state_file, {
            "status": "training",
            "progress": 0.2,
            "message": f"Loaded {len(scene.images)} images, starting {optimizer_label} training..."
        })
        
        # Configure training
        num_images = len(scene.images)
        steps_per_epoch = num_images
        total_epochs = int(num_steps / steps_per_epoch) if num_steps else 200
        refine_until = int(total_epochs * 0.95)
        
        config = frc.radiance_fields.GaussianSplatReconstructionConfig(
            max_epochs=total_epochs,
            refine_stop_epoch=refine_until,
            refine_every_epoch=0.5
        )
        
        # Calculate safe Gaussian cap for high-res images to prevent int32 overflow
        # in the CUDA rasterizer's cumulative tile-Gaussian intersection counter.
        # With tile_size=16: tiles = (W/16)*(H/16). Safe limit ≈ int32_max / avg_tiles_per_gaussian.
        # Conservative estimate: avg ~500 tiles/gaussian at high counts → cap at ~2M for large images.
        max_dim = max(scene.images[0].image_size) if scene.images else 0
        if max_dim > 1920:
            safe_max_gaussians = 1_500_000
            logger.info(f"High-res images ({max_dim}px max dim): capping Gaussians at {safe_max_gaussians:,}")
        else:
            safe_max_gaussians = -1  # unlimited for normal resolution
            logger.info(f"Normal-res images ({max_dim}px max dim): no Gaussian cap")
        
        # Select optimizer: MCMC or standard
        if use_mcmc:
            optimizer_config = frc.radiance_fields.GaussianSplatOptimizerMCMCConfig(
                max_gaussians=1_000_000,  # Cap to prevent OOM from unbounded 5% growth
                noise_lr=5e2,  # Reduced from 5e5 default to prevent position explosion
            )
            logger.info(f"Using MCMC optimizer: {num_steps} steps, {total_epochs} epochs, max_gaussians=1M, noise_lr=500")
        else:
            optimizer_config = frc.radiance_fields.GaussianSplatOptimizerConfig(
                max_gaussians=safe_max_gaussians
            )
            logger.info(f"Using standard optimizer: {num_steps} steps, {total_epochs} epochs, max_gaussians={safe_max_gaussians}")
        
        runner = frc.radiance_fields.GaussianSplatReconstruction.from_sfm_scene(
            scene,
            config=config,
            optimizer_config=optimizer_config
        )
        
        update_state(state_file, {
            "progress": 0.3,
            "message": f"{optimizer_label} training {num_steps} steps..."
        })
        
        # Train with progress and Gaussian count monitoring
        GAUSSIAN_KILL_THRESHOLD = 15_000_000  # GB10 has 120GB unified memory, can handle 15M+ easily
        MONITOR_INTERVAL = 30  # seconds - check every 30s for responsive progress
        monitor_stop = threading.Event()
        training_start_time = time.time()
        
        def monitor_training():
            """Background thread: track step progress, Gaussian count, and ETA."""
            while not monitor_stop.is_set():
                monitor_stop.wait(MONITOR_INTERVAL)
                if monitor_stop.is_set():
                    break
                try:
                    count = runner.model.num_gaussians
                    
                    # Get current step from the optimizer's internal counter
                    current_step = getattr(runner.optimizer, '_step_count', 0)
                    pct = min(current_step / num_steps * 100, 100) if num_steps > 0 else 0
                    
                    # Calculate ETA from elapsed time and step rate
                    elapsed = time.time() - training_start_time
                    if current_step > 0:
                        steps_per_sec = current_step / elapsed
                        remaining_steps = num_steps - current_step
                        eta_secs = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
                        if eta_secs > 3600:
                            eta_str = f"{eta_secs/3600:.1f}h"
                        elif eta_secs > 60:
                            eta_str = f"{eta_secs/60:.0f}min"
                        else:
                            eta_str = f"{eta_secs:.0f}s"
                        msg = f"Step {current_step:,}/{num_steps:,} ({pct:.0f}%) | {count:,} gaussians | ETA: {eta_str}"
                    else:
                        elapsed_min = elapsed / 60
                        msg = f"Initializing... {elapsed_min:.0f}min elapsed | {count:,} gaussians"
                    
                    # Training progress maps to 0.3-0.95 range in overall workflow
                    train_progress = 0.3 + (0.65 * min(current_step / num_steps, 1.0)) if num_steps > 0 else 0.3
                    
                    logger.info(f"[Monitor] {msg}")
                    update_state(state_file, {
                        "progress": round(train_progress, 3),
                        "gaussian_count": count,
                        "message": msg
                    })
                    
                    if count > GAUSSIAN_KILL_THRESHOLD:
                        logger.error(f"[Monitor] RUNAWAY DETECTED: {count:,} gaussians > {GAUSSIAN_KILL_THRESHOLD:,} threshold. Aborting!")
                        update_state(state_file, {
                            "status": "failed",
                            "message": f"Aborted: Gaussian count ({count:,}) exceeded safety limit ({GAUSSIAN_KILL_THRESHOLD:,})"
                        })
                        os.kill(os.getpid(), signal.SIGTERM)
                except Exception as e:
                    logger.warning(f"[Monitor] Check failed: {e}")
        
        monitor_thread = threading.Thread(target=monitor_training, daemon=True)
        monitor_thread.start()
        
        try:
            runner.optimize()
        finally:
            monitor_stop.set()
            monitor_thread.join(timeout=5)
        
        model = runner.model
        logger.info(f"Training complete. Final Gaussian count: {model.num_gaussians:,}")
        
        update_state(state_file, {
            "status": "exporting",
            "progress": 0.9,
            "message": "Exporting model..."
        })
        
        # Save outputs
        job_output_dir = output_dir / job_id
        job_output_dir.mkdir(exist_ok=True)
        
        ply_file = job_output_dir / f"{output_name}.ply"
        model.save_ply(str(ply_file), metadata=runner.reconstruction_metadata)
        
        # --- Post-training cleanup: remove floaters and artifacts ---
        update_state(state_file, {
            "status": "cleaning",
            "progress": 0.92,
            "message": "Cleaning Gaussians (removing floaters)..."
        })
        
        cleanup_stats = {}
        try:
            from clean_gaussians import clean_gaussians
            cleaned_ply = job_output_dir / f"{output_name}_cleaned.ply"
            cleanup_stats = clean_gaussians(
                str(ply_file), str(cleaned_ply),
                opacity_threshold=0.005,
                scale_percentile=99.5,
                spatial_percentile=99.0,
                neighbor_percentile=95.0,
                k_neighbors=10,
            )
            # Replace original with cleaned version
            import shutil
            shutil.move(str(ply_file), str(job_output_dir / f"{output_name}_raw.ply"))
            shutil.move(str(cleaned_ply), str(ply_file))
            logger.info(f"Gaussian cleanup complete: {cleanup_stats.get('removed', 0):,} removed "
                         f"({cleanup_stats.get('pct_removed', 0):.1f}%), "
                         f"{cleanup_stats.get('final', 0):,} kept")
        except Exception as e:
            logger.warning(f"Gaussian cleanup failed (non-fatal): {e}")
            cleanup_stats = {"error": str(e)}
        
        # Save metadata
        metadata = {
            "num_gaussians": model.num_gaussians,
            "num_gaussians_cleaned": cleanup_stats.get("final", model.num_gaussians),
            "device": str(model.device),
            "num_channels": model.num_channels,
            "num_images": len(scene.images),
            "training_steps": num_steps,
            "optimizer": "mcmc" if use_mcmc else "standard",
            "output_file": str(ply_file),
            "cleanup_stats": cleanup_stats
        }
        
        with open(job_output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Copy to shared models directory for immediate availability
        models_dir = Path("/app/models")
        if models_dir.exists():
            import shutil
            shared_ply = models_dir / f"{output_name}.ply"
            shutil.copy2(ply_file, shared_ply)
            logger.info(f"Copied model to shared directory: {shared_ply}")
            # Also copy metadata
            shared_meta = models_dir / f"{output_name}_metadata.json"
            shutil.copy2(job_output_dir / "metadata.json", shared_meta)
        
        # Complete
        update_state(state_file, {
            "status": "completed",
            "progress": 1.0,
            "message": "Training complete!",
            "completed_at": datetime.now().isoformat(),
            "output_files": [str(ply_file)],
            "metadata": metadata
        })
        
        logger.info(f"Training complete: {ply_file}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        update_state(state_file, {
            "status": "failed",
            "progress": 0,
            "message": f"Training failed: {str(e)}",
            "error": str(e)
        })
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 6:
        print("Usage: train_subprocess.py <job_id> <dataset_path> <num_steps> <output_name> <output_dir> [use_mcmc]")
        sys.exit(1)
    
    job_id = sys.argv[1]
    dataset_path = sys.argv[2]
    num_steps = int(sys.argv[3])
    output_name = sys.argv[4]
    output_dir = sys.argv[5]
    use_mcmc = sys.argv[6].lower() == "true" if len(sys.argv) > 6 else False
    
    run_training(job_id, dataset_path, num_steps, output_name, output_dir, use_mcmc=use_mcmc)
