#!/usr/bin/env python3
"""
fVDB Gaussian Splat Viewer Service
Native Vulkan-based viewer for Gaussian Splat models
"""

import os
import sys
import time
import logging
from pathlib import Path

# Force NVIDIA Vulkan driver (required for ARM64 with NVIDIA GPU)
os.environ['VK_ICD_FILENAMES'] = '/usr/share/vulkan/icd.d/nvidia_icd.json'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model directory
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/app/models"))
VIEWER_PORT = int(os.environ.get("VIEWER_PORT", "8085"))
VIEWER_HOST = os.environ.get("VIEWER_HOST", "0.0.0.0")

def find_models():
    """Find all PLY models in the model directory"""
    models = list(MODEL_DIR.glob("*.ply"))
    return sorted(models, key=lambda x: x.stat().st_size)  # smallest first

def main():
    import torch
    import fvdb
    import fvdb.viz as viz
    
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    logger.info(f"fVDB: {fvdb.__version__}")
    logger.info(f"Model directory: {MODEL_DIR}")
    
    # Find models
    models = find_models()
    if not models:
        logger.error(f"No PLY models found in {MODEL_DIR}")
        logger.info("Waiting for models...")
        while not models:
            time.sleep(10)
            models = find_models()
    
    # Load smallest model first (for faster startup)
    model_path = models[0]
    logger.info(f"Loading model: {model_path.name} ({model_path.stat().st_size / 1024 / 1024:.1f} MB)")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gsplat, metadata = fvdb.GaussianSplat3d.from_ply(str(model_path), device=device)
    logger.info(f"Loaded {gsplat.num_gaussians} gaussians")
    
    # Initialize viewer
    logger.info(f"Starting viewer on {VIEWER_HOST}:{VIEWER_PORT}")
    viz.init(ip_address=VIEWER_HOST, port=VIEWER_PORT, verbose=False)
    
    # Create scene
    scene = viz.Scene("GaussianSplats")
    scene.add_gaussian_splat_3d(model_path.stem, gsplat)
    
    # Set camera to view the model
    # Use model bounds to position camera
    try:
        means = gsplat.means
        center = means.mean(dim=0).cpu().numpy()
        scale = (means.max(dim=0).values - means.min(dim=0).values).max().item()
        
        scene.set_camera_lookat(
            eye=[center[0], center[1], center[2] + scale * 2],
            center=[center[0], center[1], center[2]],
            up=[0, 1, 0]
        )
        logger.info(f"Camera set: center={center}, scale={scale}")
    except Exception as e:
        logger.warning(f"Could not set camera: {e}")
    
    logger.info(f"Scene created with {model_path.name}")
    
    logger.info(f"\n{'='*50}")
    logger.info(f"🎬 Viewer running at http://{VIEWER_HOST}:{VIEWER_PORT}")
    logger.info(f"{'='*50}\n")
    
    # Keep running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down viewer...")

if __name__ == "__main__":
    main()
