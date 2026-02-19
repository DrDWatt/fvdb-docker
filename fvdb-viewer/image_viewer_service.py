#!/usr/bin/env python3
"""
fVDB Image-based Gaussian Splat Viewer Service
Renders to images instead of video stream (works without Vulkan video encoding)
"""

import os
import io
import time
import logging
import math
import json
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, Query, Form, File, UploadFile
from fastapi.responses import HTMLResponse, Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/app/models"))
VIEWER_PORT = int(os.environ.get("VIEWER_PORT", "8085"))
SAM2_CHECKPOINT = Path(os.environ.get("SAM2_CHECKPOINT", "/app/sam2-models/sam2_hiera_small.pt"))

app = FastAPI(title="fVDB Gaussian Splat Viewer")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Global state
gsplat = None
device = None
model_name = ""
model_metadata = None
available_models = []

# SAM-2 segmentation state
sam2_predictor = None
sam2_loaded = False
current_segments: Dict[str, Any] = {}
segment_labels: Dict[int, str] = {}

# Per-object summaries storage: {segment_idx: {"summary": str, "files": [], "training_data": {}}}
object_summaries: Dict[int, Dict[str, Any]] = {}

# Per-extraction RAG metadata: {job_id: {"text": str, "label": str, "files": [], ...}}
extraction_summaries: Dict[str, Dict[str, Any]] = {}

# RAG metadata from model service
rag_metadata: Dict[str, Any] = {}
rag_labels: List[str] = []
MODEL_SERVICE_URL = os.environ.get("MODEL_SERVICE_URL", "http://rendering-service:8001")

def get_available_models():
    """Get list of available PLY models (excludes extraction temp files)"""
    return sorted([m.name for m in MODEL_DIR.glob("*.ply") if not m.name.startswith('_extraction_')])


def fetch_rag_metadata(model_name: str):
    """Fetch RAG metadata from model service"""
    global rag_metadata, rag_labels
    
    try:
        import requests
        response = requests.get(f"{MODEL_SERVICE_URL}/metadata/{model_name}", timeout=5)
        if response.status_code == 200:
            data = response.json()
            rag_metadata = data.get("metadata", {})
            rag_labels = rag_metadata.get("objects", [])
            logger.info(f"Loaded RAG metadata for {model_name}: {rag_metadata}")
            return True
    except Exception as e:
        logger.warning(f"Could not fetch RAG metadata: {e}")
    
    rag_metadata = {}
    rag_labels = []
    return False


def load_sam2():
    """Load SAM-2 model for segmentation"""
    global sam2_predictor, sam2_loaded, device
    
    if sam2_loaded:
        return True
    
    try:
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Find checkpoint
        checkpoint_paths = [
            SAM2_CHECKPOINT,
            Path("/app/sam2-models/sam2_hiera_small.pt"),
            Path("/app/models/sam2_hiera_small.pt"),
        ]
        
        checkpoint_path = None
        for p in checkpoint_paths:
            if p.exists():
                checkpoint_path = p
                break
        
        if checkpoint_path is None:
            logger.error("SAM-2 checkpoint not found in any location")
            return False
        
        # Use simple config name (hydra finds it automatically)
        config_name = "sam2_hiera_s.yaml"
        
        logger.info(f"Loading SAM-2 from {checkpoint_path} with config {config_name}...")
        sam2_model = build_sam2(config_name, str(checkpoint_path), device=device)
        sam2_predictor = SAM2ImagePredictor(sam2_model)
        sam2_loaded = True
        logger.info("SAM-2 loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load SAM-2: {e}")
        import traceback
        traceback.print_exc()
        return False


# HuggingFace CLIP for object classification
clip_model = None
clip_processor = None
clip_loaded = False

# Object labels for CLIP classification
OBJECT_LABELS = [
    # Vehicles
    "car", "sedan", "sports car", "electric car", "Tesla", "automobile", "vehicle",
    "SUV", "truck", "pickup truck", "van", "motorcycle", "bicycle",
    # Construction vehicles
    "bulldozer", "toy bulldozer", "construction vehicle", "excavator", "toy truck",
    # Plants
    "plant", "potted plant", "houseplant", "leaves", "foliage", "tree",
    # Furniture
    "table", "wooden table", "desk", "countertop", "chair", "furniture", "shelf", "couch", "sofa",
    # Kitchen items
    "bowl", "cup", "mug", "plate", "dish", "bottle", "spray bottle", "container", "jar",
    # Other objects
    "book", "box", "package", "toy", "figurine", "decoration", "ornament",
    # Environment
    "floor", "wall", "background", "carpet", "ceiling", "window", "door",
    "road", "pavement", "driveway", "garage", "parking lot"
]

def load_clip_model():
    """Load HuggingFace CLIP model for object classification"""
    global clip_model, clip_processor, clip_loaded
    
    if clip_loaded:
        return True
    
    try:
        from transformers import CLIPProcessor, CLIPModel
        
        logger.info("Loading HuggingFace CLIP model (openai/clip-vit-base-patch32)...")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        clip_loaded = True
        logger.info("CLIP model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to load CLIP: {e}")
        import traceback
        traceback.print_exc()
        return False


def auto_label_segment(image: np.ndarray, mask: np.ndarray) -> str:
    """Classify segment using HuggingFace CLIP model with RAG enhancement"""
    import torch
    from PIL import Image
    
    # Load CLIP if not loaded
    if not clip_loaded:
        if not load_clip_model():
            return "Object"
    
    try:
        # Convert mask to boolean
        bool_mask = mask.astype(bool) if mask.dtype != bool else mask
        
        # Get bounding box
        coords = np.where(bool_mask)
        if len(coords[0]) == 0:
            return "Object"
        
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        
        # Add padding
        pad = 15
        y_min = max(0, y_min - pad)
        x_min = max(0, x_min - pad)
        y_max = min(image.shape[0], y_max + pad)
        x_max = min(image.shape[1], x_max + pad)
        
        # Crop region
        crop = image[y_min:y_max, x_min:x_max]
        pil_img = Image.fromarray(crop)
        
        # Combine RAG labels with default labels (RAG labels first for priority)
        labels_to_use = rag_labels + OBJECT_LABELS if rag_labels else OBJECT_LABELS
        # Remove duplicates while preserving order
        seen = set()
        unique_labels = []
        for l in labels_to_use:
            if l.lower() not in seen:
                seen.add(l.lower())
                unique_labels.append(l)
        
        # Process with CLIP
        inputs = clip_processor(
            text=unique_labels,
            images=pil_img,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = clip_model(**inputs)
            logits = outputs.logits_per_image
            probs = logits.softmax(dim=1)
            best_idx = probs[0].argmax().item()
            confidence = probs[0][best_idx].item()
        
        label = unique_labels[best_idx]
        
        # If we have RAG metadata title and this is a main object, use enhanced label
        if rag_metadata.get("title") and confidence > 0.3:
            # Check if label matches any RAG object
            for rag_obj in rag_labels:
                if rag_obj.lower() in label.lower() or label.lower() in rag_obj.lower():
                    label = rag_metadata.get("title", label)
                    break
        
        logger.info(f"CLIP classified as '{label}' (confidence {confidence:.2f}, RAG labels: {len(rag_labels)})")
        return label.title()
        
    except Exception as e:
        logger.error(f"CLIP classification failed: {e}")
        return "Object"


def segment_image(image: np.ndarray, points: List[Dict] = None, auto_mode: bool = True):
    """Segment objects in image using SAM-2 Automatic Mask Generator"""
    global current_segments, segment_labels
    
    if not sam2_loaded:
        if not load_sam2():
            return None
    
    try:
        import torch
        
        masks_data = []
        scores_data = []
        
        if auto_mode:
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            # Use SAM2 Automatic Mask Generator for proper object detection
            mask_generator = SAM2AutomaticMaskGenerator(
                model=sam2_predictor.model,
                points_per_side=32,
                points_per_batch=64,
                pred_iou_thresh=0.7,
                stability_score_thresh=0.92,
                stability_score_offset=1.0,
                crop_n_layers=1,
                box_nms_thresh=0.7,
                min_mask_region_area=100,
            )
            
            logger.info("Running SAM-2 automatic mask generation on GPU...")
            masks = mask_generator.generate(image)
            logger.info(f"Generated {len(masks)} masks")
            
            # Sort by area (largest first) and filter
            masks = sorted(masks, key=lambda x: x['area'], reverse=True)
            
            for mask_data in masks:
                # Skip very large masks (likely background)
                img_area = image.shape[0] * image.shape[1]
                if mask_data['area'] > img_area * 0.5:
                    continue
                # Skip tiny masks
                if mask_data['area'] < img_area * 0.001:
                    continue
                    
                masks_data.append(mask_data['segmentation'])
                scores_data.append(float(mask_data['stability_score']))
        
        elif points:
            # Point-based segmentation - set image first
            sam2_predictor.set_image(image)
            point_coords = np.array([[p['x'], p['y']] for p in points])
            point_labels = np.array([p.get('label', 1) for p in points])
            
            masks, scores, _ = sam2_predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True
            )
            
            # Take best mask
            best_idx = np.argmax(scores)
            masks_data.append(masks[best_idx])
            scores_data.append(float(scores[best_idx]))
        
        # Auto-label each segment
        labels = {}
        for i, mask in enumerate(masks_data):
            labels[i] = auto_label_segment(image, mask)
        
        result = {
            "masks": masks_data,
            "scores": scores_data,
            "num_segments": len(masks_data),
            "labels": labels
        }
        
        # Only set global state for auto_mode (not point-based)
        if auto_mode:
            current_segments = result
            segment_labels = labels
        
        return result
        
    except Exception as e:
        logger.error(f"Segmentation failed: {e}")
        return None


def create_overlay_with_labels(image: np.ndarray, segments: Dict, labels: Dict[int, str] = None):
    """Create image with segmentation overlay and labels"""
    import cv2
    from PIL import Image, ImageDraw, ImageFont
    
    overlay = image.copy()
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), 
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (255, 128, 0), (128, 0, 255), (0, 255, 128)
    ]
    
    if "masks" not in segments:
        return overlay
    
    label_positions = []
    
    for i, mask in enumerate(segments["masks"]):
        color = colors[i % len(colors)]
        
        # Create colored mask overlay
        colored_mask = np.zeros_like(overlay)
        colored_mask[mask > 0] = color
        overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 0.4, 0)
        
        # Draw contour
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 2)
        
        # Find centroid for label
        if len(contours) > 0:
            M = cv2.moments(contours[0])
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                label_positions.append((i, cx, cy, color))
    
    # Convert to PIL for text drawing
    pil_img = Image.fromarray(overlay)
    draw = ImageDraw.Draw(pil_img)
    
    # Draw labels
    for idx, cx, cy, color in label_positions:
        label = labels.get(idx, f"Object {idx}") if labels else f"Object {idx}"
        
        # Draw label background
        bbox = draw.textbbox((cx, cy), label)
        padding = 4
        draw.rectangle(
            [bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding],
            fill=(0, 0, 0, 180)
        )
        draw.text((cx, cy), label, fill=color)
    
    return np.array(pil_img)

def load_model(model_file=None):
    global gsplat, device, model_name, model_metadata, available_models
    import torch
    import fvdb
    
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
    
    available_models = get_available_models()
    if not available_models:
        logger.error("No models found!")
        return False
    
    # Use specified model or first available
    if model_file and model_file in available_models:
        model_path = MODEL_DIR / model_file
    else:
        model_path = MODEL_DIR / available_models[0]
    
    model_name = model_path.stem
    logger.info(f"Loading {model_path.name}...")
    
    gsplat_obj, metadata = fvdb.GaussianSplat3d.from_ply(str(model_path), device=device)
    gsplat = gsplat_obj
    model_metadata = metadata
    logger.info(f"Loaded {gsplat.num_gaussians} gaussians")
    
    # Fetch RAG metadata for this model
    fetch_rag_metadata(model_path.name)
    
    return True


def render_view(width=800, height=600, azimuth=0, elevation=0, zoom=1.0, cam_idx=0):
    """Render using cameras from PLY metadata with orbit and zoom"""
    import torch
    
    if gsplat is None or model_metadata is None:
        return None
    
    try:
        c2w_all = model_metadata.get('camera_to_world_matrices')
        K_all = model_metadata.get('projection_matrices')
        sizes = model_metadata.get('image_sizes')
        
        if c2w_all is None or K_all is None:
            logger.error("No camera matrices in metadata")
            return None
        
        num_cams = c2w_all.shape[0]
        cam_idx = cam_idx % num_cams
        
        c2w = c2w_all[cam_idx].to(device).clone()
        means = gsplat.means
        center = means.mean(dim=0)
        
        # Get camera position and direction
        cam_pos = c2w[:3, 3].clone()
        
        # Apply zoom - move camera along view direction
        view_dir = center - cam_pos
        view_dir = view_dir / view_dir.norm()
        
        # zoom > 1 = closer, zoom < 1 = farther
        dist_to_center = (center - cam_pos).norm().item()
        new_dist = dist_to_center / zoom
        new_pos = center - view_dir * new_dist
        c2w[:3, 3] = new_pos
        
        # Apply rotation around model center
        if azimuth != 0 or elevation != 0:
            az_rad = math.radians(azimuth)
            el_rad = math.radians(elevation)
            
            # Rotation around Y (azimuth)
            Ry = torch.tensor([
                [math.cos(az_rad), 0, math.sin(az_rad), 0],
                [0, 1, 0, 0],
                [-math.sin(az_rad), 0, math.cos(az_rad), 0],
                [0, 0, 0, 1]
            ], device=device, dtype=torch.float32)
            
            # Rotation around X (elevation)
            Rx = torch.tensor([
                [1, 0, 0, 0],
                [0, math.cos(el_rad), -math.sin(el_rad), 0],
                [0, math.sin(el_rad), math.cos(el_rad), 0],
                [0, 0, 0, 1]
            ], device=device, dtype=torch.float32)
            
            # Translate to center, rotate, translate back
            T_to = torch.eye(4, device=device, dtype=torch.float32)
            T_to[:3, 3] = -center
            T_back = torch.eye(4, device=device, dtype=torch.float32)
            T_back[:3, 3] = center
            
            R_orbit = T_back @ Ry @ Rx @ T_to
            c2w = R_orbit @ c2w
        
        c2w = c2w.unsqueeze(0).contiguous()
        w2c = torch.inverse(c2w).contiguous()
        
        # Scale projection matrix
        orig_h, orig_w = sizes[cam_idx].tolist()
        K = K_all[cam_idx:cam_idx+1].to(device).clone()
        K[:, 0, :] *= width / orig_w
        K[:, 1, :] *= height / orig_h
        K = K.contiguous()
        
        images, alpha = gsplat.render_images(
            world_to_camera_matrices=w2c,
            projection_matrices=K,
            image_width=width,
            image_height=height,
            near=0.01,
            far=100.0
        )
        
        img = images[0].clamp(0, 1).cpu().numpy()
        if img.shape[-1] > 3:
            img = img[..., :3]
        img = (img * 255).astype(np.uint8)
        
        return img
        
    except Exception as e:
        logger.error(f"Render error: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_camera_matrices(width, height, azimuth, elevation, zoom, cam_idx):
    """Compute world-to-camera and projection matrices matching render_view.
    
    Returns (w2c, K) tensors on the GPU device, or (None, None) if unavailable.
    w2c shape: [1, 4, 4], K shape: [1, 3, 3] (or similar from metadata).
    """
    import torch
    
    if gsplat is None or model_metadata is None:
        return None, None
    
    try:
        c2w_all = model_metadata.get('camera_to_world_matrices')
        K_all = model_metadata.get('projection_matrices')
        sizes = model_metadata.get('image_sizes')
        
        if c2w_all is None or K_all is None:
            return None, None
        
        num_cams = c2w_all.shape[0]
        cam_idx = cam_idx % num_cams
        
        c2w = c2w_all[cam_idx].to(device).clone()
        means = gsplat.means
        center = means.mean(dim=0)
        
        # Apply zoom
        cam_pos = c2w[:3, 3].clone()
        view_dir = center - cam_pos
        view_dir = view_dir / view_dir.norm()
        dist_to_center = (center - cam_pos).norm().item()
        new_dist = dist_to_center / zoom
        new_pos = center - view_dir * new_dist
        c2w[:3, 3] = new_pos
        
        # Apply rotation around model center
        if azimuth != 0 or elevation != 0:
            az_rad = math.radians(azimuth)
            el_rad = math.radians(elevation)
            
            Ry = torch.tensor([
                [math.cos(az_rad), 0, math.sin(az_rad), 0],
                [0, 1, 0, 0],
                [-math.sin(az_rad), 0, math.cos(az_rad), 0],
                [0, 0, 0, 1]
            ], device=device, dtype=torch.float32)
            
            Rx = torch.tensor([
                [1, 0, 0, 0],
                [0, math.cos(el_rad), -math.sin(el_rad), 0],
                [0, math.sin(el_rad), math.cos(el_rad), 0],
                [0, 0, 0, 1]
            ], device=device, dtype=torch.float32)
            
            T_to = torch.eye(4, device=device, dtype=torch.float32)
            T_to[:3, 3] = -center
            T_back = torch.eye(4, device=device, dtype=torch.float32)
            T_back[:3, 3] = center
            
            R_orbit = T_back @ Ry @ Rx @ T_to
            c2w = R_orbit @ c2w
        
        c2w = c2w.unsqueeze(0).contiguous()
        w2c = torch.inverse(c2w).contiguous()
        
        # Scale projection matrix to render resolution
        orig_h, orig_w = sizes[cam_idx].tolist()
        K = K_all[cam_idx:cam_idx+1].to(device).clone()
        K[:, 0, :] *= width / orig_w
        K[:, 1, :] *= height / orig_h
        K = K.contiguous()
        
        return w2c, K
        
    except Exception as e:
        logger.error(f"get_camera_matrices error: {e}")
        return None, None


@app.on_event("startup")
async def startup():
    load_model()
    # Preload CLIP model for instant classification
    logger.info("Preloading CLIP model...")
    load_clip_model()
    logger.info("Startup complete - CLIP ready")

@app.get("/", response_class=HTMLResponse)
async def root():
    models = list(MODEL_DIR.glob("*.ply"))
    model_list = ", ".join([m.name for m in models[:5]])
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>fVDB Gaussian Splat Viewer</title>
        <style>
            body {{ 
                margin: 0; 
                background: #1a1a2e; 
                color: white; 
                font-family: Arial, sans-serif;
                overflow: hidden;
            }}
            #viewer {{ 
                width: 100vw; 
                height: 100vh; 
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
            }}
            #render {{ 
                max-width: 90vw; 
                max-height: 80vh;
                border: 2px solid #76b900;
                border-radius: 8px;
            }}
            #controls {{
                position: fixed;
                top: 10px;
                left: 10px;
                background: rgba(0,0,0,0.8);
                padding: 15px;
                border-radius: 8px;
            }}
            #info {{
                position: fixed;
                top: 10px;
                right: 10px;
                background: rgba(0,0,0,0.8);
                padding: 15px;
                border-radius: 8px;
            }}
            #segmentation {{
                position: fixed;
                bottom: 10px;
                left: 10px;
                background: rgba(0,0,0,0.9);
                padding: 15px;
                border-radius: 8px;
                max-width: 350px;
            }}
            #rag-pane {{
                position: fixed;
                bottom: 10px;
                left: 50%;
                transform: translateX(-50%);
                background: rgba(0,0,0,0.9);
                padding: 12px 20px;
                border-radius: 8px;
                border: 1px solid #ffc107;
                z-index: 100;
                display: flex;
                gap: 10px;
                align-items: center;
            }}
            #rag-pane h2 {{ color: #ffc107; margin: 0; font-size: 14px; white-space: nowrap; }}
            .rag-btn {{
                padding: 8px 15px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-weight: bold;
                font-size: 13px;
            }}
            #garfield {{
                position: fixed;
                bottom: 10px;
                right: 10px;
                background: rgba(0,0,0,0.9);
                padding: 15px;
                border-radius: 8px;
                max-width: 320px;
                border: 1px solid #ff6b6b;
            }}
            #garfield h2 {{ color: #ff6b6b; margin: 0 0 10px 0; font-size: 16px; }}
            .garfield-btn {{
                padding: 8px 15px;
                margin: 3px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-weight: bold;
            }}
            .garfield-btn-primary {{ background: #ff6b6b; color: white; }}
            .garfield-btn-success {{ background: #ffc107; color: #000; }}
            #extraction-list {{
                max-height: 120px;
                overflow-y: auto;
                margin-top: 10px;
            }}
            .extraction-item {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 6px 10px;
                margin: 4px 0;
                background: rgba(255,107,107,0.15);
                border-radius: 4px;
                font-size: 12px;
            }}
            .scale-slider {{
                width: 100%;
                margin: 8px 0;
            }}
            #segmentation h2 {{ color: #00d4ff; margin: 0 0 10px 0; font-size: 16px; }}
            .seg-btn {{
                padding: 8px 15px;
                margin: 3px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-weight: bold;
            }}
            .seg-btn-primary {{ background: #00d4ff; color: #000; }}
            .seg-btn-success {{ background: #28a745; color: white; }}
            .seg-btn-danger {{ background: #dc3545; color: white; }}
            .seg-btn:disabled {{ opacity: 0.5; cursor: not-allowed; }}
            #labels-list {{
                max-height: 150px;
                overflow-y: auto;
                margin-top: 10px;
            }}
            #hover-tooltip {{
                position: fixed;
                background: rgba(0,0,0,0.85);
                color: white;
                padding: 8px 14px;
                border-radius: 6px;
                font-size: 14px;
                font-weight: bold;
                pointer-events: none;
                display: none;
                z-index: 1000;
                border: 2px solid #00d4ff;
                box-shadow: 0 4px 12px rgba(0,212,255,0.3);
            }}
            /* Object Summary Pane - see-through/transparent */
            #object-summary-pane {{
                position: fixed;
                top: 50%;
                right: 20px;
                transform: translateY(-50%);
                width: 380px;
                max-height: 70vh;
                background: rgba(26, 26, 46, 0.85);
                backdrop-filter: blur(10px);
                -webkit-backdrop-filter: blur(10px);
                border: 2px solid #00d4ff;
                border-radius: 12px;
                padding: 20px;
                display: none;
                z-index: 1500;
                box-shadow: 0 8px 32px rgba(0, 212, 255, 0.2);
                overflow-y: auto;
            }}
            #object-summary-pane.visible {{
                display: block;
            }}
            #object-summary-pane h3 {{
                color: #00d4ff;
                margin: 0 0 15px 0;
                font-size: 18px;
                display: flex;
                align-items: center;
                justify-content: space-between;
            }}
            #object-summary-pane .close-btn {{
                background: transparent;
                border: none;
                color: #ff6b6b;
                font-size: 24px;
                cursor: pointer;
                padding: 0 5px;
            }}
            #object-summary-pane .object-label {{
                display: flex;
                align-items: center;
                gap: 10px;
                margin-bottom: 15px;
                padding: 10px;
                background: rgba(255,255,255,0.1);
                border-radius: 8px;
            }}
            #object-summary-pane .object-color {{
                width: 24px;
                height: 24px;
                border-radius: 4px;
            }}
            #object-summary-pane .summary-section {{
                margin: 15px 0;
            }}
            #object-summary-pane .summary-section h4 {{
                color: #aaa;
                font-size: 12px;
                text-transform: uppercase;
                margin: 0 0 8px 0;
            }}
            #object-summary-pane textarea {{
                width: 100%;
                min-height: 120px;
                background: rgba(0,0,0,0.5);
                border: 1px solid #444;
                border-radius: 6px;
                color: white;
                padding: 10px;
                font-family: inherit;
                resize: vertical;
            }}
            #object-summary-pane .file-upload {{
                margin: 10px 0;
            }}
            #object-summary-pane .file-upload input {{
                color: #aaa;
            }}
            #object-summary-pane .btn-row {{
                display: flex;
                gap: 10px;
                margin-top: 15px;
            }}
            #object-summary-pane .btn {{
                flex: 1;
                padding: 10px 15px;
                border: none;
                border-radius: 6px;
                cursor: pointer;
                font-weight: bold;
                transition: opacity 0.2s;
            }}
            #object-summary-pane .btn:hover {{
                opacity: 0.85;
            }}
            #object-summary-pane .btn-save {{
                background: #28a745;
                color: white;
            }}
            #object-summary-pane .btn-upload {{
                background: #17a2b8;
                color: white;
            }}
            #object-summary-pane .uploaded-files {{
                margin-top: 10px;
                font-size: 12px;
            }}
            #object-summary-pane .uploaded-files li {{
                padding: 5px 0;
                border-bottom: 1px solid rgba(255,255,255,0.1);
            }}
            .object-clickable {{
                cursor: pointer;
            }}
            .label-item {{
                display: flex;
                align-items: center;
                gap: 8px;
                padding: 5px;
                margin: 3px 0;
                background: rgba(255,255,255,0.1);
                border-radius: 4px;
            }}
            .label-color {{
                width: 15px;
                height: 15px;
                border-radius: 3px;
            }}
            .label-input {{
                flex: 1;
                padding: 4px 8px;
                border: 1px solid #444;
                border-radius: 3px;
                background: #222;
                color: white;
            }}
            h1 {{ color: #76b900; margin: 0 0 10px 0; font-size: 18px; }}
            .slider-group {{ margin: 10px 0; }}
            label {{ display: block; margin-bottom: 5px; }}
            input[type="range"] {{ width: 150px; }}
            #loading {{ 
                position: fixed; 
                top: 50%; 
                left: 50%; 
                transform: translate(-50%, -50%);
                font-size: 24px;
            }}
            .click-mode {{ cursor: crosshair !important; }}
        </style>
    </head>
    <body>
        <div id="viewer">
            <div id="loading">Loading...</div>
            <img id="render" style="display:none;" />
        </div>
        
        <div id="controls">
            <h1>🎬 fVDB Viewer</h1>
            <div class="slider-group">
                <label>Model:</label>
                <select id="model-select" style="width:150px;padding:5px;">
                    <option>Loading...</option>
                </select>
                <button onclick="refreshModels()" style="margin-left:5px;padding:5px 8px;background:#28a745;color:white;border:none;border-radius:4px;cursor:pointer;" title="Refresh model list">🔄</button>
                <button onclick="deleteCurrentModel()" style="margin-left:10px;padding:5px 10px;background:#dc3545;color:white;border:none;border-radius:4px;cursor:pointer;">🗑️ Delete</button>
            </div>
            <div class="slider-group">
                <label>Rotation: <span id="az-val">0</span>°</label>
                <input type="range" id="azimuth" min="-180" max="180" value="0">
            </div>
            <div class="slider-group">
                <label>Elevation: <span id="el-val">0</span>°</label>
                <input type="range" id="elevation" min="-45" max="45" value="0">
            </div>
            <div class="slider-group">
                <label>Zoom: <span id="zoom-val">1</span>x</label>
                <input type="range" id="zoom" min="0.1" max="2" value="1" step="0.1">
            </div>
            <div class="slider-group">
                <label>Camera: <span id="cam-val">0</span></label>
                <input type="range" id="camera" min="0" max="14" value="0" step="1">
            </div>
        </div>
        
        <div id="info">
            <p><strong>Models:</strong> {model_list}</p>
            <p><strong>Gaussians:</strong> <span id="num-gs">Loading...</span></p>
            <p>Drag sliders to rotate view</p>
        </div>
        
        <div id="segmentation">
            <h2>🎯 SAM-2 Segmentation</h2>
            <div>
                <button class="seg-btn seg-btn-primary" id="autoSegBtn" onclick="runAutoSegmentation()">Auto Segment</button>
                <button class="seg-btn seg-btn-success" id="clickSegBtn" onclick="toggleClickMode()">Click to Segment</button>
                <button class="seg-btn seg-btn-danger" onclick="clearSegments()">Clear</button>
            </div>
            <div id="segStatus" style="margin-top:10px;font-size:12px;color:#888;">
                Click "Auto Segment" to detect objects<br>
                <span style="color:#00d4ff;">Double-click objects to view/edit summaries</span>
            </div>
            <div id="labels-list"></div>
        </div>
        
        <div id="rag-pane">
            <h2>📄 RAG Query</h2>
            <button class="rag-btn" style="background:#ffc107;color:#000;" onclick="showSummary()">Ask AI</button>
            <button class="rag-btn" style="background:#17a2b8;color:white;" onclick="showUploadModal()">⬆️ Upload Info</button>
        </div>
        
        <div id="hover-tooltip"></div>
        
        <!-- Object Summary Pane - see-through for per-object summaries -->
        <div id="object-summary-pane">
            <h3>
                <span>📋 Object Summary</span>
                <button class="close-btn" onclick="closeObjectSummary()">&times;</button>
            </h3>
            <div class="object-label">
                <div class="object-color" id="obj-summary-color"></div>
                <input type="text" id="obj-summary-name" class="label-input" placeholder="Object name..." style="flex:1;">
            </div>
            <div class="summary-section">
                <h4>Description / Notes</h4>
                <textarea id="obj-summary-text" placeholder="Enter summary, notes, or description for this object..."></textarea>
            </div>
            <div class="summary-section">
                <h4>Upload Documents</h4>
                <div class="file-upload">
                    <input type="file" id="obj-summary-file" accept=".pdf,.txt,.json,.md,.doc,.docx,.png,.jpg" multiple>
                </div>
                <ul class="uploaded-files" id="obj-uploaded-files"></ul>
            </div>
            <div class="summary-section">
                <h4>Training Data</h4>
                <div style="font-size:12px;color:#888;" id="obj-training-status">No training data associated</div>
            </div>
            <div class="btn-row">
                <button class="btn btn-save" onclick="saveObjectSummary()">💾 Save</button>
                <button class="btn btn-upload" onclick="uploadObjectFiles()">⬆️ Upload Files</button>
            </div>
        </div>
        
        <!-- GARField 3D Asset Extraction Pane -->
        <div id="garfield">
            <h2>🎯 GARField 3D Extraction</h2>
            <div>
                <button class="garfield-btn garfield-btn-primary" id="extractBtn" onclick="toggleExtractMode()">Click to Extract</button>
                <button class="garfield-btn" style="background:#6c757d;color:white;" onclick="clearExtractions()">Clear</button>
            </div>
            <div style="margin-top:8px;">
                <label style="font-size:12px;">Scale Level: <span id="scale-val">0.5</span></label>
                <input type="range" id="extract-scale" class="scale-slider" min="0.1" max="2.0" value="0.5" step="0.1" oninput="document.getElementById('scale-val').textContent = this.value">
            </div>
            <div id="extractStatus" style="margin-top:8px;font-size:12px;color:#888;">
                Click to extract 3D assets from scene
            </div>
            <div id="extraction-list"></div>
            <div id="extraction-rag" style="margin-top:10px;display:none;">
                <h3 style="color:#ff6b6b;font-size:13px;margin:0 0 6px 0;">📄 Extraction Metadata</h3>
                <input type="text" id="extLabel" placeholder="Label (e.g. Chair, Table)" style="width:100%;padding:5px;background:rgba(0,0,0,0.4);border:1px solid #555;color:#eee;border-radius:4px;margin-bottom:5px;box-sizing:border-box;">
                <textarea id="extSummary" placeholder="Description / notes for this extraction..." style="width:100%;min-height:60px;background:rgba(0,0,0,0.4);border:1px solid #555;color:#eee;border-radius:4px;padding:5px;resize:vertical;box-sizing:border-box;"></textarea>
                <div style="margin-top:5px;display:flex;gap:5px;align-items:center;">
                    <button class="garfield-btn" style="background:#17a2b8;color:white;font-size:11px;padding:4px 10px;" onclick="saveExtSummary()">Save</button>
                    <label style="font-size:11px;cursor:pointer;color:#17a2b8;">
                        <input type="file" id="extFileUpload" multiple style="display:none;" onchange="uploadExtFiles()"> ⬆️ Upload Docs
                    </label>
                </div>
                <div id="extFileList" style="font-size:11px;color:#aaa;margin-top:4px;"></div>
            </div>
        </div>
        
        <!-- Summary / RAG Query Modal -->
        <div id="summary-modal" style="display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.8);z-index:2000;">
            <div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);background:#1a1a2e;padding:30px;border-radius:12px;width:90%;max-width:700px;max-height:85vh;display:flex;flex-direction:column;border:2px solid #00d4ff;">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
                    <h2 style="color:#00d4ff;margin:0;">📄 Model Summary &amp; RAG Query</h2>
                    <button onclick="closeSummary()" style="background:none;border:none;color:#aaa;font-size:22px;cursor:pointer;padding:0 4px;" title="Close">&times;</button>
                </div>
                <!-- Model context summary -->
                <div id="summary-content" style="color:#eee;line-height:1.6;white-space:pre-wrap;font-size:13px;max-height:120px;overflow-y:auto;padding:10px;background:rgba(0,0,0,0.3);border-radius:6px;margin-bottom:12px;flex-shrink:0;"></div>
                <!-- LLM status -->
                <div id="rag-llm-status" style="font-size:11px;color:#888;margin-bottom:8px;"></div>
                <!-- Chat history -->
                <div id="rag-chat-history" style="flex:1;overflow-y:auto;min-height:150px;max-height:350px;padding:10px;background:rgba(0,0,0,0.25);border-radius:6px;margin-bottom:12px;"></div>
                <!-- Query input -->
                <div style="display:flex;gap:8px;flex-shrink:0;">
                    <input type="text" id="rag-query-input" placeholder="Ask about the scene (e.g. 'What objects are in the splat?')" 
                           style="flex:1;padding:10px 14px;background:rgba(0,0,0,0.4);border:1px solid #555;color:#eee;border-radius:6px;font-size:14px;"
                           onkeydown="if(event.key==='Enter') sendRagQuery()">
                    <button id="rag-send-btn" onclick="sendRagQuery()" style="padding:10px 20px;background:#00d4ff;color:#1a1a2e;border:none;border-radius:6px;cursor:pointer;font-weight:bold;font-size:14px;white-space:nowrap;">Ask</button>
                </div>
            </div>
        </div>
        
        <!-- Upload Modal -->
        <div id="upload-modal" style="display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.8);z-index:2000;">
            <div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);background:#1a1a2e;padding:30px;border-radius:12px;max-width:500px;border:2px solid #17a2b8;">
                <h2 style="color:#17a2b8;margin-top:0;">⬆️ Upload Model Info</h2>
                <p style="color:#aaa;">Upload a summary document (PDF, TXT, JSON, MD) for this model.</p>
                <input type="file" id="summary-file" accept=".pdf,.txt,.json,.md,.text" style="margin:15px 0;color:#eee;">
                <div style="margin-top:15px;">
                    <button onclick="uploadSummary()" style="padding:10px 25px;background:#17a2b8;color:white;border:none;border-radius:5px;cursor:pointer;margin-right:10px;">Upload</button>
                    <button onclick="closeUploadModal()" style="padding:10px 25px;background:#6c757d;color:white;border:none;border-radius:5px;cursor:pointer;">Cancel</button>
                </div>
                <div id="upload-status" style="margin-top:15px;color:#28a745;"></div>
            </div>
        </div>
        
        <script>
            const img = document.getElementById('render');
            const loading = document.getElementById('loading');
            const azSlider = document.getElementById('azimuth');
            const elSlider = document.getElementById('elevation');
            const zoomSlider = document.getElementById('zoom');
            const camSlider = document.getElementById('camera');
            
            let debounceTimer;
            
            async function updateRender() {{
                const az = azSlider.value;
                const el = elSlider.value;
                const zoom = zoomSlider.value;
                const cam = camSlider.value;
                
                document.getElementById('az-val').textContent = az;
                document.getElementById('el-val').textContent = el;
                document.getElementById('zoom-val').textContent = zoom;
                document.getElementById('cam-val').textContent = cam;
                
                loading.style.display = 'block';
                
                try {{
                    const response = await fetch(`/render?azimuth=${{az}}&elevation=${{el}}&zoom=${{zoom}}&cam_idx=${{cam}}&width=1024&height=768`);
                    if (response.ok) {{
                        const blob = await response.blob();
                        img.src = URL.createObjectURL(blob);
                        img.style.display = 'block';
                        loading.style.display = 'none';
                    }}
                }} catch(e) {{
                    console.error('Render error:', e);
                }}
            }}
            
            function debouncedUpdate() {{
                clearTimeout(debounceTimer);
                debounceTimer = setTimeout(updateRender, 150);
            }}
            
            azSlider.addEventListener('input', debouncedUpdate);
            elSlider.addEventListener('input', debouncedUpdate);
            zoomSlider.addEventListener('input', debouncedUpdate);
            camSlider.addEventListener('input', debouncedUpdate);
            
            // Model selection
            const modelSelect = document.getElementById('model-select');
            modelSelect.addEventListener('change', async () => {{
                loading.textContent = 'Loading model...';
                loading.style.display = 'block';
                img.style.display = 'none';
                
                const response = await fetch('/load_model?model=' + modelSelect.value);
                if (response.ok) {{
                    // Reset controls
                    azSlider.value = 0;
                    elSlider.value = 0;
                    zoomSlider.value = 1;
                    camSlider.value = 0;
                    document.getElementById('az-val').textContent = '0';
                    document.getElementById('el-val').textContent = '0';
                    document.getElementById('zoom-val').textContent = '1';
                    document.getElementById('cam-val').textContent = '0';
                    
                    // Clear segmentation state
                    showSegments = false;
                    segmentMasks = [];
                    segmentLabelMap = {{}};
                    document.getElementById('segStatus').textContent = 'Click "Auto Segment" to detect objects';
                    document.getElementById('labels-list').innerHTML = '';
                    await fetch('/segment/clear', {{ method: 'POST' }}).catch(() => {{}});
                    
                    // Clear extraction state
                    if (typeof viewingExtraction !== 'undefined') viewingExtraction = null;
                    if (typeof extractions !== 'undefined') extractions = [];
                    if (typeof extractMode !== 'undefined') extractMode = false;
                    const extList = document.getElementById('extraction-list');
                    if (extList) extList.innerHTML = '';
                    const extStatus = document.getElementById('extractStatus');
                    if (extStatus) extStatus.textContent = 'Click to extract 3D assets from scene';
                    await fetch('/garfield/clear', {{ method: 'POST' }}).catch(() => {{}});
                    
                    // Update info and render
                    const info = await fetch('/info').then(r => r.json());
                    document.getElementById('num-gs').textContent = info.num_gaussians || 'N/A';
                    updateRender();
                }} else {{
                    loading.textContent = 'Failed to load model';
                }}
            }});
            
            // Delete current model
            async function deleteCurrentModel() {{
                const model = modelSelect.value;
                if (!confirm('Delete ' + model + '? This cannot be undone.')) return;
                
                try {{
                    const response = await fetch('/delete_model?model=' + model, {{ method: 'DELETE' }});
                    if (response.ok) {{
                        alert('Deleted ' + model);
                        location.reload();
                    }} else {{
                        alert('Failed to delete: ' + (await response.text()));
                    }}
                }} catch(e) {{
                    alert('Error: ' + e);
                }}
            }}
            
            // Refresh models list from server
            async function refreshModels() {{
                const response = await fetch('/info');
                const data = await response.json();
                const select = document.getElementById('model-select');
                const currentModel = data.model_name + '.ply';
                
                // Clear and repopulate dropdown
                select.innerHTML = '';
                (data.models_available || []).forEach(m => {{
                    const opt = document.createElement('option');
                    opt.value = m;
                    opt.textContent = m;
                    if (m === currentModel || m.replace('.ply','') === data.model_name) {{
                        opt.selected = true;
                    }}
                    select.appendChild(opt);
                }});
                
                document.getElementById('num-gs').textContent = data.num_gaussians || 'N/A';
                
                // Update info panel
                const infoModels = document.querySelector('#info p strong');
                if (infoModels && infoModels.textContent === 'Models:') {{
                    infoModels.parentElement.innerHTML = '<strong>Models:</strong> ' + (data.models_available || []).join(', ');
                }}
            }}
            
            // Initial load
            refreshModels().then(() => updateRender());
            
            // ==================== Segmentation Functions ====================
            let clickMode = false;
            let showSegments = false;
            const segColors = ['#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ff00ff', '#00ffff', '#ff8000', '#8000ff', '#00ff80'];
            
            async function runAutoSegmentation() {{
                const btn = document.getElementById('autoSegBtn');
                const status = document.getElementById('segStatus');
                btn.disabled = true;
                status.textContent = 'Loading SAM-2 and segmenting...';
                
                const formData = new FormData();
                formData.append('azimuth', azSlider.value);
                formData.append('elevation', elSlider.value);
                formData.append('zoom', zoomSlider.value);
                formData.append('cam_idx', camSlider.value);
                
                try {{
                    const response = await fetch('/segment', {{ method: 'POST', body: formData }});
                    const data = await response.json();
                    
                    if (data.status === 'ok') {{
                        status.textContent = `Found ${{data.num_segments}} objects`;
                        showSegments = true;
                        updateLabelsUI(data.labels);
                        renderWithSegments();
                    }} else {{
                        status.textContent = 'Error: ' + (data.error || 'Unknown');
                    }}
                }} catch(e) {{
                    status.textContent = 'Error: ' + e.message;
                }}
                btn.disabled = false;
            }}
            
            function toggleClickMode() {{
                clickMode = !clickMode;
                const btn = document.getElementById('clickSegBtn');
                const status = document.getElementById('segStatus');
                
                if (clickMode) {{
                    btn.textContent = 'Stop Clicking';
                    btn.style.background = '#ffc107';
                    img.classList.add('click-mode');
                    status.textContent = 'Click on objects to segment them';
                }} else {{
                    btn.textContent = 'Click to Segment';
                    btn.style.background = '#28a745';
                    img.classList.remove('click-mode');
                    status.textContent = showSegments ? 'Segments active' : 'Ready';
                }}
            }}
            
            img.addEventListener('click', async (e) => {{
                if (!clickMode) return;
                
                const rect = img.getBoundingClientRect();
                const scaleX = 1024 / rect.width;
                const scaleY = 768 / rect.height;
                const x = Math.round((e.clientX - rect.left) * scaleX);
                const y = Math.round((e.clientY - rect.top) * scaleY);
                
                const status = document.getElementById('segStatus');
                status.textContent = `Segmenting at (${{x}}, ${{y}})...`;
                
                const formData = new FormData();
                formData.append('x', x);
                formData.append('y', y);
                formData.append('label', 1);
                formData.append('azimuth', azSlider.value);
                formData.append('elevation', elSlider.value);
                formData.append('zoom', zoomSlider.value);
                formData.append('cam_idx', camSlider.value);
                
                try {{
                    const response = await fetch('/segment/point', {{ method: 'POST', body: formData }});
                    const data = await response.json();
                    
                    if (data.status === 'ok') {{
                        status.textContent = `Segment ${{data.new_segment_idx}} added`;
                        showSegments = true;
                        refreshLabels();
                        renderWithSegments();
                    }}
                }} catch(e) {{
                    status.textContent = 'Error: ' + e.message;
                }}
            }});
            
            async function renderWithSegments() {{
                const az = azSlider.value;
                const el = elSlider.value;
                const zoom = zoomSlider.value;
                const cam = camSlider.value;
                
                loading.style.display = 'block';
                try {{
                    const response = await fetch(`/render_with_segments?azimuth=${{az}}&elevation=${{el}}&zoom=${{zoom}}&cam_idx=${{cam}}&width=1024&height=768`);
                    if (response.ok) {{
                        const blob = await response.blob();
                        img.src = URL.createObjectURL(blob);
                    }} else {{
                        // Segment render failed, fall back to normal render
                        console.warn('Segment render failed, falling back to normal render');
                        showSegments = false;
                        const fallback = await fetch(`/render?azimuth=${{az}}&elevation=${{el}}&zoom=${{zoom}}&cam_idx=${{cam}}&width=1024&height=768`);
                        if (fallback.ok) {{
                            const blob = await fallback.blob();
                            img.src = URL.createObjectURL(blob);
                        }}
                    }}
                }} catch(e) {{
                    console.error('Render error:', e);
                    showSegments = false;
                }} finally {{
                    img.style.display = 'block';
                    loading.style.display = 'none';
                }}
            }}
            
            async function clearSegments() {{
                await fetch('/segment/clear', {{ method: 'POST' }});
                showSegments = false;
                segmentMasks = [];
                segmentLabelMap = {{}};
                document.getElementById('segStatus').textContent = 'Segments cleared';
                document.getElementById('labels-list').innerHTML = '';
                // Force regular render without segments
                const az = azSlider.value;
                const el = elSlider.value;
                const zoom = zoomSlider.value;
                const cam = camSlider.value;
                loading.style.display = 'block';
                const response = await fetch(`/render?azimuth=${{az}}&elevation=${{el}}&zoom=${{zoom}}&cam_idx=${{cam}}&width=1024&height=768`);
                if (response.ok) {{
                    const blob = await response.blob();
                    img.src = URL.createObjectURL(blob);
                    img.style.display = 'block';
                    loading.style.display = 'none';
                }}
            }}
            
            function updateLabelsUI(labels) {{
                const list = document.getElementById('labels-list');
                list.innerHTML = '';
                
                Object.entries(labels).forEach(([idx, label]) => {{
                    const color = segColors[parseInt(idx) % segColors.length];
                    const item = document.createElement('div');
                    item.className = 'label-item';
                    item.innerHTML = `
                        <div class="label-color" style="background:${{color}}"></div>
                        <input type="text" class="label-input" value="${{label}}" 
                               onchange="updateLabel(${{idx}}, this.value)" 
                               placeholder="Enter label...">
                    `;
                    list.appendChild(item);
                }});
            }}
            
            async function refreshLabels() {{
                const response = await fetch('/segment/labels');
                const data = await response.json();
                updateLabelsUI(data.labels);
            }}
            
            async function updateLabel(idx, label) {{
                const formData = new FormData();
                formData.append('segment_idx', idx);
                formData.append('label', label);
                await fetch('/segment/label', {{ method: 'POST', body: formData }});
                renderWithSegments();
            }}
            
            // Override render update to use segments if active
            const originalUpdate = updateRender;
            updateRender = function() {{
                if (showSegments) {{
                    renderWithSegments();
                }} else {{
                    originalUpdate();
                }}
            }};
            
            // Hover tooltip functionality
            const tooltip = document.getElementById('hover-tooltip');
            let segmentMasks = [];
            let segmentLabelMap = {{}};
            
            let maskScale = 4;
            
            img.addEventListener('mousemove', async (e) => {{
                if (!showSegments || segmentMasks.length === 0) {{
                    tooltip.style.display = 'none';
                    return;
                }}
                
                const rect = img.getBoundingClientRect();
                const scaleX = 1024 / rect.width;
                const scaleY = 768 / rect.height;
                const imgX = Math.round((e.clientX - rect.left) * scaleX);
                const imgY = Math.round((e.clientY - rect.top) * scaleY);
                
                // Account for downsampled masks
                const x = Math.floor(imgX / maskScale);
                const y = Math.floor(imgY / maskScale);
                
                // Check which segment the mouse is over
                let hoveredLabel = null;
                for (let i = 0; i < segmentMasks.length; i++) {{
                    if (segmentMasks[i] && segmentMasks[i][y] && segmentMasks[i][y][x]) {{
                        hoveredLabel = segmentLabelMap[i] || `Object ${{i}}`;
                        break;
                    }}
                }}
                
                if (hoveredLabel) {{
                    tooltip.textContent = hoveredLabel;
                    tooltip.style.display = 'block';
                    tooltip.style.left = (e.clientX + 15) + 'px';
                    tooltip.style.top = (e.clientY + 15) + 'px';
                }} else {{
                    tooltip.style.display = 'none';
                }}
            }});
            
            img.addEventListener('mouseleave', () => {{
                tooltip.style.display = 'none';
            }});
            
            // Fetch segment mask data for hover detection
            async function loadSegmentMasks() {{
                try {{
                    const response = await fetch('/segment/masks');
                    const data = await response.json();
                    segmentMasks = data.masks || [];
                    segmentLabelMap = data.labels || {{}};
                    maskScale = data.scale || 4;
                }} catch(e) {{
                    console.log('Could not load segment masks for hover');
                }}
            }}
            
            // Update loadSegmentMasks after segmentation
            const origUpdateLabelsUI = updateLabelsUI;
            updateLabelsUI = function(labels) {{
                origUpdateLabelsUI(labels);
                segmentLabelMap = labels;
                loadSegmentMasks();
            }};
            
            // Summary, RAG Query, and Upload functions
            let ragChatHistory = [];
            
            async function showSummary() {{
                const modelSelect = document.getElementById('model-select');
                const modelName = modelSelect.value;
                
                document.getElementById('summary-content').textContent = 'Loading context...';
                document.getElementById('rag-chat-history').innerHTML = '';
                document.getElementById('rag-llm-status').innerHTML = '<span style="color:#ffc107;">Checking LLM...</span>';
                document.getElementById('summary-modal').style.display = 'block';
                ragChatHistory = [];
                
                // Load model context summary
                try {{
                    const response = await fetch(`/rag/context?model=${{encodeURIComponent(modelName)}}`);
                    const data = await response.json();
                    
                    let contextHtml = '';
                    if (data.model_summary) {{
                        contextHtml += `<strong>Model:</strong> ${{data.model_name || modelName}}<br>`;
                        contextHtml += data.model_summary + '<br>';
                    }}
                    if (data.segments_count > 0) {{
                        contextHtml += `<strong>Segments:</strong> ${{data.segments_count}} objects detected`;
                        if (data.segment_labels && data.segment_labels.length > 0) {{
                            contextHtml += ` (${{data.segment_labels.join(', ')}})`;
                        }}
                        contextHtml += '<br>';
                    }}
                    if (data.extractions_count > 0) {{
                        contextHtml += `<strong>Extractions:</strong> ${{data.extractions_count}} 3D extractions<br>`;
                    }}
                    if (data.documents_count > 0) {{
                        contextHtml += `<strong>Documents:</strong> ${{data.documents_count}} uploaded<br>`;
                    }}
                    if (!contextHtml) {{
                        contextHtml = '<span style="color:#ffc107;">No context data yet.</span> Segment objects or upload documents to enrich the knowledge base.';
                    }}
                    document.getElementById('summary-content').innerHTML = contextHtml;
                }} catch(e) {{
                    document.getElementById('summary-content').innerHTML = '<span style="color:#ffc107;">No summary available.</span> Upload documents or segment objects to add context.';
                }}
                
                // Check LLM status
                try {{
                    const llmResp = await fetch('/rag/status');
                    const llmData = await llmResp.json();
                    if (llmData.available) {{
                        document.getElementById('rag-llm-status').innerHTML = `<span style="color:#28a745;">● LLM ready</span> <span style="color:#666;">(${{llmData.model}})</span>`;
                    }} else {{
                        document.getElementById('rag-llm-status').innerHTML = `<span style="color:#dc3545;">● LLM unavailable:</span> <span style="color:#888;">${{llmData.error || 'Ollama not reachable'}}</span>`;
                    }}
                }} catch(e) {{
                    document.getElementById('rag-llm-status').innerHTML = '<span style="color:#dc3545;">● LLM unavailable</span>';
                }}
            }}
            
            function appendChatMessage(role, text) {{
                const history = document.getElementById('rag-chat-history');
                const div = document.createElement('div');
                div.style.marginBottom = '10px';
                if (role === 'user') {{
                    div.innerHTML = `<div style="text-align:right;"><span style="background:#00d4ff;color:#1a1a2e;padding:6px 12px;border-radius:12px 12px 2px 12px;display:inline-block;max-width:85%;text-align:left;font-size:13px;">${{text}}</span></div>`;
                }} else if (role === 'assistant') {{
                    div.innerHTML = `<div><span style="background:rgba(255,255,255,0.1);color:#eee;padding:6px 12px;border-radius:12px 12px 12px 2px;display:inline-block;max-width:85%;text-align:left;font-size:13px;white-space:pre-wrap;">${{text}}</span></div>`;
                }} else {{
                    div.innerHTML = `<div style="text-align:center;"><span style="color:#888;font-size:11px;font-style:italic;">${{text}}</span></div>`;
                }}
                history.appendChild(div);
                history.scrollTop = history.scrollHeight;
                return div;
            }}
            
            async function sendRagQuery() {{
                const input = document.getElementById('rag-query-input');
                const query = input.value.trim();
                if (!query) return;
                
                input.value = '';
                const sendBtn = document.getElementById('rag-send-btn');
                sendBtn.disabled = true;
                sendBtn.textContent = '...';
                
                appendChatMessage('user', query);
                const responseDiv = appendChatMessage('assistant', 'Thinking...');
                
                try {{
                    const modelName = document.getElementById('model-select').value;
                    const resp = await fetch('/rag/query', {{
                        method: 'POST',
                        headers: {{'Content-Type': 'application/json'}},
                        body: JSON.stringify({{ query: query, model: modelName, history: ragChatHistory }})
                    }});
                    
                    if (!resp.ok) {{
                        const err = await resp.json();
                        responseDiv.querySelector('span').textContent = 'Error: ' + (err.detail || err.error || 'Query failed');
                        responseDiv.querySelector('span').style.color = '#dc3545';
                    }} else {{
                        // Stream the response
                        const reader = resp.body.getReader();
                        const decoder = new TextDecoder();
                        let fullText = '';
                        const textSpan = responseDiv.querySelector('span');
                        
                        while (true) {{
                            const {{ done, value }} = await reader.read();
                            if (done) break;
                            const chunk = decoder.decode(value, {{ stream: true }});
                            // Parse SSE lines
                            const lines = chunk.split('\\n');
                            for (const line of lines) {{
                                if (line.startsWith('data: ')) {{
                                    try {{
                                        const parsed = JSON.parse(line.slice(6));
                                        if (parsed.token) {{
                                            fullText += parsed.token;
                                            textSpan.textContent = fullText;
                                        }}
                                        if (parsed.done) {{
                                            ragChatHistory.push({{ role: 'user', content: query }});
                                            ragChatHistory.push({{ role: 'assistant', content: fullText }});
                                        }}
                                        if (parsed.error) {{
                                            textSpan.textContent = 'Error: ' + parsed.error;
                                            textSpan.style.color = '#dc3545';
                                        }}
                                    }} catch(e) {{}}
                                }}
                            }}
                            document.getElementById('rag-chat-history').scrollTop = document.getElementById('rag-chat-history').scrollHeight;
                        }}
                    }}
                }} catch(e) {{
                    responseDiv.querySelector('span').textContent = 'Error: ' + e.message;
                    responseDiv.querySelector('span').style.color = '#dc3545';
                }} finally {{
                    sendBtn.disabled = false;
                    sendBtn.textContent = 'Ask';
                }}
            }}
            
            function closeSummary() {{
                document.getElementById('summary-modal').style.display = 'none';
            }}
            
            function showUploadModal() {{
                document.getElementById('upload-modal').style.display = 'block';
                document.getElementById('upload-status').textContent = '';
            }}
            
            function closeUploadModal() {{
                document.getElementById('upload-modal').style.display = 'none';
            }}
            
            async function uploadSummary() {{
                const fileInput = document.getElementById('summary-file');
                const modelSelect = document.getElementById('model-select');
                const modelName = modelSelect.value;
                
                if (!fileInput.files[0]) {{
                    document.getElementById('upload-status').textContent = 'Please select a file';
                    document.getElementById('upload-status').style.color = '#dc3545';
                    return;
                }}
                
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);
                
                document.getElementById('upload-status').textContent = 'Uploading...';
                document.getElementById('upload-status').style.color = '#ffc107';
                
                try {{
                    const response = await fetch(`/upload_summary?model=${{encodeURIComponent(modelName)}}`, {{
                        method: 'POST',
                        body: formData
                    }});
                    const data = await response.json();
                    
                    if (data.message) {{
                        document.getElementById('upload-status').textContent = '✅ ' + data.message;
                        document.getElementById('upload-status').style.color = '#28a745';
                        setTimeout(closeUploadModal, 2000);
                    }}
                }} catch(e) {{
                    document.getElementById('upload-status').textContent = 'Error: ' + e.message;
                    document.getElementById('upload-status').style.color = '#dc3545';
                }}
            }}
            
            // ===== Per-Object Summary Functions =====
            let currentObjectIdx = null;
            const objectSummaries = {{}};  // Local cache of object summaries
            
            function openObjectSummary(segmentIdx) {{
                currentObjectIdx = segmentIdx;
                const pane = document.getElementById('object-summary-pane');
                const color = segColors[segmentIdx % segColors.length];
                const label = segmentLabelMap[segmentIdx] || `Object ${{segmentIdx}}`;
                
                // Set object color and name
                document.getElementById('obj-summary-color').style.background = color;
                document.getElementById('obj-summary-name').value = label;
                
                // Load existing summary if any
                const summary = objectSummaries[segmentIdx] || {{}};
                document.getElementById('obj-summary-text').value = summary.text || '';
                
                // Show uploaded files
                const filesList = document.getElementById('obj-uploaded-files');
                filesList.innerHTML = '';
                if (summary.files && summary.files.length > 0) {{
                    summary.files.forEach(f => {{
                        const li = document.createElement('li');
                        li.textContent = f.name;
                        filesList.appendChild(li);
                    }});
                }}
                
                // Show training data status
                const trainingStatus = document.getElementById('obj-training-status');
                if (summary.training_data && Object.keys(summary.training_data).length > 0) {{
                    trainingStatus.innerHTML = `<span style="color:#28a745;">✓ Training data linked</span>`;
                }} else {{
                    trainingStatus.textContent = 'No training data associated';
                }}
                
                // Show the pane
                pane.classList.add('visible');
                
                // Fetch from server
                fetchObjectSummary(segmentIdx);
            }}
            
            function closeObjectSummary() {{
                document.getElementById('object-summary-pane').classList.remove('visible');
                currentObjectIdx = null;
            }}
            
            async function fetchObjectSummary(segmentIdx) {{
                try {{
                    const response = await fetch(`/object_summary/${{segmentIdx}}`);
                    const data = await response.json();
                    if (data.summary) {{
                        objectSummaries[segmentIdx] = data.summary;
                        document.getElementById('obj-summary-text').value = data.summary.text || '';
                        
                        const filesList = document.getElementById('obj-uploaded-files');
                        filesList.innerHTML = '';
                        if (data.summary.files && data.summary.files.length > 0) {{
                            data.summary.files.forEach((f, idx) => {{
                                const li = document.createElement('li');
                                li.style.cursor = 'pointer';
                                li.innerHTML = `📄 <a href="/object_summary/${{segmentIdx}}/file/${{f.idx !== undefined ? f.idx : idx}}" target="_blank" style="color:#00d4ff;text-decoration:underline;">${{f.name}}</a> <small style="color:#888;">(${{f.size || 'unknown'}})</small>`;
                                filesList.appendChild(li);
                            }});
                        }}
                        
                        // Update training data status
                        updateTrainingStatus(data.summary);
                    }}
                }} catch(e) {{
                    console.log('Could not fetch object summary:', e);
                }}
            }}
            
            function updateTrainingStatus(summary) {{
                const trainingStatus = document.getElementById('obj-training-status');
                if (summary.training_data && Object.keys(summary.training_data).length > 0) {{
                    const td = summary.training_data;
                    if (td.files_count) {{
                        trainingStatus.innerHTML = `<span style="color:#28a745;">✓ ${{td.files_count}} document(s) uploaded</span><br><small style="color:#888;">Last: ${{td.last_upload || 'N/A'}}</small>`;
                    }} else if (td.job_id) {{
                        trainingStatus.innerHTML = `<span style="color:#28a745;">✓ Training job: ${{td.job_id}}</span>`;
                    }} else {{
                        trainingStatus.innerHTML = `<span style="color:#28a745;">✓ Training data linked</span>`;
                    }}
                }} else if (summary.files && summary.files.length > 0) {{
                    trainingStatus.innerHTML = `<span style="color:#ffc107;">⚠ ${{summary.files.length}} file(s) - pending training link</span>`;
                }} else {{
                    trainingStatus.textContent = 'No training data associated';
                }}
            }}
            
            async function saveObjectSummary() {{
                if (currentObjectIdx === null) return;
                
                const label = document.getElementById('obj-summary-name').value;
                const text = document.getElementById('obj-summary-text').value;
                
                // Update label
                await updateLabel(currentObjectIdx, label);
                
                // Save summary
                const formData = new FormData();
                formData.append('segment_idx', currentObjectIdx);
                formData.append('text', text);
                formData.append('label', label);
                
                try {{
                    const response = await fetch('/object_summary', {{
                        method: 'POST',
                        body: formData
                    }});
                    const data = await response.json();
                    
                    if (data.status === 'ok') {{
                        objectSummaries[currentObjectIdx] = {{ text, label, files: objectSummaries[currentObjectIdx]?.files || [] }};
                        
                        // Flash save button green
                        const btn = document.querySelector('#object-summary-pane .btn-save');
                        btn.style.background = '#28a745';
                        btn.textContent = '✓ Saved!';
                        setTimeout(() => {{
                            btn.style.background = '#28a745';
                            btn.textContent = '💾 Save';
                        }}, 1500);
                    }}
                }} catch(e) {{
                    console.error('Save failed:', e);
                }}
            }}
            
            async function uploadObjectFiles() {{
                if (currentObjectIdx === null) return;
                
                const fileInput = document.getElementById('obj-summary-file');
                if (!fileInput.files || fileInput.files.length === 0) {{
                    alert('Please select files to upload');
                    return;
                }}
                
                const formData = new FormData();
                formData.append('segment_idx', currentObjectIdx);
                for (let i = 0; i < fileInput.files.length; i++) {{
                    formData.append('files', fileInput.files[i]);
                }}
                
                // Show uploading status
                const uploadBtn = document.querySelector('#object-summary-pane .btn-upload');
                uploadBtn.textContent = '⏳ Uploading...';
                uploadBtn.disabled = true;
                
                try {{
                    const response = await fetch('/object_summary/upload', {{
                        method: 'POST',
                        body: formData
                    }});
                    const data = await response.json();
                    
                    if (data.status === 'ok') {{
                        // Update training status immediately
                        if (data.training_data) {{
                            updateTrainingStatus({{ training_data: data.training_data }});
                        }}
                        
                        // Refresh file list
                        fetchObjectSummary(currentObjectIdx);
                        fileInput.value = '';
                        
                        // Flash success
                        uploadBtn.textContent = '✓ Uploaded!';
                        uploadBtn.style.background = '#28a745';
                        setTimeout(() => {{
                            uploadBtn.textContent = '⬆️ Upload Files';
                            uploadBtn.style.background = '#17a2b8';
                            uploadBtn.disabled = false;
                        }}, 1500);
                    }}
                }} catch(e) {{
                    console.error('Upload failed:', e);
                    uploadBtn.textContent = '⬆️ Upload Files';
                    uploadBtn.disabled = false;
                }}
            }}
            
            // Click on object in rendered image to open summary
            img.addEventListener('dblclick', async (e) => {{
                if (!showSegments || segmentMasks.length === 0) return;
                if (clickMode || extractMode) return;  // Don't interfere with other modes
                
                const rect = img.getBoundingClientRect();
                const scaleX = 1024 / rect.width;
                const scaleY = 768 / rect.height;
                const imgX = Math.round((e.clientX - rect.left) * scaleX);
                const imgY = Math.round((e.clientY - rect.top) * scaleY);
                
                const x = Math.floor(imgX / maskScale);
                const y = Math.floor(imgY / maskScale);
                
                // Find which segment was clicked
                for (let i = 0; i < segmentMasks.length; i++) {{
                    if (segmentMasks[i] && segmentMasks[i][y] && segmentMasks[i][y][x]) {{
                        openObjectSummary(i);
                        return;
                    }}
                }}
            }});
            
            let activeExtractionId = null;
            
            // RAG metadata functions for extractions
            function showExtractionRag(jobId) {{
                activeExtractionId = jobId;
                document.getElementById('extraction-rag').style.display = 'block';
                // Load existing summary
                fetch(`/extraction_summary/${{jobId}}`)
                    .then(r => r.json())
                    .then(data => {{
                        if (data.summary) {{
                            document.getElementById('extLabel').value = data.summary.label || '';
                            document.getElementById('extSummary').value = data.summary.text || '';
                            updateExtFileList(data.summary.files || []);
                        }} else {{
                            document.getElementById('extLabel').value = '';
                            document.getElementById('extSummary').value = '';
                            document.getElementById('extFileList').innerHTML = '';
                        }}
                    }}).catch(() => {{}});
            }}
            
            async function saveExtSummary() {{
                if (!activeExtractionId) return;
                const formData = new FormData();
                formData.append('job_id', activeExtractionId);
                formData.append('text', document.getElementById('extSummary').value);
                formData.append('label', document.getElementById('extLabel').value);
                try {{
                    const resp = await fetch('/extraction_summary', {{ method: 'POST', body: formData }});
                    if (resp.ok) {{
                        document.getElementById('extractStatus').textContent = '✅ Summary saved';
                    }}
                }} catch(e) {{
                    console.error('Save summary error:', e);
                }}
            }}
            
            async function uploadExtFiles() {{
                if (!activeExtractionId) return;
                const fileInput = document.getElementById('extFileUpload');
                if (!fileInput.files.length) return;
                const formData = new FormData();
                formData.append('job_id', activeExtractionId);
                for (const f of fileInput.files) formData.append('files', f);
                try {{
                    const resp = await fetch('/extraction_summary/upload', {{ method: 'POST', body: formData }});
                    const data = await resp.json();
                    if (data.uploaded) {{
                        updateExtFileList(data.uploaded.map((u, i) => ({{ idx: i, name: u.name, size: u.size }})));
                        document.getElementById('extractStatus').textContent = `✅ ${{data.uploaded.length}} file(s) uploaded`;
                    }}
                }} catch(e) {{
                    console.error('Upload error:', e);
                }}
                fileInput.value = '';
            }}
            
            function updateExtFileList(files) {{
                const el = document.getElementById('extFileList');
                if (!files || files.length === 0) {{ el.innerHTML = ''; return; }}
                el.innerHTML = files.map(f => `📎 ${{f.name}} (${{f.size || ''}})`).join('<br>');
            }}
            
            // ===== GARField 3D Extraction Functions =====
            let extractMode = false;
            let extractions = [];
            
            function toggleExtractMode() {{
                extractMode = !extractMode;
                const btn = document.getElementById('extractBtn');
                if (extractMode) {{
                    btn.textContent = '🎯 Extracting...';
                    btn.style.background = '#ffc107';
                    btn.style.color = '#000';
                    img.classList.add('click-mode');
                    document.getElementById('extractStatus').textContent = 'Click on object to extract 3D asset';
                    // Disable segment click mode if active
                    if (clickMode) toggleClickMode();
                }} else {{
                    btn.textContent = 'Click to Extract';
                    btn.style.background = '#ff6b6b';
                    btn.style.color = 'white';
                    img.classList.remove('click-mode');
                    document.getElementById('extractStatus').textContent = 'Click to extract 3D assets from scene';
                }}
            }}
            
            img.addEventListener('click', async (e) => {{
                if (!extractMode) return;
                
                const rect = img.getBoundingClientRect();
                const scaleX = 1024 / rect.width;
                const scaleY = 768 / rect.height;
                const x = Math.round((e.clientX - rect.left) * scaleX);
                const y = Math.round((e.clientY - rect.top) * scaleY);
                
                document.getElementById('extractStatus').textContent = 'Extracting 3D asset...';
                
                const modelSelect = document.getElementById('model-select');
                const scaleLevel = document.getElementById('extract-scale').value;
                
                const formData = new FormData();
                formData.append('x', x);
                formData.append('y', y);
                formData.append('model_name', modelSelect.value);
                formData.append('scale_level', scaleLevel);
                formData.append('azimuth', azSlider.value);
                formData.append('elevation', elSlider.value);
                formData.append('zoom', zoomSlider.value);
                formData.append('cam_idx', camSlider.value);
                formData.append('width', 1024);
                formData.append('height', 768);
                
                try {{
                    const controller = new AbortController();
                    const timeout = setTimeout(() => controller.abort(), 120000);
                    const response = await fetch('/garfield/extract', {{
                        method: 'POST',
                        body: formData,
                        signal: controller.signal
                    }});
                    clearTimeout(timeout);
                    const data = await response.json();
                    
                    if (data.status === 'completed') {{
                        extractions.push(data);
                        updateExtractionList();
                        document.getElementById('extractStatus').innerHTML = 
                            `✅ Extracted ${{data.num_gaussians}} gaussians <a href="/garfield/download/${{data.job_id}}" style="color:#ffc107;">Download</a>`;
                    }} else if (data.status === 'no_selection') {{
                        document.getElementById('extractStatus').textContent = '⚠️ No object found at click position';
                    }} else {{
                        document.getElementById('extractStatus').textContent = '❌ Extraction failed: ' + (data.error || data.message);
                    }}
                }} catch(e) {{
                    const msg = e.name === 'AbortError' ? 'Extraction timed out (model may be too large)' : e.message;
                    document.getElementById('extractStatus').textContent = '❌ Error: ' + msg;
                }} finally {{
                    // Ensure loading overlay is cleared
                    loading.style.display = 'none';
                    img.style.display = 'block';
                }}
            }});
            
            function updateExtractionList() {{
                const list = document.getElementById('extraction-list');
                list.innerHTML = '';
                extractions.forEach((ext, idx) => {{
                    const item = document.createElement('div');
                    item.className = 'extraction-item';
                    item.innerHTML = `
                        <span>Asset ${{idx + 1}}: ${{ext.num_gaussians}} pts</span>
                        <button onclick="viewExtraction(${{idx}})" style="background:#ffc107;color:#000;border:none;border-radius:3px;padding:2px 8px;cursor:pointer;font-size:11px;">👁️ View</button>
                        <button onclick="showExtractionRag('${{ext.job_id}}')" style="background:#17a2b8;color:#fff;border:none;border-radius:3px;padding:2px 8px;cursor:pointer;font-size:11px;">📄 Info</button>
                    `;
                    list.appendChild(item);
                }});
            }}
            
            // View extracted asset in viewer
            let viewingExtraction = null;
            let extAzimuth = 0;
            let extElevation = 0;
            let extZoom = 1.0;
            let isDragging = false;
            let lastMouseX = 0;
            let lastMouseY = 0;
            let renderPending = false;
            let lastRenderTime = 0;
            const RENDER_THROTTLE_MS = 50;  // Limit to 20 FPS for smoother rotation
            
            async function viewExtraction(idx) {{
                const ext = extractions[idx];
                if (!ext) return;
                
                viewingExtraction = ext;
                extAzimuth = 0;
                extElevation = 0;
                extZoom = 1.0;
                
                document.getElementById('extractStatus').innerHTML = 
                    `<span style="color:#ffc107;">👁️ Viewing Asset ${{idx + 1}} (${{ext.num_gaussians}} gaussians)</span><br>` +
                    `<span style="font-size:11px;color:#aaa;">Arrow keys to rotate • +/- to zoom</span><br>` +
                    `<button onclick="exitExtractionView()" style="margin-top:5px;background:#dc3545;color:white;border:none;border-radius:3px;padding:4px 10px;cursor:pointer;">Exit View</button>`;
                
                img.style.cursor = 'grab';
                renderExtractionOverlay();
            }}
            
            function exitExtractionView() {{
                viewingExtraction = null;
                img.style.cursor = 'default';
                document.getElementById('extractStatus').textContent = 'Click to extract 3D assets from scene';
                updateRender();
            }}
            
            // Mouse drag rotation for extraction view
            img.addEventListener('mousedown', (e) => {{
                if (!viewingExtraction) return;
                isDragging = true;
                lastMouseX = e.clientX;
                lastMouseY = e.clientY;
                img.style.cursor = 'grabbing';
                e.preventDefault();
            }});
            
            document.addEventListener('mousemove', (e) => {{
                if (!isDragging || !viewingExtraction) return;
                
                const deltaX = e.clientX - lastMouseX;
                const deltaY = e.clientY - lastMouseY;
                
                extAzimuth += deltaX * 0.5;
                extElevation = Math.max(-89, Math.min(89, extElevation - deltaY * 0.5));
                
                lastMouseX = e.clientX;
                lastMouseY = e.clientY;
                
                // Throttle render requests to reduce GPU pressure
                const now = Date.now();
                if (!renderPending && (now - lastRenderTime) >= RENDER_THROTTLE_MS) {{
                    renderPending = true;
                    lastRenderTime = now;
                    renderExtractionOverlay().finally(() => {{ renderPending = false; }});
                }}
            }});
            
            document.addEventListener('mouseup', () => {{
                if (isDragging && viewingExtraction) {{
                    isDragging = false;
                    img.style.cursor = 'grab';
                    // Render final position on mouse up
                    if (!renderPending) {{
                        renderPending = true;
                        renderExtractionOverlay().finally(() => {{ renderPending = false; }});
                    }}
                }}
            }});
            
            // Scroll to zoom for extraction view (throttled)
            img.addEventListener('wheel', (e) => {{
                if (!viewingExtraction) return;
                e.preventDefault();
                extZoom = Math.max(0.2, Math.min(5, extZoom - e.deltaY * 0.002));
                
                const now = Date.now();
                if (!renderPending && (now - lastRenderTime) >= RENDER_THROTTLE_MS) {{
                    renderPending = true;
                    lastRenderTime = now;
                    renderExtractionOverlay().finally(() => {{ renderPending = false; }});
                }}
            }});
            
            // Keyboard controls for rotation (works in both extraction and normal/segmentation views)
            document.addEventListener('keydown', (e) => {{
                // Skip if focused on an input/textarea
                if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') return;
                
                const rotateStep = 10;
                const zoomStep = 0.1;
                
                if (viewingExtraction) {{
                    // Extraction view: adjust extraction orbit
                    let changed = false;
                    switch(e.key) {{
                        case 'ArrowLeft':  extAzimuth -= rotateStep; changed = true; break;
                        case 'ArrowRight': extAzimuth += rotateStep; changed = true; break;
                        case 'ArrowUp':    extElevation = Math.min(89, extElevation + rotateStep); changed = true; break;
                        case 'ArrowDown':  extElevation = Math.max(-89, extElevation - rotateStep); changed = true; break;
                        case '+': case '=': extZoom = Math.min(5, extZoom + zoomStep); changed = true; break;
                        case '-': case '_': extZoom = Math.max(0.2, extZoom - zoomStep); changed = true; break;
                        case 'Escape': exitExtractionView(); return;
                    }}
                    if (changed) {{
                        e.preventDefault();
                        if (!renderPending) {{
                            renderPending = true;
                            renderExtractionOverlay().finally(() => {{ renderPending = false; }});
                        }}
                    }}
                }} else {{
                    // Normal/segmentation view: adjust main sliders
                    let changed = false;
                    switch(e.key) {{
                        case 'ArrowLeft':
                            azSlider.value = Math.max(-180, parseInt(azSlider.value) - rotateStep);
                            changed = true; break;
                        case 'ArrowRight':
                            azSlider.value = Math.min(180, parseInt(azSlider.value) + rotateStep);
                            changed = true; break;
                        case 'ArrowUp':
                            elSlider.value = Math.min(45, parseInt(elSlider.value) + rotateStep);
                            changed = true; break;
                        case 'ArrowDown':
                            elSlider.value = Math.max(-45, parseInt(elSlider.value) - rotateStep);
                            changed = true; break;
                        case '+': case '=':
                            zoomSlider.value = Math.min(2, parseFloat(zoomSlider.value) + zoomStep).toFixed(1);
                            changed = true; break;
                        case '-': case '_':
                            zoomSlider.value = Math.max(0.1, parseFloat(zoomSlider.value) - zoomStep).toFixed(1);
                            changed = true; break;
                    }}
                    if (changed) {{
                        e.preventDefault();
                        document.getElementById('az-val').textContent = azSlider.value;
                        document.getElementById('el-val').textContent = elSlider.value;
                        document.getElementById('zoom-val').textContent = zoomSlider.value;
                        debouncedUpdate();
                    }}
                }}
            }});
            
            async function renderExtractionOverlay() {{
                if (!viewingExtraction) return;
                
                // Request render with extraction
                try {{
                    const response = await fetch(`/garfield/render_extraction?job_id=${{viewingExtraction.job_id}}&azimuth=${{extAzimuth}}&elevation=${{extElevation}}&zoom=${{extZoom}}&width=1024&height=768`);
                    if (response.ok) {{
                        const blob = await response.blob();
                        img.src = URL.createObjectURL(blob);
                    }}
                }} catch(e) {{
                    console.log('Extraction render error:', e);
                }}
            }}
            
            // Override render to show extraction if viewing
            const origUpdateRender = updateRender;
            updateRender = function() {{
                if (viewingExtraction) {{
                    renderExtractionOverlay();
                }} else {{
                    origUpdateRender();
                }}
            }};
            
            async function clearExtractions() {{
                extractions = [];
                viewingExtraction = null;
                img.style.cursor = 'default';
                document.getElementById('extraction-list').innerHTML = '';
                document.getElementById('extractStatus').textContent = 'Click to extract 3D assets from scene';
                // Clean up server-side extraction cache and temp files
                try {{ await fetch('/garfield/clear', {{method: 'POST'}}); }} catch(e) {{}}
                updateRender();
            }}
        </script>
    </body>
    </html>
    """

@app.get("/render")
async def render(
    width: int = Query(800, ge=100, le=1920),
    height: int = Query(600, ge=100, le=1080),
    azimuth: float = Query(0, ge=-180, le=180),
    elevation: float = Query(0, ge=-45, le=45),
    zoom: float = Query(0.5, ge=0.1, le=2.0),
    cam_idx: int = Query(0, ge=0, le=100)
):
    """Render the gaussian splat and return as PNG"""
    img = render_view(width, height, azimuth, elevation, zoom, cam_idx)
    
    if img is None:
        return Response(content=b"Render failed", status_code=500)
    
    # Encode as PNG
    from PIL import Image
    pil_img = Image.fromarray(img)
    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG')
    buffer.seek(0)
    
    return Response(content=buffer.getvalue(), media_type="image/png")

@app.get("/load_model")
async def load_model_endpoint(model: str = Query(...)):
    """Load a different model"""
    success = load_model(model)
    if success:
        return {"status": "ok", "model": model_name, "num_gaussians": gsplat.num_gaussians}
    return {"status": "error", "message": "Failed to load model"}

@app.get("/info")
async def info():
    # Always refresh the available models list
    current_models = get_available_models()
    return {
        "model_name": model_name,
        "num_gaussians": gsplat.num_gaussians if gsplat else 0,
        "models_available": current_models
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": gsplat is not None}

@app.delete("/delete_model")
async def delete_model_endpoint(model: str = Query(...)):
    """Delete a model PLY file"""
    global gsplat, model_metadata, model_name, available_models
    
    model_path = MODEL_DIR / model
    if not model_path.exists():
        return Response(content=f"Model {model} not found", status_code=404)
    
    # Delete the file
    model_path.unlink()
    logger.info(f"Deleted model: {model}")
    
    # Also delete metadata if exists
    meta_path = MODEL_DIR / f"{model.replace('.ply', '')}_metadata.json"
    if meta_path.exists():
        meta_path.unlink()
    
    # Refresh available models and load a different one
    available_models = get_available_models()
    if available_models:
        load_model(available_models[0])
    else:
        gsplat = None
        model_metadata = None
        model_name = ""
    
    return {"status": "ok", "message": f"Deleted {model}"}


# ==================== SAM-2 Segmentation Endpoints ====================

@app.post("/segment")
async def segment_current_view(
    azimuth: float = Form(0),
    elevation: float = Form(0),
    zoom: float = Form(0.5),
    cam_idx: int = Form(0),
    points_json: Optional[str] = Form(None)
):
    """Segment objects in the current rendered view"""
    global current_segments, segment_labels
    
    # Render current view
    img = render_view(1024, 768, azimuth, elevation, zoom, cam_idx)
    if img is None:
        return JSONResponse({"error": "Failed to render view"}, status_code=500)
    
    # Parse points if provided
    points = json.loads(points_json) if points_json else None
    auto_mode = points is None
    
    # Run segmentation
    segments = segment_image(img, points=points, auto_mode=auto_mode)
    
    if segments is None:
        return JSONResponse({"error": "Segmentation failed"}, status_code=500)
    
    # Use auto-generated labels from segmentation
    segment_labels = segments.get("labels", {i: f"Object {i}" for i in range(segments["num_segments"])})
    
    return {
        "status": "ok",
        "num_segments": segments["num_segments"],
        "scores": segments["scores"],
        "labels": segment_labels
    }


@app.get("/render_with_segments")
async def render_with_segments(
    width: int = Query(1024),
    height: int = Query(768),
    azimuth: float = Query(0),
    elevation: float = Query(0),
    zoom: float = Query(0.5),
    cam_idx: int = Query(0)
):
    """Render view with segmentation overlay and labels"""
    img = render_view(width, height, azimuth, elevation, zoom, cam_idx)
    if img is None:
        return Response(content=b"Render failed", status_code=500)
    
    # Apply segmentation overlay if segments exist
    if current_segments and "masks" in current_segments:
        img = create_overlay_with_labels(img, current_segments, segment_labels)
    
    from PIL import Image
    pil_img = Image.fromarray(img)
    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG')
    buffer.seek(0)
    
    return Response(content=buffer.getvalue(), media_type="image/png")


@app.post("/segment/point")
async def segment_at_point(
    x: int = Form(...),
    y: int = Form(...),
    label: int = Form(1),
    azimuth: float = Form(0),
    elevation: float = Form(0),
    zoom: float = Form(0.5),
    cam_idx: int = Form(0)
):
    """Segment object at specific point click"""
    global current_segments, segment_labels
    
    img = render_view(1024, 768, azimuth, elevation, zoom, cam_idx)
    if img is None:
        return JSONResponse({"error": "Failed to render view"}, status_code=500)
    
    points = [{"x": x, "y": y, "label": label}]
    segments = segment_image(img, points=points, auto_mode=False)
    
    if segments is None:
        return JSONResponse({"error": "Segmentation failed"}, status_code=500)
    
    # Add to existing segments or create new
    if not current_segments or "masks" not in current_segments:
        current_segments = {"masks": [], "scores": [], "num_segments": 0}
    
    new_idx = -1
    for i, (mask, score) in enumerate(zip(segments["masks"], segments["scores"])):
        new_idx = len(current_segments["masks"])
        current_segments["masks"].append(mask)
        current_segments["scores"].append(score)
        current_segments["num_segments"] += 1
        # Use auto-generated label from segment_image
        segment_labels[new_idx] = segments.get("labels", {}).get(i, f"Object {new_idx}")
    
    return {
        "status": "ok",
        "num_segments": current_segments["num_segments"],
        "new_segment_idx": current_segments["num_segments"] - 1
    }


@app.post("/segment/label")
async def update_segment_label(
    segment_idx: int = Form(...),
    label: str = Form(...)
):
    """Update label for a segment"""
    global segment_labels
    
    if segment_idx < 0 or (current_segments and segment_idx >= current_segments.get("num_segments", 0)):
        return JSONResponse({"error": "Invalid segment index"}, status_code=400)
    
    segment_labels[segment_idx] = label
    return {"status": "ok", "segment_idx": segment_idx, "label": label}


@app.get("/segment/labels")
async def get_segment_labels():
    """Get all current segment labels"""
    return {
        "labels": segment_labels,
        "num_segments": current_segments.get("num_segments", 0) if current_segments else 0
    }


@app.get("/segment/masks")
async def get_segment_masks():
    """Get segment mask data for hover detection (downsampled for efficiency)"""
    if not current_segments or "masks" not in current_segments:
        return {"masks": [], "labels": segment_labels}
    
    # Downsample masks for transfer (every 4th pixel)
    downsampled = []
    for mask in current_segments["masks"]:
        # Convert to list of lists, downsampled
        ds_mask = mask[::4, ::4].tolist()
        downsampled.append(ds_mask)
    
    return {
        "masks": downsampled,
        "labels": segment_labels,
        "scale": 4  # Client needs to multiply coordinates by this
    }


@app.post("/segment/clear")
async def clear_segments():
    """Clear all segments"""
    global current_segments, segment_labels
    current_segments = {}
    segment_labels = {}
    return {"status": "ok", "message": "Segments cleared"}


@app.get("/segment/status")
async def segment_status():
    """Get SAM-2 and segmentation status"""
    return {
        "sam2_loaded": sam2_loaded,
        "has_segments": bool(current_segments and current_segments.get("masks")),
        "num_segments": current_segments.get("num_segments", 0) if current_segments else 0,
        "labels": segment_labels
    }


# ===== Summary Endpoints =====

@app.get("/model_summary")
async def get_model_summary(model: str = Query(...)):
    """Get summary for a model from the model service"""
    try:
        import requests
        response = requests.get(f"{MODEL_SERVICE_URL}/summary/{model}", timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        logger.error(f"Error fetching summary: {e}")
    
    return {"model": model, "has_summary": False, "summary": None}


@app.post("/upload_summary")
async def upload_model_summary(model: str = Query(...), file: UploadFile = File(...)):
    """Upload summary document for a model"""
    try:
        import requests
        content = await file.read()
        files = {'file': (file.filename, content, file.content_type)}
        response = requests.post(
            f"{MODEL_SERVICE_URL}/summary/{model}",
            files=files,
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        return {"error": f"Upload failed: {response.status_code}"}
    except Exception as e:
        logger.error(f"Error uploading summary: {e}")
        return {"error": str(e)}


# ===== Per-Object Summary Endpoints =====

@app.get("/object_summary/{segment_idx}")
async def get_object_summary(segment_idx: int):
    """Get summary for a specific segmented object"""
    summary = object_summaries.get(segment_idx, {})
    label = segment_labels.get(segment_idx, f"Object {segment_idx}")
    
    # Create a serializable copy (exclude binary file data, include file index)
    if summary:
        serializable_summary = {
            "text": summary.get("text", ""),
            "label": summary.get("label", label),
            "training_data": summary.get("training_data", {}),
            "files": [
                {"idx": i, "name": f.get("name"), "size": f.get("size"), "content_type": f.get("content_type")}
                for i, f in enumerate(summary.get("files", []))
            ]
        }
    else:
        serializable_summary = None
    
    return {
        "segment_idx": segment_idx,
        "label": label,
        "summary": serializable_summary
    }


@app.post("/object_summary")
async def save_object_summary(
    segment_idx: int = Form(...),
    text: str = Form(""),
    label: str = Form("")
):
    """Save summary text for a specific segmented object"""
    global object_summaries, segment_labels
    
    # Initialize if not exists
    if segment_idx not in object_summaries:
        object_summaries[segment_idx] = {"text": "", "files": [], "training_data": {}}
    
    object_summaries[segment_idx]["text"] = text
    object_summaries[segment_idx]["label"] = label
    
    # Also update the segment label
    if label:
        segment_labels[segment_idx] = label
    
    logger.info(f"Saved summary for object {segment_idx}: label={label}, text={text[:50] if text else 'empty'}...")
    return {"status": "ok", "segment_idx": segment_idx, "text_saved": bool(text)}


@app.post("/object_summary/upload")
async def upload_object_files(
    segment_idx: int = Form(...),
    files: List[UploadFile] = File(...)
):
    """Upload files associated with a specific segmented object for RAG training"""
    global object_summaries
    
    # Initialize if not exists
    if segment_idx not in object_summaries:
        object_summaries[segment_idx] = {"text": "", "files": [], "training_data": {}}
    
    uploaded = []
    for file in files:
        content = await file.read()
        file_info = {
            "name": file.filename,
            "content_type": file.content_type,
            "size": f"{len(content) / 1024:.1f} KB",
            "data": content  # Store in memory for now
        }
        object_summaries[segment_idx]["files"].append(file_info)
        uploaded.append({"name": file.filename, "size": file_info["size"]})
    
    # Update training data status when files are uploaded
    total_files = len(object_summaries[segment_idx]["files"])
    object_summaries[segment_idx]["training_data"] = {
        "files_count": total_files,
        "last_upload": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "documents_uploaded"
    }
    
    logger.info(f"Uploaded {len(uploaded)} files for object {segment_idx}, total: {total_files}")
    return {"status": "ok", "segment_idx": segment_idx, "uploaded": uploaded, "training_data": object_summaries[segment_idx]["training_data"]}


@app.get("/object_summary/{segment_idx}/file/{file_idx}")
async def get_object_file(segment_idx: int, file_idx: int):
    """Serve an uploaded file for viewing"""
    if segment_idx not in object_summaries:
        return Response(content="Object not found", status_code=404)
    
    files = object_summaries[segment_idx].get("files", [])
    if file_idx < 0 or file_idx >= len(files):
        return Response(content="File not found", status_code=404)
    
    file_info = files[file_idx]
    content = file_info.get("data", b"")
    content_type = file_info.get("content_type", "application/octet-stream")
    filename = file_info.get("name", "file")
    
    return Response(
        content=content,
        media_type=content_type,
        headers={"Content-Disposition": f"inline; filename=\"{filename}\""}
    )


@app.get("/object_summaries")
async def get_all_object_summaries():
    """Get summaries for all segmented objects"""
    result = {}
    for idx, summary in object_summaries.items():
        result[idx] = {
            "label": segment_labels.get(idx, f"Object {idx}"),
            "text": summary.get("text", ""),
            "files_count": len(summary.get("files", [])),
            "has_training": bool(summary.get("training_data"))
        }
    return {"summaries": result, "count": len(result)}


# ===== Per-Extraction RAG Metadata Endpoints =====

@app.get("/extraction_summary/{job_id}")
async def get_extraction_summary(job_id: str):
    """Get RAG summary for a specific extraction job"""
    summary = extraction_summaries.get(job_id, {})
    if summary:
        serializable = {
            "text": summary.get("text", ""),
            "label": summary.get("label", ""),
            "training_data": summary.get("training_data", {}),
            "files": [
                {"idx": i, "name": f.get("name"), "size": f.get("size"), "content_type": f.get("content_type")}
                for i, f in enumerate(summary.get("files", []))
            ]
        }
    else:
        serializable = None
    return {"job_id": job_id, "summary": serializable}


@app.post("/extraction_summary")
async def save_extraction_summary(
    job_id: str = Form(...),
    text: str = Form(""),
    label: str = Form("")
):
    """Save RAG summary text for a specific extraction"""
    global extraction_summaries
    if job_id not in extraction_summaries:
        extraction_summaries[job_id] = {"text": "", "label": "", "files": [], "training_data": {}}
    extraction_summaries[job_id]["text"] = text
    extraction_summaries[job_id]["label"] = label
    logger.info(f"Saved extraction summary for {job_id}: label={label}")
    return {"status": "ok", "job_id": job_id}


@app.post("/extraction_summary/upload")
async def upload_extraction_files(
    job_id: str = Form(...),
    files: List[UploadFile] = File(...)
):
    """Upload documents associated with an extraction for RAG training"""
    global extraction_summaries
    if job_id not in extraction_summaries:
        extraction_summaries[job_id] = {"text": "", "label": "", "files": [], "training_data": {}}
    
    uploaded = []
    for file in files:
        content = await file.read()
        file_info = {
            "name": file.filename,
            "content_type": file.content_type,
            "size": f"{len(content) / 1024:.1f} KB",
            "data": content
        }
        extraction_summaries[job_id]["files"].append(file_info)
        uploaded.append({"name": file.filename, "size": file_info["size"]})
    
    total_files = len(extraction_summaries[job_id]["files"])
    extraction_summaries[job_id]["training_data"] = {
        "files_count": total_files,
        "last_upload": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "documents_uploaded"
    }
    logger.info(f"Uploaded {len(uploaded)} files for extraction {job_id}, total: {total_files}")
    return {"status": "ok", "job_id": job_id, "uploaded": uploaded}


@app.get("/extraction_summary/{job_id}/file/{file_idx}")
async def get_extraction_file(job_id: str, file_idx: int):
    """Serve an uploaded file for an extraction"""
    if job_id not in extraction_summaries:
        return Response(content="Extraction not found", status_code=404)
    files = extraction_summaries[job_id].get("files", [])
    if file_idx < 0 or file_idx >= len(files):
        return Response(content="File not found", status_code=404)
    file_info = files[file_idx]
    return Response(
        content=file_info.get("data", b""),
        media_type=file_info.get("content_type", "application/octet-stream"),
        headers={"Content-Disposition": f"inline; filename=\"{file_info.get('name', 'file')}\""}
    )


@app.get("/extraction_summaries")
async def get_all_extraction_summaries():
    """Get summaries for all extractions"""
    result = {}
    for job_id, summary in extraction_summaries.items():
        result[job_id] = {
            "label": summary.get("label", ""),
            "text": summary.get("text", ""),
            "files_count": len(summary.get("files", [])),
            "has_training": bool(summary.get("training_data"))
        }
    return {"summaries": result, "count": len(result)}


# ===== RAG Query Endpoints =====

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "nemotron-mini")


def _extract_file_text(file_info: dict, max_chars: int = 2000) -> str:
    """Extract readable text from an uploaded file for RAG context."""
    ct = file_info.get("content_type", "")
    data = file_info.get("data", b"")
    name = file_info.get("name", "file")
    
    # Plain text, JSON, CSV, markdown, etc.
    if "text" in ct or ct in ("application/json", "application/csv"):
        try:
            return data.decode("utf-8", errors="replace")[:max_chars]
        except Exception:
            return ""
    
    # PDF extraction
    if ct == "application/pdf" or name.lower().endswith(".pdf"):
        try:
            import io
            import PyPDF2
            reader = PyPDF2.PdfReader(io.BytesIO(data))
            text_parts = []
            for page in reader.pages[:10]:
                text_parts.append(page.extract_text() or "")
            return "\n".join(text_parts)[:max_chars]
        except Exception:
            pass
        # Fallback: try pdfminer
        try:
            from pdfminer.high_level import extract_text as pdf_extract
            import io
            return pdf_extract(io.BytesIO(data))[:max_chars]
        except Exception:
            return f"[PDF file: {name}, {file_info.get('size', 'unknown size')}]"
    
    # Non-text files: include name/type as context
    return f"[Uploaded file: {name}, type: {ct}, size: {file_info.get('size', 'unknown')}]"


def _build_rag_context(model_name: str = None) -> str:
    """Build a text context from all available RAG data sources.
    
    Aggregates data from:
    - Model info and metadata
    - SAM-2 segmentation (current_segments, segment_labels, object_summaries)
    - GARField 3D extractions (extraction_cache, extraction_summaries)
    - Uploaded documents (text, PDF, and other files)
    """
    parts = []

    # Model info
    if gsplat is not None:
        parts.append(f"Model: {model_name or 'unknown'}, {gsplat.num_gaussians} gaussians")

    # Model-level RAG metadata
    if rag_metadata:
        if rag_metadata.get("title"):
            parts.append(f"Scene title: {rag_metadata['title']}")
        if rag_metadata.get("description"):
            parts.append(f"Scene description: {rag_metadata['description']}")
        if rag_labels:
            parts.append(f"Known objects (from metadata): {', '.join(rag_labels)}")

    # === SAM-2 Segmentation Data ===
    # Include segment count even without manual labels
    if current_segments and current_segments.get("masks"):
        num_segs = current_segments.get("num_segments", len(current_segments["masks"]))
        parts.append(f"SAM-2 segmentation: {num_segs} object(s) detected in scene")
    
    # Include all segment labels (auto-generated and manual)
    if segment_labels:
        labels_list = [f"{idx}: {lbl}" for idx, lbl in sorted(segment_labels.items())]
        parts.append(f"Segmented objects: {'; '.join(labels_list)}")
    
    # Include per-object summaries and uploaded documents
    if object_summaries:
        for idx, s in sorted(object_summaries.items()):
            label = segment_labels.get(idx, f"Object {idx}")
            text = s.get("text", "")
            files = s.get("files", [])
            entry = f"SAM-2 Object '{label}'"
            if text:
                entry += f": {text}"
            if files:
                entry += f" ({len(files)} document(s) uploaded)"
            for f in files:
                doc_text = _extract_file_text(f)
                if doc_text:
                    entry += f"\n  Document '{f.get('name', 'file')}': {doc_text}"
            parts.append(entry)

    # === GARField 3D Extraction Data ===
    # Include extraction_cache entries (actual extraction data, even without user-saved summaries)
    if extraction_cache:
        for job_id, ext_data in extraction_cache.items():
            num_gs = len(ext_data.get('indices', []))
            ext_model = ext_data.get('model_name', '')
            cam_idx = ext_data.get('cam_idx', 0)
            
            # Check if there's a corresponding user-saved summary
            summary = extraction_summaries.get(job_id, {})
            label = summary.get("label", "") or f"extraction_{job_id}"
            text = summary.get("text", "")
            files = summary.get("files", [])
            
            entry = f"3D Extraction '{label}': {num_gs} gaussians extracted from camera {cam_idx}"
            if text:
                entry += f" - {text}"
            if files:
                entry += f" ({len(files)} document(s) uploaded)"
                for f in files:
                    doc_text = _extract_file_text(f)
                    if doc_text:
                        entry += f"\n  Document '{f.get('name', 'file')}': {doc_text}"
            parts.append(entry)
    
    # Include extraction_summaries that may not have a corresponding cache entry
    # (e.g., cache was cleared but summaries remain)
    if extraction_summaries:
        for job_id, s in extraction_summaries.items():
            if job_id in extraction_cache:
                continue  # Already handled above
            label = s.get("label", job_id)
            text = s.get("text", "")
            files = s.get("files", [])
            if text or label or files:
                entry = f"3D Extraction '{label}'"
                if text:
                    entry += f": {text}"
                if files:
                    entry += f" ({len(files)} document(s))"
                    for f in files:
                        doc_text = _extract_file_text(f)
                        if doc_text:
                            entry += f"\n  Document '{f.get('name', 'file')}': {doc_text}"
                parts.append(entry)

    # Model-level uploaded summary (from rendering service)
    if model_name:
        try:
            import requests as req
            resp = req.get(f"{MODEL_SERVICE_URL}/summary/{model_name}", timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("has_summary") and data.get("summary"):
                    parts.append(f"Model document summary: {data['summary'][:3000]}")
        except Exception:
            pass

    return "\n".join(parts) if parts else ""


@app.get("/rag/context")
async def get_rag_context(model: str = Query(None)):
    """Get aggregated RAG context for the current model"""
    context_text = _build_rag_context(model)

    seg_labels = [segment_labels.get(idx, f"Object {idx}") for idx in sorted(segment_labels.keys())] if segment_labels else []
    docs_count = sum(len(s.get("files", [])) for s in object_summaries.values())
    docs_count += sum(len(s.get("files", [])) for s in extraction_summaries.values())
    
    # Count extractions from both cache and summaries
    all_extraction_ids = set(extraction_cache.keys()) | set(extraction_summaries.keys())
    
    # Count SAM-2 segments from current_segments (auto-detected)
    sam2_count = current_segments.get("num_segments", 0) if current_segments else 0

    model_summary = None
    if model:
        try:
            import requests as req
            resp = req.get(f"{MODEL_SERVICE_URL}/summary/{model}", timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("has_summary"):
                    model_summary = data.get("summary", "")[:500]
        except Exception:
            pass

    return {
        "model_name": model,
        "model_summary": model_summary,
        "segments_count": max(len(segment_labels), sam2_count),
        "segment_labels": seg_labels,
        "extractions_count": len(all_extraction_ids),
        "documents_count": docs_count,
        "context_length": len(context_text),
    }


@app.get("/rag/status")
async def get_rag_status():
    """Check if the Ollama LLM is reachable and the model is available"""
    try:
        import requests as req
        resp = req.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            # Check if our model (or a prefix match) is available
            matched = [m for m in models if OLLAMA_MODEL in m]
            if matched:
                return {"available": True, "model": matched[0], "all_models": models}
            else:
                return {"available": False, "error": f"Model '{OLLAMA_MODEL}' not found. Available: {models}", "all_models": models}
        return {"available": False, "error": f"Ollama returned {resp.status_code}"}
    except Exception as e:
        return {"available": False, "error": str(e)}


@app.post("/rag/query")
async def rag_query(body: dict):
    """Query the LLM with RAG context from the scene. Streams SSE tokens."""
    from starlette.responses import StreamingResponse
    import requests as req

    query = body.get("query", "").strip()
    model_name = body.get("model")
    history = body.get("history", [])

    if not query:
        return JSONResponse({"error": "Empty query"}, status_code=400)

    # Build context
    context = _build_rag_context(model_name)

    # Build messages
    system_msg = (
        "You are an AI assistant for a 3D Gaussian Splat viewer application. "
        "You help users understand what is in their 3D scene based on available data including: "
        "SAM-2 segmentation results, GARField 3D extractions, uploaded documents/data, "
        "and scene metadata. Answer concisely and accurately based on the "
        "available context. If you don't have enough information, say so.\n\n"
        "=== Scene Context ===\n"
        f"{context if context else 'No context data available yet. Try: (1) Run SAM-2 Auto Segment to detect objects, (2) Use GARField Click to Extract for 3D extractions, (3) Upload documents via Upload Info.'}\n"
        "=== End Context ==="
    )

    messages = [{"role": "system", "content": system_msg}]
    # Add conversation history (last 10 exchanges max)
    for msg in history[-20:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": query})

    # Find the actual model name
    try:
        tags_resp = req.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        available_models = [m["name"] for m in tags_resp.json().get("models", [])]
        actual_model = next((m for m in available_models if OLLAMA_MODEL in m), OLLAMA_MODEL)
    except Exception:
        actual_model = OLLAMA_MODEL

    def stream_response():
        try:
            resp = req.post(
                f"{OLLAMA_URL}/api/chat",
                json={"model": actual_model, "messages": messages, "stream": True},
                stream=True,
                timeout=120,
            )
            if resp.status_code != 200:
                yield f"data: {json.dumps({'error': f'Ollama error {resp.status_code}: {resp.text[:200]}'})}\n\n"
                return

            for line in resp.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        token = chunk.get("message", {}).get("content", "")
                        done = chunk.get("done", False)
                        if token:
                            yield f"data: {json.dumps({'token': token})}\n\n"
                        if done:
                            yield f"data: {json.dumps({'done': True})}\n\n"
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(stream_response(), media_type="text/event-stream")


# ===== GARField 3D Extraction Endpoints =====


@app.post("/garfield/extract")
async def garfield_extract(
    x: int = Form(...),
    y: int = Form(...),
    model_name: str = Form(...),
    scale_level: float = Form(0.5),
    azimuth: float = Form(0),
    elevation: float = Form(0),
    zoom: float = Form(1.0),
    cam_idx: int = Form(0),
    width: int = Form(1024),
    height: int = Form(768)
):
    """Extract 3D object using SAM-2 mask + accurate camera projection"""
    return await local_garfield_extract(
        x, y, model_name, scale_level, azimuth, elevation, zoom,
        width, height, cam_idx
    )


async def local_garfield_extract(x, y, model_name_param, scale_level, azimuth, elevation, zoom, width, height, cam_idx=0):
    """Extract 3D object using SAM-2 mask + actual camera projection + DBSCAN filtering.
    
    Pipeline:
    1. Render current view matching the user's viewport
    2. Generate SAM-2 mask from click point (semantic object selection)
    3. Project all Gaussian means to 2D using actual camera matrices
    4. Select Gaussians whose projections fall inside SAM mask
    5. DBSCAN 3D clustering to remove outlier Gaussians
    6. Save extracted PLY and load as independent sub-splat for 3D viewing
    """
    import uuid
    import torch
    import fvdb
    from pathlib import Path
    
    job_id = str(uuid.uuid4())[:8]
    
    try:
        if gsplat is None:
            return {"status": "error", "error": "No model loaded", "job_id": job_id}
        
        # 1. Render current view (same as what user sees)
        img = render_view(width, height, azimuth, elevation, zoom, cam_idx)
        if img is None:
            return {"status": "error", "error": "Render failed", "job_id": job_id}
        
        # 2. Generate SAM-2 mask from click point
        mask = None
        mask_method = "fallback"
        if not sam2_loaded:
            load_sam2()
        
        if sam2_loaded and sam2_predictor is not None:
            try:
                sam2_predictor.set_image(img)
                point_coords = np.array([[x, y]])
                point_labels = np.array([1])
                masks, scores, _ = sam2_predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=True
                )
                best_idx = np.argmax(scores)
                mask = masks[best_idx].astype(np.uint8)
                mask_method = "sam2"
                logger.info(f"SAM-2 mask: {np.sum(mask > 0)} pixels, score={scores[best_idx]:.3f}")
            except Exception as e:
                logger.warning(f"SAM-2 mask generation failed: {e}")
        
        # Fallback: circular mask if SAM-2 unavailable
        if mask is None:
            mask = np.zeros((height, width), dtype=np.uint8)
            radius = int(60 * max(scale_level, 0.3))
            yy, xx = np.ogrid[:height, :width]
            circle = (xx - x)**2 + (yy - y)**2 <= radius**2
            mask[circle] = 1
            logger.info(f"Fallback circular mask, radius={radius}")
        
        # 3. Get camera matrices matching the current view (same as render_view)
        w2c, K = get_camera_matrices(width, height, azimuth, elevation, zoom, cam_idx)
        if w2c is None:
            return {"status": "error", "error": "Camera matrices unavailable", "job_id": job_id}
        
        # 4. Project all Gaussian means to 2D using actual camera matrices
        means = gsplat.means  # [N, 3] on GPU
        N = means.shape[0]
        
        ones = torch.ones(N, 1, device=device, dtype=means.dtype)
        means_h = torch.cat([means, ones], dim=1)  # [N, 4]
        
        # World to camera transform
        p_cam = (w2c[0] @ means_h.T).T  # [N, 4]
        z_vals = p_cam[:, 2]
        
        # Project to 2D pixel coordinates using K matrix
        p_2d = (K[0] @ p_cam[:, :3].T).T  # [N, 3]
        px = (p_2d[:, 0] / (p_2d[:, 2] + 1e-8)).cpu().numpy()
        py = (p_2d[:, 1] / (p_2d[:, 2] + 1e-8)).cpu().numpy()
        z_np = z_vals.cpu().numpy()
        
        # 5. Select Gaussians whose projections fall inside the SAM mask
        valid = (z_np > 0.01) & np.isfinite(px) & np.isfinite(py)
        in_bounds = valid & (px >= 0) & (px < width) & (py >= 0) & (py < height)
        
        # Vectorized mask lookup — only cast valid in_bounds pixels to int
        # NaN/inf values would produce garbage int indices, so zero them first
        px_safe = np.where(in_bounds, px, 0)
        py_safe = np.where(in_bounds, py, 0)
        px_int = np.clip(px_safe.astype(np.int32), 0, width - 1)
        py_int = np.clip(py_safe.astype(np.int32), 0, height - 1)
        in_mask = in_bounds & (mask[py_int, px_int] > 0)
        
        selected_indices = np.where(in_mask)[0]
        logger.info(f"Projection selected {len(selected_indices)} / {N} Gaussians")
        
        if len(selected_indices) == 0:
            return {"status": "no_selection", "message": "No gaussians found at click position", "job_id": job_id}
        
        # Cap selection to prevent OOM on very large models
        MAX_EXTRACTION_GAUSSIANS = 200000
        if len(selected_indices) > MAX_EXTRACTION_GAUSSIANS:
            logger.warning(f"Selection too large ({len(selected_indices)}), sampling down to {MAX_EXTRACTION_GAUSSIANS}")
            rng = np.random.default_rng(42)
            selected_indices = rng.choice(selected_indices, MAX_EXTRACTION_GAUSSIANS, replace=False)
            selected_indices.sort()
        
        # 6. DBSCAN 3D clustering to remove outlier Gaussians
        if len(selected_indices) > 10:
            positions_sel = means[selected_indices].cpu().numpy()
            try:
                from sklearn.cluster import DBSCAN
                # Adaptive eps: use interquartile range of distances from centroid
                centroid = positions_sel.mean(axis=0)
                dists = np.linalg.norm(positions_sel - centroid, axis=1)
                eps = np.percentile(dists, 75) * 0.3 * max(scale_level, 0.2)
                eps = max(eps, 0.005)
                
                # Subsample for DBSCAN if too many points (prevent OOM)
                DBSCAN_MAX = 50000
                if len(positions_sel) > DBSCAN_MAX:
                    logger.info(f"Subsampling {len(positions_sel)} -> {DBSCAN_MAX} for DBSCAN")
                    rng = np.random.default_rng(42)
                    subsample_idx = rng.choice(len(positions_sel), DBSCAN_MAX, replace=False)
                    clustering = DBSCAN(eps=eps, min_samples=5).fit(positions_sel[subsample_idx])
                    # Assign remaining points to nearest cluster via centroid distance
                    sub_labels = clustering.labels_
                    unique_labels = set(sub_labels) - {-1}
                    if unique_labels:
                        largest = max(unique_labels, key=lambda l: np.sum(sub_labels == l))
                        cluster_pts = positions_sel[subsample_idx][sub_labels == largest]
                        cluster_centroid = cluster_pts.mean(axis=0)
                        cluster_radius = np.percentile(np.linalg.norm(cluster_pts - cluster_centroid, axis=1), 95) * 1.2
                        all_dists = np.linalg.norm(positions_sel - cluster_centroid, axis=1)
                        cluster_mask = all_dists <= cluster_radius
                        selected_indices = selected_indices[cluster_mask]
                        logger.info(f"DBSCAN (subsampled) filtered to {len(selected_indices)} Gaussians")
                else:
                    clustering = DBSCAN(eps=eps, min_samples=5).fit(positions_sel)
                    labels = clustering.labels_
                    unique_labels = set(labels) - {-1}
                    
                    if unique_labels:
                        largest = max(unique_labels, key=lambda l: np.sum(labels == l))
                        cluster_mask = labels == largest
                        selected_indices = selected_indices[cluster_mask]
                        logger.info(f"DBSCAN filtered to {len(selected_indices)} Gaussians (largest cluster)")
            except ImportError:
                # Fallback: median absolute deviation outlier removal
                median_pos = np.median(positions_sel, axis=0)
                dists = np.linalg.norm(positions_sel - median_pos, axis=1)
                threshold = np.median(dists) * (1.0 + scale_level * 2)
                inlier_mask = dists <= threshold
                selected_indices = selected_indices[inlier_mask]
                logger.info(f"MAD filtered to {len(selected_indices)} Gaussians")
        
        if len(selected_indices) == 0:
            return {"status": "no_selection", "message": "All Gaussians filtered as outliers", "job_id": job_id}
        
        # 7. Save extracted PLY as independent sub-splat
        model_path = None
        for ext in ['.ply', '']:
            test_path = MODEL_DIR / f"{model_name_param}{ext}"
            if test_path.exists():
                model_path = test_path
                break
        
        if model_path is None:
            return {"status": "error", "error": f"Model not found: {model_name_param}", "job_id": job_id}
        
        from plyfile import PlyData, PlyElement
        plydata = PlyData.read(str(model_path))
        vertex = plydata['vertex']
        new_data = vertex.data[selected_indices]
        new_element = PlyElement.describe(new_data, 'vertex')
        
        output_path = MODEL_DIR / f"_extraction_{job_id}.ply"
        PlyData([new_element]).write(str(output_path))
        logger.info(f"Saved extracted PLY: {output_path} ({len(selected_indices)} Gaussians)")
        
        # 8. Load as independent GaussianSplat3d for 3D viewing
        sub_gsplat, sub_metadata = fvdb.GaussianSplat3d.from_ply(str(output_path), device=device)
        
        # Cache for viewing (includes extraction camera for sharp initial rendering)
        extraction_cache[job_id] = {
            'indices': selected_indices.tolist(),
            'model_name': model_name_param,
            'positions': means[selected_indices].cpu().numpy().tolist(),
            'sub_gsplat': sub_gsplat,
            'sub_metadata': sub_metadata,
            'output_path': str(output_path),
            'cam_idx': cam_idx,
            'azimuth': azimuth,
            'elevation': elevation,
            'zoom': zoom,
        }
        
        return {
            "status": "completed",
            "job_id": job_id,
            "num_gaussians": int(len(selected_indices)),
            "model_name": model_name_param,
            "click": {"x": x, "y": y},
            "mask_method": mask_method,
            "message": f"Extracted {len(selected_indices)} Gaussians using {mask_method} mask"
        }
        
    except Exception as e:
        logger.error(f"Local extraction error: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e), "job_id": job_id}


@app.get("/garfield/download/{job_id}")
async def garfield_download(job_id: str):
    """Download extracted PLY file from local cache"""
    if job_id in extraction_cache:
        output_path = Path(extraction_cache[job_id].get('output_path', ''))
        if output_path.exists():
            from fastapi.responses import FileResponse
            return FileResponse(
                path=str(output_path),
                filename=f"extracted_{job_id}.ply",
                media_type="application/octet-stream"
            )
    return JSONResponse({"error": "Extraction not found"}, status_code=404)


# Store extraction data for viewing
extraction_cache = {}


@app.post("/garfield/clear")
async def garfield_clear():
    """Clear all extraction cache and delete temp PLY files"""
    global extraction_cache
    deleted = 0
    for job_id, data in list(extraction_cache.items()):
        output_path = data.get('output_path')
        if output_path:
            p = Path(output_path)
            if p.exists():
                p.unlink()
                deleted += 1
    extraction_cache = {}
    # Also clean any orphaned extraction files
    for f in MODEL_DIR.glob("_extraction_*.ply"):
        f.unlink()
        deleted += 1
    logger.info(f"Cleared extraction cache, deleted {deleted} temp files")
    return {"status": "ok", "deleted": deleted}


@app.get("/garfield/render_extraction")
async def render_extraction_view(
    job_id: str = Query(...),
    azimuth: float = Query(0),
    elevation: float = Query(0),
    zoom: float = Query(0.5),
    width: int = Query(1024),
    height: int = Query(768)
):
    """Render only the extracted sub-splat with orbit rotation"""
    if job_id not in extraction_cache:
        return Response(content=b"Extraction not in cache", status_code=404)
    
    img = render_extracted_gaussians(job_id, width, height, azimuth, elevation, zoom)
    
    if img is None:
        return Response(content=b"Render failed", status_code=500)
    
    from PIL import Image
    pil_img = Image.fromarray(img)
    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG')
    buffer.seek(0)
    
    return Response(content=buffer.getvalue(), media_type="image/png")


def render_extracted_gaussians(job_id, width, height, azimuth, elevation, zoom):
    """Render extracted object by orbiting around extraction centroid and cropping.
    
    Strategy:
    1. Start from extraction camera, apply user orbit around extraction centroid
    2. Render the full scene from the orbited camera
    3. Project extracted Gaussian means to 2D → bounding box
    4. Crop the full-scene render to the extraction bbox
    5. Resize crop to fill the output viewport
    """
    import torch
    from PIL import Image as PILImage
    
    if job_id not in extraction_cache:
        return None
    
    ext_data = extraction_cache[job_id]
    indices = ext_data.get('indices', [])
    
    if gsplat is None or model_metadata is None or not indices:
        return None
    
    try:
        ext_cam_idx = ext_data.get('cam_idx', 0)
        ext_azimuth = ext_data.get('azimuth', 0)
        ext_elevation = ext_data.get('elevation', 0)
        ext_zoom_orig = ext_data.get('zoom', 1.0)
        
        c2w_all = model_metadata.get('camera_to_world_matrices')
        K_all = model_metadata.get('projection_matrices')
        sizes = model_metadata.get('image_sizes')
        
        if c2w_all is None or K_all is None:
            return None
        
        num_cams = c2w_all.shape[0]
        ext_cam_idx = ext_cam_idx % num_cams
        
        # Get extraction centroid and compute camera orbit
        idx_tensor = torch.tensor(indices, device=device, dtype=torch.long)
        ext_means = gsplat.means[idx_tensor]
        centroid = ext_means.mean(dim=0)
        
        # Start from extraction camera, recreating the user's original viewpoint
        c2w = c2w_all[ext_cam_idx].to(device).clone()
        scene_center = gsplat.means.mean(dim=0)
        
        # Apply extraction orbit around scene center (matching render_view)
        if ext_azimuth != 0 or ext_elevation != 0:
            az_rad = math.radians(ext_azimuth)
            el_rad = math.radians(ext_elevation)
            Ry = torch.tensor([[math.cos(az_rad),0,math.sin(az_rad),0],[0,1,0,0],[-math.sin(az_rad),0,math.cos(az_rad),0],[0,0,0,1]], device=device, dtype=torch.float32)
            Rx = torch.tensor([[1,0,0,0],[0,math.cos(el_rad),-math.sin(el_rad),0],[0,math.sin(el_rad),math.cos(el_rad),0],[0,0,0,1]], device=device, dtype=torch.float32)
            T_to = torch.eye(4, device=device, dtype=torch.float32); T_to[:3,3] = -scene_center
            T_back = torch.eye(4, device=device, dtype=torch.float32); T_back[:3,3] = scene_center
            c2w = (T_back @ Ry @ Rx @ T_to) @ c2w
        
        # Apply extraction zoom
        cam_pos = c2w[:3, 3].clone()
        vd = scene_center - cam_pos
        vd = vd / (vd.norm() + 1e-8)
        dist_val = (scene_center - cam_pos).norm().item()
        c2w[:3, 3] = scene_center - vd * (dist_val / max(ext_zoom_orig, 0.1))
        
        # Apply user orbit rotation around extraction CENTROID
        if azimuth != 0 or elevation != 0:
            az_rad = math.radians(azimuth)
            el_rad = math.radians(elevation)
            Ry = torch.tensor([[math.cos(az_rad),0,math.sin(az_rad),0],[0,1,0,0],[-math.sin(az_rad),0,math.cos(az_rad),0],[0,0,0,1]], device=device, dtype=torch.float32)
            Rx = torch.tensor([[1,0,0,0],[0,math.cos(el_rad),-math.sin(el_rad),0],[0,math.sin(el_rad),math.cos(el_rad),0],[0,0,0,1]], device=device, dtype=torch.float32)
            T_to = torch.eye(4, device=device, dtype=torch.float32); T_to[:3,3] = -centroid
            T_back = torch.eye(4, device=device, dtype=torch.float32); T_back[:3,3] = centroid
            c2w = (T_back @ Ry @ Rx @ T_to) @ c2w
        
        # Render full scene at 2x resolution from orbited camera
        render_w = width * 2
        render_h = height * 2
        
        c2w_batch = c2w.unsqueeze(0).contiguous()
        w2c = torch.inverse(c2w_batch).contiguous()
        
        orig_h, orig_w = sizes[ext_cam_idx].tolist()
        K = K_all[ext_cam_idx:ext_cam_idx+1].to(device).clone()
        K[:, 0, :] *= render_w / orig_w
        K[:, 1, :] *= render_h / orig_h
        K = K.contiguous()
        
        full_images, _ = gsplat.render_images(
            world_to_camera_matrices=w2c,
            projection_matrices=K,
            image_width=render_w,
            image_height=render_h,
            near=0.01,
            far=200.0
        )
        full_img = full_images[0].clamp(0, 1).cpu().numpy()
        if full_img.shape[-1] > 3:
            full_img = full_img[..., :3]
        full_img = (full_img * 255).astype(np.uint8)
        
        # Project extracted Gaussian means to 2D to find bounding box
        ones = torch.ones(len(indices), 1, device=device, dtype=ext_means.dtype)
        means_h = torch.cat([ext_means, ones], dim=1)
        p_cam = (w2c[0] @ means_h.T).T
        z_vals = p_cam[:, 2]
        valid = z_vals > 0.01
        
        K_mat = K[0]
        p_2d = (K_mat @ p_cam[valid, :3].T).T
        px = (p_2d[:, 0] / (p_2d[:, 2] + 1e-8)).cpu().numpy()
        py = (p_2d[:, 1] / (p_2d[:, 2] + 1e-8)).cpu().numpy()
        
        # Filter to in-bounds projections
        in_bounds = (px >= 0) & (px < render_w) & (py >= 0) & (py < render_h)
        px = px[in_bounds]
        py = py[in_bounds]
        
        if len(px) < 5:
            logger.warning("Too few extraction points visible from this camera")
            return None
        
        # Compute bounding box with padding
        pad = max(render_w, render_h) * 0.05
        x_min = max(0, int(np.percentile(px, 2) - pad))
        x_max = min(render_w, int(np.percentile(px, 98) + pad))
        y_min = max(0, int(np.percentile(py, 2) - pad))
        y_max = min(render_h, int(np.percentile(py, 98) + pad))
        
        # Make bbox square (centered) for uniform zoom
        cx_box = (x_min + x_max) / 2
        cy_box = (y_min + y_max) / 2
        side = max(x_max - x_min, y_max - y_min)
        side = side / max(zoom, 0.1)
        
        x_min = max(0, int(cx_box - side / 2))
        y_min = max(0, int(cy_box - side / 2))
        x_max = min(render_w, int(cx_box + side / 2))
        y_max = min(render_h, int(cy_box + side / 2))
        
        if x_max - x_min < 10 or y_max - y_min < 10:
            return None
        
        # Crop and resize to output dimensions
        cropped = full_img[y_min:y_max, x_min:x_max]
        pil_crop = PILImage.fromarray(cropped)
        pil_crop = pil_crop.resize((width, height), PILImage.LANCZOS)
        img = np.array(pil_crop)
        
        # Golden border to indicate extraction view mode
        border = 3
        img[:border, :] = [255, 200, 0]
        img[-border:, :] = [255, 200, 0]
        img[:, :border] = [255, 200, 0]
        img[:, -border:] = [255, 200, 0]
        
        return img
        
    except Exception as e:
        logger.error(f"Extraction render error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=VIEWER_PORT)
