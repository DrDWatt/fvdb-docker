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

# RAG metadata from model service
rag_metadata: Dict[str, Any] = {}
rag_labels: List[str] = []
MODEL_SERVICE_URL = os.environ.get("MODEL_SERVICE_URL", "http://rendering-service:8001")

def get_available_models():
    """Get list of available PLY models"""
    return sorted([m.name for m in MODEL_DIR.glob("*.ply")])


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
                <label>Zoom: <span id="zoom-val">0.5</span>x</label>
                <input type="range" id="zoom" min="0.1" max="2" value="0.5" step="0.1">
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
            <div style="margin-top:8px;">
                <button class="seg-btn" style="background:#6c757d;color:white;" onclick="showSummary()">📄 Summary</button>
                <button class="seg-btn" style="background:#17a2b8;color:white;" onclick="showUploadModal()">⬆️ Upload Info</button>
            </div>
            <div id="segStatus" style="margin-top:10px;font-size:12px;color:#888;">
                Click "Auto Segment" to detect objects<br>
                <span style="color:#00d4ff;">Double-click objects to view/edit summaries</span>
            </div>
            <div id="labels-list"></div>
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
            <div style="margin-top:10px;">
                <label style="font-size:12px;">Scale Level: <span id="scale-val">0.5</span></label>
                <input type="range" id="extract-scale" class="scale-slider" min="0.1" max="2.0" value="0.5" step="0.1" oninput="document.getElementById('scale-val').textContent = this.value">
            </div>
            <div id="extractStatus" style="margin-top:8px;font-size:12px;color:#888;">
                Click to extract 3D assets from scene
            </div>
            <div id="extraction-list"></div>
        </div>
        
        <!-- Summary Modal -->
        <div id="summary-modal" style="display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.8);z-index:2000;">
            <div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);background:#1a1a2e;padding:30px;border-radius:12px;max-width:600px;max-height:80vh;overflow-y:auto;border:2px solid #00d4ff;">
                <h2 style="color:#00d4ff;margin-top:0;">📄 Model Summary</h2>
                <div id="summary-content" style="color:#eee;line-height:1.6;white-space:pre-wrap;"></div>
                <button onclick="closeSummary()" style="margin-top:20px;padding:10px 25px;background:#dc3545;color:white;border:none;border-radius:5px;cursor:pointer;">Close</button>
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
                    zoomSlider.value = 0.5;
                    camSlider.value = 0;
                    
                    // Update info and render
                    const info = await fetch('/info').then(r => r.json());
                    document.getElementById('num-gs').textContent = info.num_gaussians || 'N/A';
                    updateRender();
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
                        img.style.display = 'block';
                        loading.style.display = 'none';
                    }}
                }} catch(e) {{
                    console.error('Render error:', e);
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
            
            // Summary and Upload functions
            async function showSummary() {{
                const modelSelect = document.getElementById('model-select');
                const modelName = modelSelect.value;
                
                document.getElementById('summary-content').textContent = 'Loading...';
                document.getElementById('summary-modal').style.display = 'block';
                
                try {{
                    const response = await fetch(`/model_summary?model=${{encodeURIComponent(modelName)}}`);
                    const data = await response.json();
                    
                    if (data.has_summary) {{
                        document.getElementById('summary-content').textContent = data.summary;
                    }} else {{
                        document.getElementById('summary-content').innerHTML = `
                            <p style="color:#ffc107;">No summary available for this model.</p>
                            <p>Click "Upload Info" to add a summary document (PDF, TXT, JSON, or MD).</p>
                        `;
                    }}
                }} catch(e) {{
                    document.getElementById('summary-content').textContent = 'Error loading summary: ' + e.message;
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
                formData.append('width', 1024);
                formData.append('height', 768);
                
                try {{
                    const response = await fetch('/garfield/extract', {{
                        method: 'POST',
                        body: formData
                    }});
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
                    document.getElementById('extractStatus').textContent = '❌ Error: ' + e.message;
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
            
            // Keyboard controls for extraction view (more reliable than trackpad)
            document.addEventListener('keydown', (e) => {{
                if (!viewingExtraction) return;
                
                let changed = false;
                const rotateStep = 10;
                const zoomStep = 0.1;
                
                switch(e.key) {{
                    case 'ArrowLeft':
                        extAzimuth -= rotateStep;
                        changed = true;
                        break;
                    case 'ArrowRight':
                        extAzimuth += rotateStep;
                        changed = true;
                        break;
                    case 'ArrowUp':
                        extElevation = Math.min(89, extElevation + rotateStep);
                        changed = true;
                        break;
                    case 'ArrowDown':
                        extElevation = Math.max(-89, extElevation - rotateStep);
                        changed = true;
                        break;
                    case '+':
                    case '=':
                        extZoom = Math.min(5, extZoom + zoomStep);
                        changed = true;
                        break;
                    case '-':
                    case '_':
                        extZoom = Math.max(0.2, extZoom - zoomStep);
                        changed = true;
                        break;
                    case 'Escape':
                        exitExtractionView();
                        return;
                }}
                
                if (changed) {{
                    e.preventDefault();
                    if (!renderPending) {{
                        renderPending = true;
                        renderExtractionOverlay().finally(() => {{ renderPending = false; }});
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


@app.post("/object_summary/{segment_idx}/training")
async def link_training_data(
    segment_idx: int,
    training_job_id: str = Form(None),
    model_path: str = Form(None)
):
    """Link training data to a specific segmented object"""
    global object_summaries
    
    if segment_idx not in object_summaries:
        object_summaries[segment_idx] = {"text": "", "files": [], "training_data": {}}
    
    object_summaries[segment_idx]["training_data"] = {
        "job_id": training_job_id,
        "model_path": model_path,
        "linked_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    logger.info(f"Linked training data for object {segment_idx}: job={training_job_id}")
    return {"status": "ok", "segment_idx": segment_idx}


# ===== GARField 3D Extraction Endpoints =====

GARFIELD_SERVICE_URL = os.environ.get("GARFIELD_SERVICE_URL", "http://garfield-extraction:8006")


@app.post("/garfield/extract")
async def garfield_extract(
    x: int = Form(...),
    y: int = Form(...),
    model_name: str = Form(...),
    scale_level: float = Form(0.5),
    azimuth: float = Form(0),
    elevation: float = Form(0),
    zoom: float = Form(1.0),
    width: int = Form(1024),
    height: int = Form(768)
):
    """Proxy extraction request to GARField service"""
    try:
        import requests
        
        # Render current view to send with extraction request
        img = render_view(width, height, azimuth, elevation, zoom, 0)
        img_bytes = None
        if img is not None:
            from PIL import Image
            pil_img = Image.fromarray(img)
            buffer = io.BytesIO()
            pil_img.save(buffer, format='PNG')
            img_bytes = buffer.getvalue()
        
        # Send to GARField service
        data = {
            'x': x,
            'y': y,
            'model_name': model_name,
            'scale_level': scale_level,
            'azimuth': azimuth,
            'elevation': elevation,
            'zoom': zoom,
            'width': width,
            'height': height
        }
        
        files = {}
        if img_bytes:
            files['image'] = ('render.png', img_bytes, 'image/png')
        
        response = requests.post(
            f"{GARFIELD_SERVICE_URL}/extract",
            data=data,
            files=files if files else None,
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()
        return {"status": "error", "error": f"GARField service error: {response.status_code}"}
    except requests.exceptions.ConnectionError:
        # Fallback: perform local extraction if GARField service unavailable
        return await local_garfield_extract(x, y, model_name, scale_level, azimuth, elevation, zoom, width, height)
    except Exception as e:
        logger.error(f"GARField extraction error: {e}")
        return {"status": "error", "error": str(e)}


async def local_garfield_extract(x, y, model_name, scale_level, azimuth, elevation, zoom, width, height):
    """Local fallback for GARField extraction when service is unavailable"""
    import uuid
    from pathlib import Path
    
    job_id = str(uuid.uuid4())[:8]
    
    try:
        # Find model file
        model_path = None
        for ext in ['.ply', '']:
            test_path = MODEL_DIR / f"{model_name}{ext}"
            if test_path.exists():
                model_path = test_path
                break
        
        if not model_path:
            return {"status": "error", "error": f"Model not found: {model_name}", "job_id": job_id}
        
        # Simple extraction based on click position projection
        from plyfile import PlyData
        plydata = PlyData.read(str(model_path))
        vertex = plydata['vertex']
        
        positions = np.stack([
            np.array(vertex['x']),
            np.array(vertex['y']),
            np.array(vertex['z'])
        ], axis=-1)
        
        # Project points and find those near click
        focal = width / (2 * np.tan(np.radians(30)))
        cx, cy = width / 2, height / 2
        
        az_rad = np.radians(azimuth)
        el_rad = np.radians(elevation)
        
        rotated = positions.copy()
        cos_az, sin_az = np.cos(az_rad), np.sin(az_rad)
        temp_x = rotated[:, 0] * cos_az + rotated[:, 2] * sin_az
        temp_z = -rotated[:, 0] * sin_az + rotated[:, 2] * cos_az
        rotated[:, 0] = temp_x
        rotated[:, 2] = temp_z
        
        cos_el, sin_el = np.cos(el_rad), np.sin(el_rad)
        temp_y = rotated[:, 1] * cos_el - rotated[:, 2] * sin_el
        temp_z = rotated[:, 1] * sin_el + rotated[:, 2] * cos_el
        rotated[:, 1] = temp_y
        rotated[:, 2] = temp_z
        
        z_vals = rotated[:, 2]
        valid = z_vals > 0.1
        
        proj_x = np.where(valid, rotated[:, 0] / z_vals * focal + cx, -1)
        proj_y = np.where(valid, rotated[:, 1] / z_vals * focal + cy, -1)
        
        # Find points near click within radius based on scale
        radius = 50 * scale_level
        distances = np.sqrt((proj_x - x)**2 + (proj_y - y)**2)
        selected = (distances < radius) & valid
        
        num_selected = np.sum(selected)
        
        if num_selected == 0:
            return {"status": "no_selection", "message": "No gaussians at click position", "job_id": job_id}
        
        # Get selected indices and cache for viewing
        selected_indices = np.where(selected)[0].tolist()
        extraction_cache[job_id] = {
            'indices': selected_indices,
            'model_name': model_name,
            'positions': positions[selected].tolist()
        }
        
        return {
            "status": "completed",
            "job_id": job_id,
            "num_gaussians": int(num_selected),
            "model_name": model_name,
            "click": {"x": x, "y": y},
            "message": "Local extraction (GARField service unavailable)"
        }
        
    except Exception as e:
        logger.error(f"Local extraction error: {e}")
        return {"status": "error", "error": str(e), "job_id": job_id}


@app.get("/garfield/download/{job_id}")
async def garfield_download(job_id: str):
    """Proxy download request to GARField service"""
    try:
        import requests
        response = requests.get(f"{GARFIELD_SERVICE_URL}/download/{job_id}", timeout=30, stream=True)
        if response.status_code == 200:
            return Response(
                content=response.content,
                media_type="application/octet-stream",
                headers={"Content-Disposition": f"attachment; filename=extracted_{job_id}.ply"}
            )
        return JSONResponse({"error": "Download failed"}, status_code=response.status_code)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/garfield/status")
async def garfield_status():
    """Check GARField service status"""
    try:
        import requests
        response = requests.get(f"{GARFIELD_SERVICE_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return {"status": "unavailable", "message": "GARField service not connected"}


# Store extraction data for viewing
extraction_cache = {}


@app.get("/garfield/render_extraction")
async def render_extraction_view(
    job_id: str = Query(...),
    azimuth: float = Query(0),
    elevation: float = Query(0),
    zoom: float = Query(0.5),
    width: int = Query(1024),
    height: int = Query(768)
):
    """Render only the extracted gaussians with rotation"""
    global extraction_cache
    
    # Get extraction indices from cache or from last extraction
    if job_id not in extraction_cache:
        # For local extractions, we need to re-extract with saved params
        return Response(content=b"Extraction not in cache", status_code=404)
    
    ext_data = extraction_cache[job_id]
    indices = ext_data.get('indices', [])
    
    if not indices:
        return Response(content=b"No indices in extraction", status_code=400)
    
    # Render only the extracted gaussians
    img = render_extracted_gaussians(indices, width, height, azimuth, elevation, zoom)
    
    if img is None:
        return Response(content=b"Render failed", status_code=500)
    
    from PIL import Image
    pil_img = Image.fromarray(img)
    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG')
    buffer.seek(0)
    
    return Response(content=buffer.getvalue(), media_type="image/png")


def render_extracted_gaussians(indices, width, height, azimuth, elevation, zoom):
    """Render full scene with camera focused on extracted region centroid"""
    import torch
    import fvdb
    
    if gsplat is None or model_metadata is None:
        return None
    
    try:
        # Get extraction data to find centroid
        ext_data = None
        for job_id, data in extraction_cache.items():
            if data.get('indices') == indices:
                ext_data = data
                break
        
        if ext_data is None or 'positions' not in ext_data:
            return render_view(width, height, azimuth, elevation, zoom, 0)
        
        positions = np.array(ext_data['positions'])
        if len(positions) == 0:
            return render_view(width, height, azimuth, elevation, zoom, 0)
        
        # Calculate centroid of extracted region
        centroid = torch.tensor(positions.mean(axis=0), dtype=torch.float32, device=device)
        
        # Get camera matrices from metadata
        c2w_all = model_metadata.get('camera_to_world_matrices')
        K_all = model_metadata.get('projection_matrices')
        
        if c2w_all is None or K_all is None:
            return render_view(width, height, azimuth, elevation, zoom, 0)
        
        K = K_all[0].to(device).clone()
        
        # Apply rotation around centroid
        az_rad = math.radians(azimuth)
        el_rad = math.radians(elevation)
        
        cos_az, sin_az = math.cos(az_rad), math.sin(az_rad)
        cos_el, sin_el = math.cos(el_rad), math.sin(el_rad)
        
        # Rotation matrices (GPU accelerated via torch)
        rot_y = torch.tensor([
            [cos_az, 0, sin_az],
            [0, 1, 0],
            [-sin_az, 0, cos_az]
        ], dtype=torch.float32, device=device)
        
        rot_x = torch.tensor([
            [1, 0, 0],
            [0, cos_el, -sin_el],
            [0, sin_el, cos_el]
        ], dtype=torch.float32, device=device)
        
        rot = rot_x @ rot_y
        
        # Camera orbit around centroid
        cam_distance = 1.5 / max(zoom, 0.1)
        cam_offset = torch.tensor([0, 0, cam_distance], dtype=torch.float32, device=device)
        cam_pos = centroid + (rot.T @ cam_offset)
        
        # Look at centroid
        forward = centroid - cam_pos
        forward = forward / (forward.norm() + 1e-8)
        world_up = torch.tensor([0, 1, 0], dtype=torch.float32, device=device)
        right = torch.cross(forward, world_up)
        right = right / (right.norm() + 1e-8)
        up = torch.cross(right, forward)
        
        # Build camera-to-world matrix
        c2w = torch.eye(4, dtype=torch.float32, device=device)
        c2w[:3, 0] = right
        c2w[:3, 1] = up  
        c2w[:3, 2] = -forward
        c2w[:3, 3] = cam_pos
        
        # Render full scene with new camera (GPU accelerated via fVDB/CUDA)
        with torch.cuda.amp.autocast():  # Use mixed precision for TensorRT-like acceleration
            img_tensor = gsplat.render(c2w, K, width, height)
        
        img_np = (img_tensor.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        
        # Add border to indicate extraction view mode
        img_np[:2, :] = [255, 200, 0]
        img_np[-2:, :] = [255, 200, 0]
        img_np[:, :2] = [255, 200, 0]
        img_np[:, -2:] = [255, 200, 0]
        
        return img_np
        
    except Exception as e:
        logger.error(f"Extraction render error: {e}")
        import traceback
        traceback.print_exc()
        return render_view(width, height, azimuth, elevation, zoom, 0)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=VIEWER_PORT)
