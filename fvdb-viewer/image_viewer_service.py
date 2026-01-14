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
from fastapi import FastAPI, Query, Form
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

def get_available_models():
    """Get list of available PLY models"""
    return sorted([m.name for m in MODEL_DIR.glob("*.ply")])


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


# CLIP model for object classification
clip_model = None
clip_preprocess = None
clip_loaded = False

OBJECT_LABELS = [
    "bulldozer", "excavator", "toy truck", "construction vehicle", 
    "plant", "potted plant", "foliage", "leaves",
    "table", "wooden table", "desk",
    "bowl", "cup", "mug", "container",
    "bottle", "spray bottle", "jar",
    "book", "box", "package",
    "chair", "bench", "furniture",
    "person", "hand", "arm",
    "floor", "wall", "background",
    "toy", "figurine", "decoration"
]

def load_clip():
    """Load CLIP model for object classification"""
    global clip_model, clip_preprocess, clip_loaded
    
    if clip_loaded:
        return True
    
    try:
        import clip
        import torch
        
        logger.info("Loading CLIP model...")
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
        clip_loaded = True
        logger.info("CLIP loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to load CLIP: {e}")
        return False


def clip_classify(image: np.ndarray, mask: np.ndarray) -> str:
    """Use CLIP to classify object in masked region"""
    if not clip_loaded:
        if not load_clip():
            return None
    
    try:
        import clip
        import torch
        from PIL import Image
        
        # Get bounding box
        bool_mask = mask.astype(bool) if mask.dtype != bool else mask
        coords = np.where(bool_mask)
        if len(coords[0]) == 0:
            return None
        
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        
        # Add padding
        pad = 10
        y_min = max(0, y_min - pad)
        x_min = max(0, x_min - pad)
        y_max = min(image.shape[0], y_max + pad)
        x_max = min(image.shape[1], x_max + pad)
        
        # Crop region
        crop = image[y_min:y_max, x_min:x_max]
        pil_img = Image.fromarray(crop)
        
        # Prepare for CLIP
        img_input = clip_preprocess(pil_img).unsqueeze(0).to(device)
        text_inputs = clip.tokenize(OBJECT_LABELS).to(device)
        
        # Get predictions
        with torch.no_grad():
            image_features = clip_model.encode_image(img_input)
            text_features = clip_model.encode_text(text_inputs)
            
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            best_idx = similarity[0].argmax().item()
            confidence = similarity[0][best_idx].item()
        
        if confidence > 0.1:
            return OBJECT_LABELS[best_idx].title()
        return None
        
    except Exception as e:
        logger.error(f"CLIP classification failed: {e}")
        return None


# Auto-labeling using CLIP and heuristics
def auto_label_segment(image: np.ndarray, mask: np.ndarray) -> str:
    """Generate automatic label for a segment using CLIP, fallback to heuristics"""
    import cv2
    
    # Try CLIP classification first
    clip_label = clip_classify(image, mask)
    if clip_label:
        return clip_label
    
    # Fallback to heuristics
    # Convert mask to boolean if needed
    bool_mask = mask.astype(bool) if mask.dtype != bool else mask
    
    # Get bounding box
    coords = np.where(bool_mask)
    if len(coords[0]) == 0:
        return "Unknown"
    
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()
    
    # Calculate features
    height = y_max - y_min
    width = x_max - x_min
    area = np.sum(mask)
    aspect_ratio = width / max(height, 1)
    
    # Get dominant color in masked region
    roi = image[y_min:y_max, x_min:x_max]
    roi_mask = bool_mask[y_min:y_max, x_min:x_max]
    if roi_mask.any():
        mean_color = roi[roi_mask].mean(axis=0)
    else:
        mean_color = [128, 128, 128]
    
    # Position in image (relative)
    img_h, img_w = image.shape[:2]
    center_y = (y_min + y_max) / 2 / img_h
    center_x = (x_min + x_max) / 2 / img_w
    relative_size = area / (img_h * img_w)
    
    # RGB values
    r, g, b = mean_color[0], mean_color[1], mean_color[2]
    
    # Yellow/orange detection for construction vehicles
    is_yellow = (r > 180 and g > 140 and b < 120)
    is_orange = (r > 200 and g > 100 and g < 180 and b < 100)
    
    # Improved heuristic labeling
    if relative_size > 0.3:
        return "Background"
    elif relative_size > 0.12 and center_y > 0.5:
        return "Table/Surface"
    elif is_yellow or is_orange:
        if 0.005 < relative_size < 0.1 and aspect_ratio > 0.8:
            return "Bulldozer/Vehicle"
        return "Yellow Object"
    elif g > r and g > b:
        return "Plant/Foliage"
    elif r > 150 and g < 100 and b < 100:
        return "Red Object"
    elif b > 150 and r < 100 and g < 150:
        return "Blue Object"
    elif r > 180 and g > 150 and b > 150:
        return "Bowl/Container"
    elif center_y > 0.7 and relative_size < 0.02:
        return "Small Item"
    elif aspect_ratio > 2.5:
        return "Elongated Object"
    elif aspect_ratio < 0.4:
        return "Tall Object"
    else:
        return "Object"


def segment_image(image: np.ndarray, points: List[Dict] = None, auto_mode: bool = True):
    """Segment objects in image using SAM-2 Automatic Mask Generator"""
    global current_segments, segment_labels
    
    if not sam2_loaded:
        if not load_sam2():
            return None
    
    try:
        import torch
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        
        masks_data = []
        scores_data = []
        
        if auto_mode:
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
        
        current_segments = {
            "masks": masks_data,
            "scores": scores_data,
            "num_segments": len(masks_data),
            "labels": labels
        }
        
        return current_segments
        
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
            <div id="segStatus" style="margin-top:10px;font-size:12px;color:#888;">
                Click "Auto Segment" to detect objects
            </div>
            <div id="labels-list"></div>
        </div>
        
        <div id="hover-tooltip"></div>
        
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
    if "masks" not in current_segments:
        current_segments = {"masks": [], "scores": [], "num_segments": 0}
    
    for mask, score in zip(segments["masks"], segments["scores"]):
        idx = len(current_segments["masks"])
        current_segments["masks"].append(mask)
        current_segments["scores"].append(score)
        current_segments["num_segments"] += 1
        segment_labels[idx] = f"Object {idx}"
    
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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=VIEWER_PORT)
