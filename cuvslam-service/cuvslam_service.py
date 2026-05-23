"""
cuVSLAM Processing Service
Drop-in replacement for COLMAP service using NVIDIA cuVSLAM for visual SLAM.
Processes stereo frame pairs to produce COLMAP-compatible sparse reconstruction.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Tuple
from pathlib import Path
import logging
import shutil
import zipfile
import json
import os
import struct
import subprocess
import numpy as np
from datetime import datetime
import asyncio
from splatking_parser import is_splatking_zip, extract_splatking_images

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="cuVSLAM Processing Service",
    description="Visual SLAM processing for 3D reconstruction (replaces COLMAP)",
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

# Directories (same layout as COLMAP service for compatibility)
BASE_DIR = Path("/app")
UPLOAD_DIR = BASE_DIR / "uploads"
PROCESSING_DIR = BASE_DIR / "processing"
OUTPUT_DIR = BASE_DIR / "outputs"
TEMP_DIR = BASE_DIR / "temp"
WORKFLOW_STATE_FILE = BASE_DIR / "workflow_state.json"

for dir_path in [UPLOAD_DIR, PROCESSING_DIR, OUTPUT_DIR, TEMP_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# Check if cuVSLAM is available
CUVSLAM_AVAILABLE = False
try:
    import cuvslam as vslam
    CUVSLAM_AVAILABLE = True
    logger.info(f"cuVSLAM loaded: version {vslam.__version__}")
except ImportError as e:
    logger.warning(f"cuVSLAM not available: {e}")

# File-based workflow state (persists across requests)
def load_workflows() -> Dict:
    if WORKFLOW_STATE_FILE.exists():
        try:
            with open(WORKFLOW_STATE_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_workflows(workflows: Dict):
    with open(WORKFLOW_STATE_FILE, 'w') as f:
        json.dump(workflows, f)

def update_workflow(workflow_id: str, updates: Dict):
    workflows = load_workflows()
    if workflow_id in workflows:
        workflows[workflow_id].update(updates)
    else:
        workflows[workflow_id] = updates
    save_workflows(workflows)

# Job tracking
processing_jobs = {}


async def _poll_training_completion(workflow_id: str, training_job_id: str, training_url: str):
    """Poll training service until job completes or fails, updating workflow state."""
    import httpx
    max_wait = 43200  # 12 hours
    elapsed = 0
    async with httpx.AsyncClient(timeout=30.0) as client:
        while elapsed < max_wait:
            await asyncio.sleep(15)
            elapsed += 15
            try:
                resp = await client.get(f"{training_url}/jobs/{training_job_id}")
                if resp.status_code == 200:
                    tdata = resp.json()
                    tstatus = tdata.get("status", "")
                    tprogress = tdata.get("progress", 0)
                    tmessage = tdata.get("message", "")

                    update_workflow(workflow_id, {
                        "progress": 0.75 + tprogress * 0.20,
                        "current_step": f"Training: {tmessage}",
                        "trainingDetails": {
                            "progress": tprogress,
                            "message": tmessage
                        }
                    })

                    if tstatus == "completed":
                        update_workflow(workflow_id, {
                            "status": "completed",
                            "progress": 1.0,
                            "current_step": "Pipeline complete! View splat at :8085"
                        })
                        logger.info(f"[{workflow_id}] Training complete (poll)")
                        return
                    elif tstatus == "failed":
                        update_workflow(workflow_id, {
                            "status": "failed",
                            "progress": 0.8,
                            "current_step": f"Training failed: {tmessage}",
                            "error": f"Training failed: {tmessage}"
                        })
                        logger.error(f"[{workflow_id}] Training failed (poll)")
                        return
            except Exception as e:
                logger.warning(f"[{workflow_id}] Training poll error: {e}")

    logger.warning(f"[{workflow_id}] Training poll timed out after {max_wait}s")


@app.on_event("startup")
async def startup_resume_training_polls():
    """On startup, resume polling for any workflows stuck in 'training' status."""
    training_url = os.environ.get("TRAINING_SERVICE_URL", "http://fvdb-training-gpu:8000")
    workflows = load_workflows()
    for wid, w in workflows.items():
        if w.get("status") != "training":
            continue
        tjob = w.get("training_job_id")
        if not tjob:
            continue
        logger.info(f"Resuming training poll for {wid} (job {tjob})")
        asyncio.create_task(_poll_training_completion(wid, tjob, training_url))


# HEIC/HEIF support: register opener so PIL can read Apple HEIC images
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIF_AVAILABLE = True
    logger.info("HEIF/HEIC support available via pillow-heif")
except ImportError:
    HEIF_AVAILABLE = False
    logger.warning("pillow-heif not installed — HEIC images will not be supported")


def convert_heic_to_jpeg(image_data: bytes, output_path: Path) -> bool:
    """Convert HEIC/HEIF image bytes to JPEG. Returns True if conversion was needed and succeeded."""
    # Detect HEIC by 'ftypheic' or 'ftypheix' in header
    if len(image_data) < 12:
        return False
    header = image_data[4:12]
    if header[:4] not in (b'ftyp',):
        return False
    ftype = image_data[8:12]
    if ftype not in (b'heic', b'heix', b'hevc', b'hevx', b'heim', b'heis', b'mif1'):
        return False

    if not HEIF_AVAILABLE:
        logger.error(f"HEIC image detected but pillow-heif not installed: {output_path}")
        return False

    from PIL import Image
    import io
    img = Image.open(io.BytesIO(image_data))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img.save(str(output_path), 'JPEG', quality=95)
    return True


def save_image_as_jpeg(image_data: bytes, output_path: Path) -> bool:
    """Save image data to JPEG, converting from HEIC if needed. Returns True on success."""
    # Try HEIC conversion first
    if convert_heic_to_jpeg(image_data, output_path):
        return True
    # Already a standard format — write directly
    with open(output_path, "wb") as f:
        f.write(image_data)
    return True


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
            # Fall back to using all frames if filtering fails
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


def create_stereo_rig(width: int, height: int, baseline: float = 0.12,
                      fx: float = None, fy: float = None,
                      cx: float = None, cy: float = None):
    """Create a cuVSLAM stereo rig from camera parameters.
    
    Default intrinsics are typical ZED 2i factory calibration values.
    Reference resolution: HD720 (1280x720) with fx=fy≈528, cx≈636, cy≈362.
    """
    if fx is None:
        # ZED 2i typical intrinsics scaled from HD720 reference
        # HD720 reference: fx=527.6, fy=527.6, cx=636.4, cy=361.5
        scale_x = width / 1280.0
        scale_y = height / 720.0
        fx = 527.6 * scale_x
        fy = 527.6 * scale_y
        cx = 636.4 * scale_x
        cy = 361.5 * scale_y

    rig = vslam.Rig()

    # Left camera (reference)
    left_cam = vslam.Camera()
    left_cam.distortion = vslam.Distortion(vslam.Distortion.Model.Pinhole)
    left_cam.focal = (fx, fy)
    left_cam.principal = (cx, cy)
    left_cam.size = (width, height)
    # Identity pose (reference camera)
    left_cam.rig_from_camera = vslam.Pose(
        rotation=[0.0, 0.0, 0.0, 1.0],
        translation=[0.0, 0.0, 0.0]
    )

    # Right camera (offset by baseline along x-axis)
    right_cam = vslam.Camera()
    right_cam.distortion = vslam.Distortion(vslam.Distortion.Model.Pinhole)
    right_cam.focal = (fx, fy)
    right_cam.principal = (cx, cy)
    right_cam.size = (width, height)
    right_cam.rig_from_camera = vslam.Pose(
        rotation=[0.0, 0.0, 0.0, 1.0],
        translation=[baseline, 0.0, 0.0]
    )

    rig.cameras = [left_cam, right_cam]
    return rig, fx, fy, cx, cy


def pose_to_colmap_qtvec(pose):
    """Convert cuVSLAM world_from_rig Pose to COLMAP quaternion + translation.
    
    COLMAP uses world-from-camera (camera-to-world) convention but stores
    the INVERSE (camera extrinsics = world-to-camera) in images.txt.
    cuVSLAM gives world_from_rig (rig-to-world).
    """
    from scipy.spatial.transform import Rotation

    # cuVSLAM pose: world_from_rig
    quat_xyzw = np.array(pose.rotation)  # [x, y, z, w]
    translation = np.array(pose.translation)

    # Convert to rotation matrix (world_from_camera)
    rot_w2c_inv = Rotation.from_quat(quat_xyzw)

    # COLMAP stores camera-to-world inverse = world-to-camera
    rot_c2w = rot_w2c_inv.inv()
    quat_wxyz = rot_c2w.as_quat()[[3, 0, 1, 2]]  # COLMAP uses [w, x, y, z]
    t_c2w = -rot_c2w.apply(translation)

    return quat_wxyz, t_c2w


def read_pfm(path: Path) -> np.ndarray:
    """Read a PFM (Portable Float Map) depth file.
    
    Returns depth as a 2D float32 numpy array.
    """
    with open(path, 'rb') as f:
        header = f.readline().strip()
        dims = f.readline().strip().split()
        w, h = int(dims[0]), int(dims[1])
        scale = float(f.readline().strip())
        data = np.frombuffer(f.read(), dtype=np.float32).reshape(h, w)
        if scale < 0:
            data = np.flipud(data)
    return data


def generate_stereo_depth_points(left_dir: Path, right_dir: Path,
                                 image_names: List[str], poses: List,
                                 fx: float, fy: float, cx: float, cy: float,
                                 baseline: float = 0.12,
                                 depth_dir: Path = None,
                                 max_points_per_image: int = 500,
                                 sample_stride: int = 3):
    """Generate dense 3D points from stereo depth.
    
    Uses PFM depth maps from ZED camera if available (depth_dir), otherwise
    falls back to computing disparity from left/right image pairs via StereoSGBM.
    
    PFM depth maps from ZED provide far superior depth (99%+ valid pixels, real
    metric depth) compared to computed stereo disparity.
    
    Args:
        left_dir: Path to left images
        right_dir: Path to right images  
        image_names: List of image filenames
        poses: List of camera poses (world_from_rig)
        fx, fy, cx, cy: Camera intrinsics
        baseline: Stereo baseline in meters (used for StereoSGBM fallback)
        depth_dir: Path to PFM depth maps (if available from ZED)
        max_points_per_image: Max 3D points to sample per frame
        sample_stride: Only process every Nth frame to manage point count
        
    Returns:
        points3d: dict of {point3d_id: (X, Y, Z, R, G, B, error, [(image_id, pt2d_idx)])}
        image_points2d: dict of {image_id: [(x, y, point3d_id), ...]}
    """
    import cv2
    from scipy.spatial.transform import Rotation

    points3d = {}
    image_points2d = {}
    point3d_id = 1

    # Check if PFM depth maps are available
    has_pfm_depth = (depth_dir is not None and depth_dir.exists() and
                     len(list(depth_dir.glob("*.pfm"))) > 0)

    # Build list of valid frames
    valid_frames = []
    for i, (name, pose) in enumerate(zip(image_names, poses)):
        if pose is None:
            continue
        left_path = left_dir / name
        if not left_path.exists():
            continue

        if has_pfm_depth:
            # Match PFM by frame number
            pfm_name = name.rsplit('.', 1)[0] + '.pfm'
            pfm_path = depth_dir / pfm_name
            if pfm_path.exists():
                valid_frames.append((i + 1, pose, left_path, pfm_path))
        elif right_dir is not None and right_dir.exists():
            right_path = right_dir / name
            if right_path.exists():
                valid_frames.append((i + 1, pose, left_path, right_path))

    if len(valid_frames) == 0:
        logger.warning("No valid frames for depth point generation")
        return {}, {}

    # Configure StereoSGBM as fallback (only if no PFM depth)
    stereo = None
    if not has_pfm_depth:
        stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=128,
            blockSize=5,
            P1=8 * 3 * 5**2,
            P2=32 * 3 * 5**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32
        )

    frames_to_process = valid_frames[::sample_stride]
    logger.info(f"Generating depth points from {len(frames_to_process)} frames "
                f"(stride={sample_stride}, {len(valid_frames)} total valid, "
                f"source={'PFM depth maps' if has_pfm_depth else 'stereo disparity'})")

    for img_id, pose, left_path, depth_or_right_path in frames_to_process:
        left_img = cv2.imread(str(left_path))
        if left_img is None:
            continue

        h_img, w_img = left_img.shape[:2]

        if has_pfm_depth:
            # Use ZED PFM depth directly (metric depth in meters)
            depth_map = read_pfm(depth_or_right_path)
            # PFM may be full side-by-side resolution — use left half
            if depth_map.shape[1] > w_img:
                depth_map = depth_map[:, :w_img]
            # Resize depth to match left image if needed
            if depth_map.shape[0] != h_img or depth_map.shape[1] != w_img:
                depth_map = cv2.resize(depth_map, (w_img, h_img),
                                       interpolation=cv2.INTER_NEAREST)
            # Valid depth mask (ZED uses 0 or NaN/Inf for invalid)
            valid_mask = (depth_map > 0.1) & (depth_map < 20.0) & np.isfinite(depth_map)
        else:
            # Compute disparity from left/right pair
            right_img = cv2.imread(str(depth_or_right_path))
            if right_img is None:
                continue
            left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
            disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
            valid_mask = disparity > 0
            # Convert disparity to depth
            depth_map = np.where(valid_mask, fx * baseline / disparity, 0)
            valid_mask = (depth_map > 0.1) & (depth_map < 20.0)

        # Get valid pixel coordinates
        ys, xs = np.where(valid_mask)
        if len(xs) == 0:
            continue

        # Sub-sample to limit points
        if len(xs) > max_points_per_image:
            indices = np.random.choice(len(xs), max_points_per_image, replace=False)
            xs = xs[indices]
            ys = ys[indices]

        # Get depths at sampled pixels
        depths = depth_map[ys, xs]

        # Back-project to camera frame
        X_cam = (xs.astype(np.float64) - cx) * depths / fx
        Y_cam = (ys.astype(np.float64) - cy) * depths / fy
        Z_cam = depths

        # Transform to world frame using pose (world_from_rig)
        q_xyzw = np.array(pose.rotation)
        t = np.array(pose.translation)
        R_w = Rotation.from_quat(q_xyzw).as_matrix()

        pts_cam = np.stack([X_cam, Y_cam, Z_cam], axis=1)  # Nx3
        pts_world = (R_w @ pts_cam.T).T + t

        # Initialize image_points2d for this image
        if img_id not in image_points2d:
            image_points2d[img_id] = []

        # Add points
        for j in range(len(pts_world)):
            X, Y, Z = pts_world[j]
            px, py = int(xs[j]), int(ys[j])

            # Get color from left image
            if 0 <= py < left_img.shape[0] and 0 <= px < left_img.shape[1]:
                B, G, R = left_img[py, px]
            else:
                R, G, B = 128, 128, 128

            pt2d_idx = len(image_points2d[img_id])
            track = [(img_id, pt2d_idx)]
            points3d[point3d_id] = (X, Y, Z, int(R), int(G), int(B), 0.5, track)
            image_points2d[img_id].append((float(xs[j]), float(ys[j]), point3d_id))
            point3d_id += 1

    logger.info(f"Generated {len(points3d)} depth-based 3D points from "
                f"{len(frames_to_process)} frames")
    return points3d, image_points2d


def triangulate_sparse_points(left_dir: Path, image_names: List[str],
                              poses: List, fx: float, fy: float,
                              cx: float, cy: float):
    """Triangulate sparse 3D points from consecutive frame pairs using ORB features.
    
    Returns:
        points3d: dict of {point3d_id: (X, Y, Z, R, G, B, error, [(image_id, pt2d_idx)])}
        image_points2d: dict of {image_id: [(x, y, point3d_id), ...]}
    """
    import cv2
    from scipy.spatial.transform import Rotation

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    # Build list of valid (image_id, pose, image_path) tuples
    valid_frames = []
    for i, (name, pose) in enumerate(zip(image_names, poses)):
        if pose is None:
            continue
        img_path = left_dir / name
        if img_path.exists():
            valid_frames.append((i + 1, pose, img_path))

    if len(valid_frames) < 2:
        return {}, {}

    orb = cv2.ORB_create(nfeatures=1000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    points3d = {}
    image_points2d = {frame[0]: [] for frame in valid_frames}
    point3d_id = 1
    max_points_per_pair = 200

    for idx in range(len(valid_frames) - 1):
        img_id1, pose1, path1 = valid_frames[idx]
        img_id2, pose2, path2 = valid_frames[idx + 1]

        img1 = cv2.imread(str(path1), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(str(path2), cv2.IMREAD_GRAYSCALE)
        if img1 is None or img2 is None:
            continue

        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            continue

        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda m: m.distance)[:max_points_per_pair]
        if len(matches) < 8:
            continue

        # Build projection matrices (world-to-camera)
        def pose_to_proj(pose):
            q_xyzw = np.array(pose.rotation)
            t = np.array(pose.translation)
            R_w2r = Rotation.from_quat(q_xyzw).as_matrix()
            # world_from_rig -> invert to get rig_from_world
            R_r2w = R_w2r.T
            t_r2w = -R_r2w @ t
            P = K @ np.hstack([R_r2w, t_r2w.reshape(3, 1)])
            return P

        P1 = pose_to_proj(pose1)
        P2 = pose_to_proj(pose2)

        pts1 = np.array([kp1[m.queryIdx].pt for m in matches], dtype=np.float64)
        pts2 = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float64)

        # Triangulate
        pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        pts3d_h = pts4d.T
        pts3d_h /= pts3d_h[:, 3:4]
        pts3d_xyz = pts3d_h[:, :3]

        # Read color image for point colors
        img1_color = cv2.imread(str(path1))

        for j in range(len(matches)):
            X, Y, Z = pts3d_xyz[j]
            # Filter outliers: reject points too far or behind cameras
            if np.abs(X) > 100 or np.abs(Y) > 100 or np.abs(Z) > 100:
                continue

            x1, y1 = pts1[j]
            x2, y2 = pts2[j]
            pt2d_idx1 = len(image_points2d[img_id1])
            pt2d_idx2 = len(image_points2d[img_id2])

            # Get color from first image
            px, py = int(round(x1)), int(round(y1))
            if img1_color is not None and 0 <= py < img1_color.shape[0] and 0 <= px < img1_color.shape[1]:
                B, G, R = img1_color[py, px]
            else:
                R, G, B = 128, 128, 128

            track = [(img_id1, pt2d_idx1), (img_id2, pt2d_idx2)]
            points3d[point3d_id] = (X, Y, Z, int(R), int(G), int(B), 1.0, track)
            image_points2d[img_id1].append((x1, y1, point3d_id))
            image_points2d[img_id2].append((x2, y2, point3d_id))
            point3d_id += 1

    return points3d, image_points2d


def augment_colmap_with_depth(sparse_dir: Path, images_dir: Path,
                              depth_dir: Path, camera_params: dict,
                              workflow_id: str = ""):
    """Augment COLMAP sparse reconstruction with dense 3D points from PFM depth maps.
    
    Reads existing COLMAP images.txt to get poses and cameras.txt for intrinsics,
    then generates depth-based 3D points from PFM files and rewrites points3D.txt.
    """
    import cv2
    from scipy.spatial.transform import Rotation

    # Read cameras.txt for intrinsics
    cameras_file = sparse_dir / "cameras.txt"
    fx, fy, cx, cy = 0, 0, 0, 0
    with open(cameras_file) as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 8 and parts[1] == "PINHOLE":
                fx, fy, cx, cy = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                break

    if fx == 0:
        raise Exception("Could not read camera intrinsics from cameras.txt")

    # Read images.txt to get registered image poses
    images_file = sparse_dir / "images.txt"
    registered_images = []  # [(image_id, image_name, R_world, t_world)]
    with open(images_file) as f:
        lines = [l for l in f.readlines() if not l.startswith('#')]

    for i in range(0, len(lines), 2):
        parts = lines[i].strip().split()
        if len(parts) < 9:
            continue
        image_id = int(parts[0])
        qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
        image_name = parts[9] if len(parts) > 9 else parts[8]

        # COLMAP stores world-to-camera (R, t): p_cam = R * p_world + t
        # We need camera-to-world for back-projection
        R_c2w = Rotation.from_quat([qx, qy, qz, qw])  # scipy uses xyzw
        R_c2w_mat = R_c2w.inv().as_matrix()
        t_c2w = -R_c2w_mat @ np.array([tx, ty, tz])

        registered_images.append((image_id, image_name, R_c2w_mat, t_c2w))

    if len(registered_images) < 3:
        logger.warning(f"[{workflow_id}] Only {len(registered_images)} registered images, "
                       "skipping depth augmentation")
        return

    # Generate depth points from PFM maps for registered images
    points3d = {}
    image_points2d = {}
    point3d_id = 1
    max_points_per_image = 500
    sample_stride = max(1, len(registered_images) // 25)  # Process ~25 frames

    pfm_files = sorted(depth_dir.glob("*.pfm"))
    pfm_map = {p.stem: p for p in pfm_files}

    frames_processed = 0
    for img_id, img_name, R_c2w, t_c2w in registered_images[::sample_stride]:
        # Find matching PFM
        stem = img_name.rsplit('.', 1)[0]
        pfm_path = pfm_map.get(stem)
        if pfm_path is None:
            continue

        # Read depth and left image
        depth_map = read_pfm(pfm_path)
        img_path = images_dir / img_name
        if not img_path.exists():
            continue
        left_img = cv2.imread(str(img_path))
        if left_img is None:
            continue

        h_img, w_img = left_img.shape[:2]

        # PFM may be full SBS resolution — use left half
        if depth_map.shape[1] > w_img:
            depth_map = depth_map[:, :w_img]
        if depth_map.shape[0] != h_img or depth_map.shape[1] != w_img:
            depth_map = cv2.resize(depth_map, (w_img, h_img),
                                   interpolation=cv2.INTER_NEAREST)

        # Valid depth mask
        valid_mask = (depth_map > 0.1) & (depth_map < 20.0) & np.isfinite(depth_map)
        ys, xs = np.where(valid_mask)
        if len(xs) == 0:
            continue

        # Sub-sample
        if len(xs) > max_points_per_image:
            indices = np.random.choice(len(xs), max_points_per_image, replace=False)
            xs = xs[indices]
            ys = ys[indices]

        depths = depth_map[ys, xs]

        # Back-project to camera frame then to world
        X_cam = (xs.astype(np.float64) - cx) * depths / fx
        Y_cam = (ys.astype(np.float64) - cy) * depths / fy
        Z_cam = depths

        pts_cam = np.stack([X_cam, Y_cam, Z_cam], axis=1)
        pts_world = (R_c2w @ pts_cam.T).T + t_c2w

        if img_id not in image_points2d:
            image_points2d[img_id] = []

        for j in range(len(pts_world)):
            X, Y, Z = pts_world[j]
            px, py = int(xs[j]), int(ys[j])
            if 0 <= py < left_img.shape[0] and 0 <= px < left_img.shape[1]:
                B, G, R = left_img[py, px]
            else:
                R, G, B = 128, 128, 128

            pt2d_idx = len(image_points2d[img_id])
            track = [(img_id, pt2d_idx)]
            points3d[point3d_id] = (X, Y, Z, int(R), int(G), int(B), 0.5, track)
            image_points2d[img_id].append((float(xs[j]), float(ys[j]), point3d_id))
            point3d_id += 1

        frames_processed += 1

    if len(points3d) == 0:
        logger.warning(f"[{workflow_id}] No depth points generated")
        return

    # Rewrite points3D.txt with augmented points (keep existing + add new)
    existing_points = {}
    points3d_file = sparse_dir / "points3D.txt"
    if points3d_file.exists():
        with open(points3d_file) as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) >= 8:
                    pid = int(parts[0])
                    existing_points[pid] = line.strip()

    # Write merged points (existing + depth-augmented)
    offset = max(existing_points.keys()) if existing_points else 0
    with open(points3d_file, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        total_points = len(existing_points) + len(points3d)
        f.write(f"# Number of points: {total_points}\n")
        # Write existing
        for line in existing_points.values():
            f.write(line + "\n")
        # Write new depth points
        for pid, (X, Y, Z, R, G, B, err, track) in points3d.items():
            new_pid = pid + offset
            track_str = " ".join(f"{tid} {tidx}" for tid, tidx in track)
            f.write(f"{new_pid} {X:.6f} {Y:.6f} {Z:.6f} {R} {G} {B} {err:.4f} {track_str}\n")

    logger.info(f"[{workflow_id}] Augmented COLMAP with {len(points3d)} depth points "
                f"from {frames_processed} frames (total: {total_points})")


def write_depth_only_reconstruction(images_dir: Path, depth_dir: Path,
                                     output_dir: Path, camera_params: dict,
                                     workflow_id: str = "") -> Tuple[int, int]:
    """Fallback reconstruction when COLMAP fails but PFM depth maps are available.
    
    Assigns each image a unique pose spaced along a line (simulating a forward-facing
    capture), then generates dense 3D points from depth maps. This ensures training
    always has multiple unique viewpoints with proper initialization.
    
    Returns (num_registered, total_images).
    """
    import cv2
    from scipy.spatial.transform import Rotation

    all_image_files = sorted(images_dir.glob("*.jpg"))
    if not all_image_files:
        raise Exception("No images found for depth-only reconstruction")

    pfm_files = sorted(depth_dir.glob("*.pfm"))
    pfm_map = {p.stem: p for p in pfm_files}

    # Only include images that have a matching PFM depth file.
    # The COLMAP text parser uses iter(readline, "") which stops on the first
    # empty line — so every image MUST have 2D observations (no empty lines).
    image_files = [f for f in all_image_files if f.stem in pfm_map]
    if not image_files:
        # Fallback: use all images if PFM stems don't match
        image_files = all_image_files

    total_images = len(image_files)
    params = camera_params or {}

    # Read first image for dimensions
    first_img = cv2.imread(str(image_files[0]))
    h, w = first_img.shape[:2]
    fx = params.get("fx", w * 0.7)
    fy = params.get("fy", fx)
    cx = params.get("cx", w / 2.0)
    cy = params.get("cy", h / 2.0)

    # Generate synthetic poses: cameras spaced along an arc
    # Each camera is offset by ~0.05m and slightly rotated
    sparse_dir = output_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    # Write cameras.txt
    with open(sparse_dir / "cameras.txt", "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"1 PINHOLE {w} {h} {fx} {fy} {cx} {cy}\n")

    # Generate synthetic poses for all images
    image_poses = []  # [(image_id, name, qw, qx, qy, qz, tx, ty, tz)]
    for idx, img_file in enumerate(image_files):
        angle = (idx - total_images / 2) * 0.02  # Small yaw variation
        tx = idx * 0.05  # 5cm spacing along X

        rot = Rotation.from_euler('y', angle)
        quat_xyzw = rot.as_quat()
        qw_v, qx_v, qy_v, qz_v = quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]

        image_poses.append((idx + 1, img_file.name, qw_v, qx_v, qy_v, qz_v, tx, 0.0, 0.0))

    # Generate 3D points from depth maps and track 2D observations per image.
    # Every image MUST have at least one 2D observation, otherwise the COLMAP text
    # parser stops reading (it uses iter(readline, "") which halts on empty lines).
    points3d = {}
    image_points2d = {pose[0]: [] for pose in image_poses}  # {image_id: [(x, y, point3d_id)]}
    point3d_id = 1
    max_points_per_image = 200

    for idx in range(total_images):
        img_file = image_files[idx]
        stem = img_file.stem
        pfm_path = pfm_map.get(stem)
        if pfm_path is None:
            continue

        depth_map = read_pfm(pfm_path)
        left_img = cv2.imread(str(img_file))
        if left_img is None:
            continue

        h_img, w_img = left_img.shape[:2]
        if depth_map.shape[1] > w_img:
            depth_map = depth_map[:, :w_img]
        if depth_map.shape[0] != h_img or depth_map.shape[1] != w_img:
            depth_map = cv2.resize(depth_map, (w_img, h_img),
                                   interpolation=cv2.INTER_NEAREST)

        valid_mask = (depth_map > 0.1) & (depth_map < 20.0) & np.isfinite(depth_map)
        ys, xs = np.where(valid_mask)
        if len(xs) == 0:
            continue

        if len(xs) > max_points_per_image:
            indices = np.random.choice(len(xs), max_points_per_image, replace=False)
            xs = xs[indices]
            ys = ys[indices]

        depths = depth_map[ys, xs]
        image_id = idx + 1

        # Synthetic pose for this frame
        angle = (idx - total_images / 2) * 0.02
        tx_cam = idx * 0.05
        rot = Rotation.from_euler('y', angle)
        R_mat = rot.as_matrix()
        t_vec = np.array([tx_cam, 0.0, 0.0])

        # Back-project to camera frame then to world
        X_cam = (xs.astype(np.float64) - cx) * depths / fx
        Y_cam = (ys.astype(np.float64) - cy) * depths / fy
        Z_cam = depths
        pts_cam = np.stack([X_cam, Y_cam, Z_cam], axis=1)

        # Camera-to-world transform (inverse of stored w2c)
        R_c2w = R_mat.T
        t_c2w = -R_c2w @ t_vec
        pts_world = (R_c2w @ pts_cam.T).T + t_c2w

        for j in range(len(pts_world)):
            X, Y, Z = pts_world[j]
            px, py = int(xs[j]), int(ys[j])
            B, G, R_val = left_img[py, px]

            pt2d_idx = len(image_points2d[image_id])
            points3d[point3d_id] = (
                X, Y, Z, int(R_val), int(G), int(B), 0.5,
                [(image_id, pt2d_idx)]
            )
            image_points2d[image_id].append((float(xs[j]), float(ys[j]), point3d_id))
            point3d_id += 1

    # Write images.txt with 2D point observations.
    # CRITICAL: The COLMAP text parser uses iter(readline, "") which stops on the
    # first empty line. Only include images that have at least one 2D observation.
    valid_poses = [(p, image_points2d.get(p[0], [])) for p in image_poses]
    valid_poses = [(p, pts) for p, pts in valid_poses if pts]

    with open(sparse_dir / "images.txt", "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        for (image_id, name, qw_v, qx_v, qy_v, qz_v, tx_v, ty_v, tz_v), pts2d in valid_poses:
            f.write(f"{image_id} {qw_v:.10f} {qx_v:.10f} {qy_v:.10f} {qz_v:.10f} "
                    f"{tx_v:.10f} {ty_v:.10f} {tz_v:.10f} 1 {name}\n")
            pts2d_str = " ".join(
                f"{x:.2f} {y:.2f} {pid}" for x, y, pid in pts2d
            )
            f.write(f"{pts2d_str}\n")

    total_images = len(valid_poses)

    # Write points3D.txt
    with open(sparse_dir / "points3D.txt", "w") as f:
        f.write("# 3D point list\n")
        f.write(f"# Number of points: {len(points3d)}\n")
        for pid, (X, Y, Z, R_val, G, B, err, track) in points3d.items():
            track_str = " ".join(f"{tid} {tidx}" for tid, tidx in track)
            f.write(f"{pid} {X:.6f} {Y:.6f} {Z:.6f} {R_val} {G} {B} {err:.4f} {track_str}\n")

    logger.info(f"[{workflow_id}] Depth-only reconstruction: {total_images} images, "
                f"{len(points3d)} points")

    return total_images, total_images


def write_colmap_sparse(sparse_dir: Path, image_names: List[str],
                        poses: List, fx: float, fy: float,
                        cx: float, cy: float, width: int, height: int,
                        left_dir: Path = None, right_dir: Path = None,
                        baseline: float = 0.12, depth_dir: Path = None):
    """Write COLMAP-format sparse reconstruction files (text format).
    
    Creates cameras.txt, images.txt, points3D.txt in the sparse directory.
    Uses PFM depth maps if available, otherwise stereo disparity from right images.
    Falls back to ORB triangulation if depth methods fail or are unavailable.
    """
    sparse_dir.mkdir(parents=True, exist_ok=True)

    # Generate 3D points — prefer PFM depth > stereo disparity > ORB triangulation
    points3d = {}
    image_points2d = {}

    # Try depth-based point generation (PFM or stereo disparity)
    has_depth_source = (depth_dir is not None and depth_dir.exists()) or \
                       (right_dir is not None and right_dir.exists())
    if has_depth_source and left_dir is not None:
        try:
            points3d, image_points2d = generate_stereo_depth_points(
                left_dir, right_dir, image_names, poses,
                fx, fy, cx, cy, baseline=baseline,
                depth_dir=depth_dir
            )
            logger.info(f"Depth-based generation produced {len(points3d)} 3D points")
        except Exception as e:
            logger.warning(f"Depth-based point generation failed: {e}")
            points3d = {}
            image_points2d = {}

    # Fall back to ORB triangulation if stereo depth gave too few points
    if len(points3d) < 1000 and left_dir is not None and left_dir.exists():
        try:
            orb_points, orb_img_pts = triangulate_sparse_points(
                left_dir, image_names, poses, fx, fy, cx, cy
            )
            # Merge ORB points with any stereo points
            if orb_points:
                offset = max(points3d.keys()) if points3d else 0
                for pid, pdata in orb_points.items():
                    points3d[pid + offset] = pdata
                for img_id, pts in orb_img_pts.items():
                    if img_id not in image_points2d:
                        image_points2d[img_id] = []
                    image_points2d[img_id].extend(
                        [(x, y, pid + offset) for x, y, pid in pts]
                    )
            logger.info(f"ORB triangulation added points, total: {len(points3d)}")
        except Exception as e:
            logger.warning(f"ORB triangulation failed: {e}")

    # cameras.txt - single pinhole camera
    with open(sparse_dir / "cameras.txt", "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: 1\n")
        f.write(f"1 PINHOLE {width} {height} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")

    # images.txt - per-image poses with 2D point observations
    with open(sparse_dir / "images.txt", "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(poses)}\n")

        for i, (name, pose) in enumerate(zip(image_names, poses)):
            if pose is None:
                continue
            quat_wxyz, tvec = pose_to_colmap_qtvec(pose)
            image_id = i + 1
            camera_id = 1
            f.write(f"{image_id} {quat_wxyz[0]:.10f} {quat_wxyz[1]:.10f} "
                    f"{quat_wxyz[2]:.10f} {quat_wxyz[3]:.10f} "
                    f"{tvec[0]:.10f} {tvec[1]:.10f} {tvec[2]:.10f} "
                    f"{camera_id} {name}\n")
            # POINTS2D line
            pts2d = image_points2d.get(image_id, [])
            if pts2d:
                parts = [f"{x:.2f} {y:.2f} {pid}" for x, y, pid in pts2d]
                f.write(" ".join(parts) + "\n")
            else:
                f.write("\n")

    # points3D.txt
    with open(sparse_dir / "points3D.txt", "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(points3d)}\n")
        for pid, (X, Y, Z, R, G, B, err, track) in points3d.items():
            track_str = " ".join(f"{tid} {tidx}" for tid, tidx in track)
            f.write(f"{pid} {X:.6f} {Y:.6f} {Z:.6f} {R} {G} {B} {err:.4f} {track_str}\n")

    logger.info(f"Wrote COLMAP sparse to {sparse_dir}: "
                f"{len([p for p in poses if p is not None])} images, "
                f"{len(points3d)} points, 1 camera")


def run_cuvslam_on_frames(left_dir: Path, right_dir: Path,
                          output_dir: Path, workflow_id: str,
                          camera_params: dict = None,
                          depth_dir: Path = None):
    """Run cuVSLAM on extracted stereo frame pairs and produce COLMAP output."""
    import cv2

    # Find matching left/right frame pairs
    left_files = sorted(left_dir.glob("*.jpg")) + sorted(left_dir.glob("*.png"))
    right_files = sorted(right_dir.glob("*.jpg")) + sorted(right_dir.glob("*.png"))

    if len(left_files) == 0:
        raise Exception("No left frames found")
    if len(right_files) == 0:
        raise Exception("No right frames found - stereo pairs required for cuVSLAM")

    # Match by index (frame_0000.jpg pairs with frame_0000.jpg)
    num_pairs = min(len(left_files), len(right_files))
    logger.info(f"[{workflow_id}] Processing {num_pairs} stereo frame pairs")

    # Read first frame to get dimensions
    first_img = cv2.imread(str(left_files[0]))
    if first_img is None:
        raise Exception(f"Cannot read image: {left_files[0]}")
    h, w = first_img.shape[:2]

    # Create stereo rig
    params = camera_params or {}
    rig, fx, fy, cx, cy = create_stereo_rig(
        width=w, height=h,
        baseline=params.get("baseline", 0.12),
        fx=params.get("fx"), fy=params.get("fy"),
        cx=params.get("cx"), cy=params.get("cy")
    )

    # Configure tracker
    cfg = vslam.Tracker.OdometryConfig(
        async_sba=False,
        enable_final_landmarks_export=True,
        enable_observations_export=True,
        horizontal_stereo_camera=True
    )

    tracker = vslam.Tracker(rig, cfg)
    logger.info(f"[{workflow_id}] cuVSLAM tracker initialized: {w}x{h}, "
                f"fx={fx:.1f}, baseline={params.get('baseline', 0.12)}")

    # Process frames
    poses = []
    image_names = []
    failed_frames = 0
    timestamp_ns = 0
    # Simulate 30fps timestamps
    frame_period_ns = int(1e9 / 30)

    for i in range(num_pairs):
        left_img = cv2.imread(str(left_files[i]))
        right_img = cv2.imread(str(right_files[i]))

        if left_img is None or right_img is None:
            logger.warning(f"[{workflow_id}] Skipping frame {i}: cannot read images")
            poses.append(None)
            image_names.append(left_files[i].name)
            failed_frames += 1
            timestamp_ns += frame_period_ns
            continue

        # Convert BGR to RGB
        left_rgb = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
        right_rgb = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)

        # Ensure contiguous arrays
        left_rgb = np.ascontiguousarray(left_rgb)
        right_rgb = np.ascontiguousarray(right_rgb)

        try:
            pose_estimate, _ = tracker.track(
                timestamp_ns, images=[left_rgb, right_rgb]
            )

            if pose_estimate.world_from_rig is not None:
                poses.append(pose_estimate.world_from_rig.pose)
            else:
                poses.append(None)
                failed_frames += 1
                logger.warning(f"[{workflow_id}] Frame {i}: tracking lost")
        except Exception as e:
            poses.append(None)
            failed_frames += 1
            logger.warning(f"[{workflow_id}] Frame {i} tracking error: {e}")

        image_names.append(left_files[i].name)
        timestamp_ns += frame_period_ns

        # Update progress
        progress = (i + 1) / num_pairs
        update_workflow(workflow_id, {
            "progress": 0.3 + progress * 0.35,
            "current_step": f"cuVSLAM: processed {i+1}/{num_pairs} frames"
        })

    valid_poses = len([p for p in poses if p is not None])
    logger.info(f"[{workflow_id}] cuVSLAM complete: {valid_poses}/{num_pairs} frames tracked "
                f"({failed_frames} failed)")

    if valid_poses < 3:
        raise Exception(f"Only {valid_poses} frames tracked successfully, need at least 3")

    # Write COLMAP-format sparse output (with stereo depth + triangulated 3D points)
    sparse_dir = output_dir / "sparse" / "0"
    baseline = params.get("baseline", 0.12)
    write_colmap_sparse(sparse_dir, image_names, poses,
                        fx, fy, cx, cy, w, h,
                        left_dir=left_dir, right_dir=right_dir,
                        baseline=baseline, depth_dir=depth_dir)

    return valid_poses, num_pairs


async def run_colmap_monocular(images_dir: Path, output_dir: Path, workflow_id: str,
                               camera_params: dict = None) -> Tuple[int, int]:
    """Run COLMAP monocular reconstruction for non-stereo captures (e.g., SplatKing).
    
    Uses the same CPU-based COLMAP pipeline as the video workflow:
    feature extraction (use_gpu=0), matching, and incremental mapper.
    
    Returns (num_registered, total_images) tuple.
    """
    import cv2

    image_files = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    if len(image_files) == 0:
        raise Exception("No images found for COLMAP reconstruction")

    total_images = len(image_files)
    logger.info(f"[{workflow_id}] Running COLMAP monocular on {total_images} images")

    # Read first image for dimensions
    first_img = cv2.imread(str(image_files[0]))
    if first_img is None:
        raise Exception(f"Cannot read image: {image_files[0]}")
    h, w = first_img.shape[:2]

    # Setup COLMAP workspace
    sparse_dir = output_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    database_path = output_dir / "database.db"

    # Headless environment for COLMAP (same as video workflow)
    env = os.environ.copy()
    env['QT_QPA_PLATFORM'] = 'offscreen'

    # Build camera params for COLMAP PINHOLE model
    params = camera_params or {}
    fx = params.get("fx", w * 0.7)
    fy = params.get("fy", fx)
    cx = params.get("cx", w / 2.0)
    cy = params.get("cy", h / 2.0)
    camera_params_str = f"{fx},{fy},{cx},{cy}"

    update_workflow(workflow_id, {
        "current_step": f"COLMAP: Feature extraction ({total_images} images, CPU)",
        "progress": 0.32
    })

    # Step 1: Feature extraction on CPU with known camera intrinsics
    # Use high feature count and lower peak threshold for indoor/challenging scenes
    cmd_extract = [
        "colmap", "feature_extractor",
        "--database_path", str(database_path),
        "--image_path", str(images_dir),
        "--ImageReader.camera_model", "PINHOLE",
        "--ImageReader.camera_params", camera_params_str,
        "--ImageReader.single_camera", "1",
        "--SiftExtraction.max_image_size", "2048",
        "--SiftExtraction.max_num_features", "32768",
        "--SiftExtraction.peak_threshold", "0.004",
        "--SiftExtraction.use_gpu", "0",
    ]
    result = await asyncio.to_thread(
        subprocess.run, cmd_extract, capture_output=True, text=True, timeout=1800, env=env
    )
    if result.returncode != 0:
        logger.error(f"[{workflow_id}] COLMAP feature extraction failed: {result.stderr[:500]}")
        raise Exception(f"COLMAP feature extraction failed: {result.stderr[:200]}")

    # Auto-switch to sequential matcher for large image counts (same as video workflow)
    effective_matcher = "exhaustive"
    if total_images > 200:
        effective_matcher = "sequential"
        logger.info(f"[{workflow_id}] Auto-switching to sequential matcher ({total_images} images)")

    update_workflow(workflow_id, {
        "current_step": f"COLMAP: Matching features ({effective_matcher}, CPU)",
        "progress": 0.42
    })

    # Step 2: Feature matching on CPU
    if effective_matcher == "exhaustive":
        cmd_match = [
            "colmap", "exhaustive_matcher",
            "--database_path", str(database_path),
            "--SiftMatching.use_gpu", "0",
        ]
    else:
        cmd_match = [
            "colmap", "sequential_matcher",
            "--database_path", str(database_path),
            "--SequentialMatching.overlap", "10",
            "--SiftMatching.use_gpu", "0",
        ]
    result = await asyncio.to_thread(
        subprocess.run, cmd_match, capture_output=True, text=True, timeout=7200, env=env
    )
    if result.returncode != 0:
        logger.error(f"[{workflow_id}] COLMAP matching failed: {result.stderr[:500]}")
        raise Exception(f"COLMAP matching failed: {result.stderr[:200]}")

    update_workflow(workflow_id, {
        "current_step": "COLMAP: Incremental mapping (3D reconstruction)",
        "progress": 0.52
    })

    # Step 3: Incremental mapper with relaxed settings for challenging scenes
    cmd_mapper = [
        "colmap", "mapper",
        "--database_path", str(database_path),
        "--image_path", str(images_dir),
        "--output_path", str(sparse_dir),
        "--Mapper.ba_global_max_num_iterations", "50",
        "--Mapper.ba_global_max_refinements", "3",
        "--Mapper.init_min_num_inliers", "15",
        "--Mapper.min_num_matches", "10",
        "--Mapper.multiple_models", "1",
    ]
    result = await asyncio.to_thread(
        subprocess.run, cmd_mapper, capture_output=True, text=True, timeout=3600, env=env
    )
    if result.returncode != 0:
        # Log stderr for debugging
        logger.error(f"[{workflow_id}] COLMAP mapper failed: {result.stderr[:500]}")
        # Check if any partial model was created
        any_model = any(
            (sparse_dir / d / "images.bin").exists()
            for d in os.listdir(sparse_dir) if (sparse_dir / d).is_dir()
        ) if sparse_dir.exists() else False
        if not any_model:
            raise Exception(f"COLMAP mapper failed: {result.stderr[:200]}")

    # Find the best reconstruction (largest sub-model)
    best_model = None
    best_count = 0
    for sub_dir in sorted(sparse_dir.iterdir()):
        if sub_dir.is_dir():
            images_file = sub_dir / "images.bin"
            if images_file.exists():
                size = images_file.stat().st_size
                if size > best_count:
                    best_count = size
                    best_model = sub_dir

    if best_model is None:
        raise Exception("COLMAP mapper produced no reconstruction")

    # Move best model to sparse/0
    target_dir = output_dir / "sparse" / "0"
    if best_model.name != "0":
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.move(str(best_model), str(target_dir))
    
    # Convert binary to text format for compatibility with training service
    update_workflow(workflow_id, {
        "current_step": "COLMAP: Converting model to text format",
        "progress": 0.62
    })
    cmd_convert = [
        "colmap", "model_converter",
        "--input_path", str(target_dir),
        "--output_path", str(target_dir),
        "--output_type", "TXT",
    ]
    await asyncio.to_thread(
        subprocess.run, cmd_convert, capture_output=True, text=True, timeout=120, env=env
    )

    # Count registered images from images.txt
    num_registered = 0
    images_txt = target_dir / "images.txt"
    if images_txt.exists():
        with open(images_txt, "r") as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    num_registered += 1
        # images.txt has 2 lines per image (pose line + points line)
        num_registered = num_registered // 2

    logger.info(f"[{workflow_id}] COLMAP complete: {num_registered}/{total_images} images registered")
    return num_registered, total_images


@app.get("/")
async def root():
    return {
        "service": "cuVSLAM Processing Service",
        "version": "1.0.0",
        "cuvslam_available": CUVSLAM_AVAILABLE,
        "cuvslam_version": vslam.__version__ if CUVSLAM_AVAILABLE else None,
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "cuvslam_available": CUVSLAM_AVAILABLE,
        # Match COLMAP health response key for compatibility
        "colmap_available": CUVSLAM_AVAILABLE,
        "service": "cuVSLAM",
        "active_jobs": len([j for j in processing_jobs.values()
                           if j.get("status") == "processing"])
    }


@app.post("/workflow/video-to-model")
async def workflow_video_to_model(
    file: UploadFile = File(...),
    dataset_id: str = Form(...),
    fps: float = Form(1.0),
    camera_model: str = Form("SIMPLE_RADIAL"),
    matcher: str = Form("exhaustive"),
    num_training_steps: int = Form(30000),
    use_mcmc: str = Form("false"),
    filter_blur: str = Form("false"),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Complete workflow: Upload video -> Extract frames -> Run cuVSLAM -> Train Gaussian Splat.
    Extracts frames from video at specified FPS using OpenCV.
    """
    import httpx
    import cv2

    workflow_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

    update_workflow(workflow_id, {
        "workflow_id": workflow_id,
        "status": "uploading",
        "progress": 0.0,
        "current_step": "Uploading video",
        "dataset_id": dataset_id,
        "started_at": datetime.now().isoformat(),
        "colmap_job_id": None,
        "training_job_id": None,
        "error": None
    })

    # Stream video to disk in chunks to avoid loading 600MB+ into memory
    video_path = TEMP_DIR / f"{dataset_id}_{file.filename}"
    file_size = 0
    with open(video_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):  # 1MB chunks
            f.write(chunk)
            file_size += len(chunk)
    logger.info(f"[{workflow_id}] Video saved: {file_size / (1024*1024):.1f} MB")

    async def run_video_workflow():
        job_id = None
        try:
            # Step 1: Save uploaded file
            update_workflow(workflow_id, {
                "current_step": "Saving video file",
                "progress": 0.1
            })

            dataset_dir = UPLOAD_DIR / dataset_id
            dataset_dir.mkdir(exist_ok=True, parents=True)

            output_dir = OUTPUT_DIR / dataset_id
            output_dir.mkdir(exist_ok=True, parents=True)
            images_dir = output_dir / "images"
            images_dir.mkdir(exist_ok=True, parents=True)

            logger.info(f"[{workflow_id}] Saved video to {video_path}")

            # Step 2: Extract frames with ffmpeg
            update_workflow(workflow_id, {
                "current_step": "Extracting frames from video",
                "progress": 0.2
            })

            job_id = f"colmap_{dataset_id}_{datetime.now().strftime('%H%M%S')}"
            processing_jobs[job_id] = {
                "job_id": job_id,
                "dataset_id": dataset_id,
                "status": "processing",
                "progress": 0.0,
                "message": "Starting COLMAP processing",
                "started_at": datetime.now().isoformat()
            }
            update_workflow(workflow_id, {"colmap_job_id": job_id})

            result = await asyncio.to_thread(subprocess.run, [
                "ffmpeg", "-i", str(video_path),
                "-vf", f"fps={fps}",
                "-q:v", "2",
                str(images_dir / "frame_%04d.jpg")
            ], capture_output=True, text=True, timeout=600)

            video_path.unlink(missing_ok=True)

            if result.returncode != 0:
                raise Exception(f"Frame extraction failed: {result.stderr}")

            num_images = len(list(images_dir.glob("*.jpg")))
            logger.info(f"[{workflow_id}] Extracted {num_images} frames")
            processing_jobs[job_id]["progress"] = 0.3
            processing_jobs[job_id]["message"] = f"Extracted {num_images} frames"
            update_workflow(workflow_id, {"progress": 0.25})

            if num_images < 3:
                raise Exception(f"Only extracted {num_images} frames, need at least 3. Try higher FPS.")

            # Filter blurry frames if enabled
            if filter_blur.lower() == "true":
                update_workflow(workflow_id, {
                    "progress": 0.27,
                    "current_step": "Filtering blurry frames..."
                })
                num_images = filter_sharp_frames(images_dir, workflow_id=workflow_id)
                logger.info(f"[{workflow_id}] After blur filter: {num_images} frames")
                update_workflow(workflow_id, {
                    "progress": 0.3,
                    "current_step": f"Sharp filter: kept {num_images} frames"
                })

            if num_images < 3:
                raise Exception(f"Only {num_images} sharp frames remain, need at least 3.")

            # Step 3: Run COLMAP reconstruction
            update_workflow(workflow_id, {
                "current_step": "Running COLMAP reconstruction",
                "progress": 0.35
            })

            sparse_dir = output_dir / "sparse" / "0"
            sparse_dir.mkdir(exist_ok=True, parents=True)
            database_path = output_dir / "database.db"

            env = os.environ.copy()
            env['QT_QPA_PLATFORM'] = 'offscreen'

            # Feature extraction
            processing_jobs[job_id]["message"] = "Extracting features..."
            update_workflow(workflow_id, {"progress": 0.4})

            cmd_extract = [
                "colmap", "feature_extractor",
                "--database_path", str(database_path),
                "--image_path", str(images_dir),
                "--ImageReader.single_camera", "1",
                "--ImageReader.camera_model", camera_model,
                "--SiftExtraction.max_image_size", "2048",
                "--SiftExtraction.max_num_features", "16384",
                "--SiftExtraction.use_gpu", "0"
            ]

            result = await asyncio.to_thread(subprocess.run, cmd_extract, capture_output=True, text=True, timeout=1800, env=env)
            if result.returncode != 0:
                raise Exception(f"Feature extraction failed: {result.stderr}")

            processing_jobs[job_id]["progress"] = 0.5
            update_workflow(workflow_id, {"progress": 0.5})

            # Auto-switch to sequential matcher for large frame counts.
            # Exhaustive is O(n²) on CPU and will timeout for >200 images.
            # Video frames are sequential, so sequential_matcher is correct and fast.
            effective_matcher = matcher
            if matcher == "exhaustive" and num_images > 200:
                effective_matcher = "sequential"
                logger.info(f"[{workflow_id}] Auto-switching from exhaustive to sequential matcher ({num_images} images)")

            processing_jobs[job_id]["message"] = f"Matching features ({effective_matcher})..."
            update_workflow(workflow_id, {"current_step": f"COLMAP: matching features ({effective_matcher})"})

            if effective_matcher == "exhaustive":
                cmd_match = [
                    "colmap", "exhaustive_matcher",
                    "--database_path", str(database_path),
                    "--SiftMatching.use_gpu", "0"
                ]
            else:
                cmd_match = [
                    "colmap", "sequential_matcher",
                    "--database_path", str(database_path),
                    "--SequentialMatching.overlap", "10",
                    "--SiftMatching.use_gpu", "0"
                ]

            result = await asyncio.to_thread(subprocess.run, cmd_match, capture_output=True, text=True, timeout=7200, env=env)
            if result.returncode != 0:
                raise Exception(f"Feature matching failed: {result.stderr}")

            processing_jobs[job_id]["progress"] = 0.7
            processing_jobs[job_id]["message"] = "Running sparse reconstruction..."
            update_workflow(workflow_id, {"progress": 0.6})

            # Sparse reconstruction
            cmd_mapper = [
                "colmap", "mapper",
                "--database_path", str(database_path),
                "--image_path", str(images_dir),
                "--output_path", str(sparse_dir.parent)
            ]

            result = await asyncio.to_thread(subprocess.run, cmd_mapper, capture_output=True, text=True, timeout=3600, env=env)
            if result.returncode != 0:
                raise Exception(f"Sparse reconstruction failed: {result.stderr}")

            processing_jobs[job_id]["status"] = "completed"
            processing_jobs[job_id]["progress"] = 1.0
            processing_jobs[job_id]["message"] = "COLMAP processing complete"
            processing_jobs[job_id]["num_images"] = num_images
            processing_jobs[job_id]["completed_at"] = datetime.now().isoformat()

            update_workflow(workflow_id, {
                "current_step": "COLMAP complete, starting training",
                "progress": 0.7
            })
            logger.info(f"[{workflow_id}] COLMAP processing complete")

            # Step 4: Trigger training
            update_workflow(workflow_id, {
                "current_step": "Starting Gaussian Splat training"
            })

            training_url = os.environ.get(
                "TRAINING_SERVICE_URL", "http://fvdb-training-gpu:8000"
            )

            training_job_id = None
            async with httpx.AsyncClient(timeout=30.0) as client:
                try:
                    response = await client.post(
                        f"{training_url}/train",
                        json={
                            "dataset_id": dataset_id,
                            "num_training_steps": num_training_steps,
                            "output_name": f"{dataset_id}_model",
                            "use_mcmc": use_mcmc.lower() == "true"
                        }
                    )

                    if response.status_code == 200:
                        train_data = response.json()
                        training_job_id = train_data.get("job_id")
                        update_workflow(workflow_id, {
                            "training_job_id": training_job_id,
                            "status": "training",
                            "progress": 0.75,
                            "current_step": "Training in progress"
                        })
                        logger.info(f"[{workflow_id}] Training started: {training_job_id}")
                    else:
                        raise Exception(
                            f"Training service returned {response.status_code}"
                        )

                except Exception as e:
                    logger.error(f"[{workflow_id}] Failed to start training: {e}")
                    update_workflow(workflow_id, {
                        "status": "completed_colmap_only",
                        "progress": 0.7,
                        "current_step": "COLMAP complete, training failed to start",
                        "error": f"Training failed: {str(e)}"
                    })

            # Poll training status until complete (reuse shared poller)
            if training_job_id:
                await _poll_training_completion(workflow_id, training_job_id, training_url)

        except Exception as e:
            logger.error(f"[{workflow_id}] Video workflow failed: {e}")
            video_path.unlink(missing_ok=True)
            update_workflow(workflow_id, {
                "status": "failed",
                "error": str(e),
                "current_step": f"Failed: {str(e)}"
            })
            if job_id and job_id in processing_jobs:
                processing_jobs[job_id]["status"] = "failed"
                processing_jobs[job_id]["message"] = str(e)

    background_tasks.add_task(run_video_workflow)

    return {
        "workflow_id": workflow_id,
        "status": "started",
        "message": "Video workflow initiated. Monitor at GET /workflow/status/{workflow_id}",
        "dataset_id": dataset_id
    }


@app.post("/workflow/photos-to-model")
async def workflow_photos_to_model(
    files: List[UploadFile] = File(...),
    dataset_id: str = Form(...),
    camera_model: str = Form("PINHOLE"),
    matcher: str = Form("sequential"),
    num_training_steps: int = Form(30000),
    use_mcmc: str = Form("false"),
    filter_blur: str = Form("false"),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Complete workflow: Upload stereo photos -> Run cuVSLAM -> Train Gaussian Splat.
    
    Accepts a ZIP with images/ (left frames) and images_right/ (right frames),
    or two ZIPs, or individual files. Falls back to monocular if no right frames.
    """
    import httpx

    workflow_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

    update_workflow(workflow_id, {
        "workflow_id": workflow_id,
        "status": "uploading",
        "progress": 0.0,
        "current_step": "Uploading photos",
        "dataset_id": dataset_id,
        "started_at": datetime.now().isoformat(),
        "colmap_job_id": None,
        "training_job_id": None,
        "error": None
    })

    async def run_workflow():
        try:
            update_workflow(workflow_id, {
                "current_step": "Saving photos",
                "progress": 0.1
            })

            output_dir = OUTPUT_DIR / dataset_id
            output_dir.mkdir(exist_ok=True, parents=True)

            images_dir = output_dir / "images"
            images_right_dir = output_dir / "images_right"
            depth_dir = output_dir / "depth"
            images_dir.mkdir(exist_ok=True, parents=True)
            images_right_dir.mkdir(exist_ok=True, parents=True)

            num_left = 0
            num_right = 0
            camera_params = {}

            is_splatking = False

            for file in files:
                filename_lower = file.filename.lower()

                if filename_lower.endswith('.zip'):
                    content = await file.read()
                    zip_path = output_dir / "upload.zip"
                    with open(zip_path, "wb") as f:
                        f.write(content)

                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        # Detect SplatKing format
                        if is_splatking_zip(zip_ref):
                            is_splatking = True
                            logger.info(f"[{workflow_id}] Detected SplatKing ZIP format")
                            update_workflow(workflow_id, {
                                "current_step": "Parsing SplatKing capture (quality filtering + metadata extraction)",
                                "progress": 0.12
                            })
                            num_left, num_right, camera_params = extract_splatking_images(
                                zip_ref, output_dir,
                                quality_threshold=0.35,
                                preferred_stream="ultra",
                                workflow_id=workflow_id
                            )
                        else:
                            # Check if this is a ZED still image ZIP (PNGs + PFMs)
                            all_names = [z.filename for z in zip_ref.filelist]
                            has_pfm = any(n.lower().endswith('.pfm') for n in all_names)
                            has_png = any(n.lower().endswith('.png') for n in all_names)
                            is_zed_still = has_pfm and has_png

                            if is_zed_still:
                                # ZED still image ZIP: side-by-side PNGs + PFM depth maps
                                import cv2 as cv2_extract
                                depth_dir = output_dir / "depth"
                                depth_dir.mkdir(exist_ok=True, parents=True)

                                png_names = sorted(
                                    [n for n in all_names if n.lower().endswith('.png')]
                                )
                                pfm_names = sorted(
                                    [n for n in all_names if n.lower().endswith('.pfm')]
                                )
                                # Build PFM map by stem name
                                pfm_stem_map = {}
                                for pfm_n in pfm_names:
                                    stem = Path(pfm_n).stem
                                    pfm_stem_map[stem] = pfm_n

                                for png_name in png_names:
                                    stem = Path(png_name).stem
                                    # Extract and split side-by-side PNG
                                    png_data = zip_ref.read(png_name)
                                    arr = np.frombuffer(png_data, np.uint8)
                                    img = cv2_extract.imdecode(arr, cv2_extract.IMREAD_COLOR)
                                    if img is None:
                                        continue
                                    h_img, w_img = img.shape[:2]

                                    # Split side-by-side: left half and right half
                                    left_img = img[:, :w_img // 2, :]
                                    right_img = img[:, w_img // 2:, :]

                                    left_path = images_dir / f"frame_{num_left:04d}.jpg"
                                    right_path = images_right_dir / f"frame_{num_left:04d}.jpg"
                                    cv2_extract.imwrite(str(left_path), left_img,
                                                        [cv2_extract.IMWRITE_JPEG_QUALITY, 95])
                                    cv2_extract.imwrite(str(right_path), right_img,
                                                        [cv2_extract.IMWRITE_JPEG_QUALITY, 95])

                                    # Extract matching PFM depth map
                                    if stem in pfm_stem_map:
                                        pfm_data = zip_ref.read(pfm_stem_map[stem])
                                        pfm_path = depth_dir / f"frame_{num_left:04d}.pfm"
                                        with open(pfm_path, 'wb') as pf:
                                            pf.write(pfm_data)

                                    num_left += 1
                                    num_right += 1

                                logger.info(f"[{workflow_id}] ZED still image ZIP: "
                                            f"split {num_left} side-by-side frames, "
                                            f"{len(list(depth_dir.glob('*.pfm')))} depth maps")
                            else:
                                # Standard ZIP: extract images by directory structure.
                                # Sort entries to ensure consistent ordering between
                                # images and their corresponding PFM depth maps.
                                depth_dir.mkdir(exist_ok=True, parents=True)

                                # Separate entries by type and sort for consistent ordering
                                left_entries = []
                                right_entries = []
                                pfm_entries = []
                                for zip_info in zip_ref.filelist:
                                    zname = zip_info.filename.lower()
                                    if zname.endswith('.pfm') and 'depth' in zname:
                                        pfm_entries.append(zip_info)
                                    elif zname.endswith(('.jpg', '.jpeg', '.png', '.heic', '.heif')):
                                        if 'right' in zname or 'images_right' in zname:
                                            right_entries.append(zip_info)
                                        else:
                                            left_entries.append(zip_info)

                                left_entries.sort(key=lambda z: z.filename)
                                right_entries.sort(key=lambda z: z.filename)
                                pfm_entries.sort(key=lambda z: z.filename)

                                # Extract left images (sorted)
                                for zip_info in left_entries:
                                    extracted = zip_ref.read(zip_info.filename)
                                    img_path = images_dir / f"frame_{num_left:04d}.jpg"
                                    save_image_as_jpeg(extracted, img_path)
                                    num_left += 1

                                # Extract right images (sorted)
                                for zip_info in right_entries:
                                    extracted = zip_ref.read(zip_info.filename)
                                    img_path = images_right_dir / f"frame_{num_right:04d}.jpg"
                                    save_image_as_jpeg(extracted, img_path)
                                    num_right += 1

                                # Extract PFM depth maps (sorted, renumbered to match)
                                for pfm_idx, zip_info in enumerate(pfm_entries):
                                    pfm_data = zip_ref.read(zip_info.filename)
                                    pfm_path = depth_dir / f"frame_{pfm_idx:04d}.pfm"
                                    with open(pfm_path, 'wb') as pf:
                                        pf.write(pfm_data)

                            # Check for camera_params.json in ZIP
                            try:
                                params_data = zip_ref.read("camera_params.json")
                                camera_params = json.loads(params_data)
                                logger.info(f"[{workflow_id}] Loaded camera params from ZIP: {camera_params}")
                            except (KeyError, json.JSONDecodeError):
                                pass

                    zip_path.unlink()

                elif filename_lower.endswith(('.jpg', '.jpeg', '.png', '.heic', '.heif')):
                    content = await file.read()
                    if 'right' in file.filename.lower():
                        img_path = images_right_dir / f"frame_{num_right:04d}.jpg"
                        save_image_as_jpeg(content, img_path)
                        num_right += 1
                    else:
                        img_path = images_dir / f"frame_{num_left:04d}.jpg"
                        save_image_as_jpeg(content, img_path)
                        num_left += 1

                elif filename_lower == 'camera_params.json':
                    content = await file.read()
                    try:
                        camera_params = json.loads(content)
                    except json.JSONDecodeError:
                        pass

            logger.info(f"[{workflow_id}] Saved {num_left} left, {num_right} right frames")
            update_workflow(workflow_id, {
                "progress": 0.2,
                "current_step": f"Saved {num_left} left + {num_right} right frames"
            })

            # Filter blurry frames using sharp-frames if enabled
            # Skip for SplatKing captures since quality_flags.csv was already applied during extraction
            if filter_blur.lower() == "true" and not is_splatking:
                update_workflow(workflow_id, {
                    "progress": 0.22,
                    "current_step": "Filtering blurry frames..."
                })
                right_dir = images_right_dir if num_right > 0 else None
                num_left = filter_sharp_frames(images_dir, right_dir, workflow_id)
                if num_right > 0:
                    num_right = len(list(images_right_dir.glob("*.jpg")))
                logger.info(f"[{workflow_id}] After blur filter: {num_left} left, {num_right} right")
                update_workflow(workflow_id, {
                    "progress": 0.25,
                    "current_step": f"Sharp filter: kept {num_left} frames"
                })
            elif is_splatking:
                logger.info(f"[{workflow_id}] Skipping sharp-frames filter (SplatKing quality_flags already applied)")

            if num_left < 3:
                raise Exception(f"Need at least 3 left images, got {num_left}")

            # Choose reconstruction method based on capture type
            job_id = f"cuvslam_{dataset_id}_{datetime.now().strftime('%H%M%S')}"
            processing_jobs[job_id] = {
                "job_id": job_id,
                "dataset_id": dataset_id,
                "status": "processing",
                "progress": 0.0,
                "message": "Starting reconstruction",
                "started_at": datetime.now().isoformat()
            }
            update_workflow(workflow_id, {
                "colmap_job_id": job_id,
                "progress": 0.3
            })

            # Determine if PFM depth maps are available (ZED still image ZIP)
            has_pfm_depth = depth_dir.exists() and \
                len(list(depth_dir.glob("*.pfm"))) > 0

            if is_splatking or num_right == 0 or has_pfm_depth:
                # Use COLMAP monocular for:
                # - SplatKing (different focal lengths per stream)
                # - No stereo right images
                # - ZED still images (wide-baseline shots need COLMAP, not SLAM)
                #   cuVSLAM requires sequential video frames with small inter-frame motion.
                #   Still images from different positions need feature-based SfM.
                reason = "splatking" if is_splatking else \
                         "ZED still images (wide-baseline)" if has_pfm_depth else \
                         "no right frames"
                logger.info(f"[{workflow_id}] Using COLMAP monocular reconstruction "
                            f"(reason={reason}, right_frames={num_right})")
                update_workflow(workflow_id, {
                    "current_step": "Running COLMAP reconstruction (feature matching)",
                    "progress": 0.3
                })

                colmap_succeeded = False
                try:
                    valid_poses, total_frames = await run_colmap_monocular(
                        images_dir, output_dir, workflow_id, camera_params
                    )
                    colmap_succeeded = True
                except Exception as colmap_err:
                    logger.warning(f"[{workflow_id}] COLMAP failed: {colmap_err}")

                if colmap_succeeded and has_pfm_depth and valid_poses >= 3:
                    # Augment COLMAP sparse points with PFM depth
                    try:
                        sparse_dir = output_dir / "sparse" / "0"
                        augment_colmap_with_depth(
                            sparse_dir, images_dir, depth_dir, camera_params,
                            workflow_id
                        )
                    except Exception as e:
                        logger.warning(f"[{workflow_id}] Depth augmentation failed: {e}")
                elif not colmap_succeeded and has_pfm_depth:
                    # Fallback: COLMAP failed but we have depth maps.
                    # Generate reconstruction using depth-only with synthetic poses.
                    logger.info(f"[{workflow_id}] Falling back to depth-only reconstruction")
                    update_workflow(workflow_id, {
                        "current_step": "COLMAP failed, using depth-only reconstruction",
                        "progress": 0.5
                    })
                    valid_poses, total_frames = write_depth_only_reconstruction(
                        images_dir, depth_dir, output_dir, camera_params,
                        workflow_id
                    )
                elif not colmap_succeeded:
                    raise Exception("COLMAP reconstruction failed and no depth maps available")
            else:
                # Sequential stereo video: use cuVSLAM
                if not CUVSLAM_AVAILABLE:
                    raise Exception("cuVSLAM library not available in this container")
                update_workflow(workflow_id, {
                    "current_step": "Running cuVSLAM stereo reconstruction",
                    "progress": 0.3
                })
                valid_poses, total_frames = run_cuvslam_on_frames(
                    images_dir, images_right_dir, output_dir,
                    workflow_id, camera_params,
                    depth_dir=None
                )

            processing_jobs[job_id]["status"] = "completed"
            processing_jobs[job_id]["progress"] = 1.0
            processing_jobs[job_id]["message"] = "Reconstruction complete"
            processing_jobs[job_id]["num_images"] = valid_poses
            processing_jobs[job_id]["completed_at"] = datetime.now().isoformat()

            update_workflow(workflow_id, {
                "current_step": f"Reconstruction complete ({valid_poses}/{total_frames} frames)",
                "progress": 0.7
            })

            logger.info(f"[{workflow_id}] Reconstruction complete")

            # Trigger training
            update_workflow(workflow_id, {
                "current_step": "Starting Gaussian Splat training"
            })

            training_url = os.environ.get(
                "TRAINING_SERVICE_URL", "http://fvdb-training-gpu:8000"
            )

            training_job_id = None
            async with httpx.AsyncClient(timeout=30.0) as client:
                try:
                    response = await client.post(
                        f"{training_url}/train",
                        json={
                            "dataset_id": dataset_id,
                            "num_training_steps": num_training_steps,
                            "output_name": f"{dataset_id}_model",
                            "use_mcmc": use_mcmc.lower() == "true"
                        }
                    )

                    if response.status_code == 200:
                        train_data = response.json()
                        training_job_id = train_data.get("job_id")
                        update_workflow(workflow_id, {
                            "training_job_id": training_job_id,
                            "status": "training",
                            "progress": 0.75,
                            "current_step": "Training in progress"
                        })
                        logger.info(f"[{workflow_id}] Training started: "
                                    f"{training_job_id}")
                    else:
                        raise Exception(
                            f"Training service returned {response.status_code}"
                        )

                except Exception as e:
                    logger.error(f"[{workflow_id}] Failed to start training: {e}")
                    update_workflow(workflow_id, {
                        "status": "completed_colmap_only",
                        "progress": 0.7,
                        "current_step": "cuVSLAM complete, training failed to start",
                        "error": f"Training failed: {str(e)}"
                    })

            # Poll training status until complete (reuse shared poller)
            if training_job_id:
                await _poll_training_completion(workflow_id, training_job_id, training_url)

        except Exception as e:
            logger.error(f"[{workflow_id}] Workflow failed: {e}")
            update_workflow(workflow_id, {
                "status": "failed",
                "error": str(e),
                "current_step": f"Failed: {str(e)}"
            })

    background_tasks.add_task(run_workflow)

    return {
        "workflow_id": workflow_id,
        "status": "started",
        "message": "cuVSLAM workflow initiated. Monitor at GET /workflow/status/{workflow_id}",
        "dataset_id": dataset_id,
        "num_files": len(files)
    }


@app.get("/workflow/status/{workflow_id}")
async def get_workflow_status(workflow_id: str):
    workflows = load_workflows()
    if workflow_id not in workflows:
        raise HTTPException(404, f"Workflow {workflow_id} not found")
    return workflows[workflow_id]


@app.get("/workflow/list")
async def list_workflows():
    workflows = load_workflows()
    return {"workflows": list(workflows.values()), "count": len(workflows)}


@app.delete("/workflow/clear/all")
async def clear_all_workflows():
    """Clear all workflows"""
    workflows = load_workflows()
    count = len(workflows)
    save_workflows({})
    return {"message": f"Cleared {count} workflows"}


@app.delete("/workflow/{workflow_id}")
async def delete_workflow(workflow_id: str):
    """Delete a specific workflow"""
    workflows = load_workflows()
    if workflow_id not in workflows:
        raise HTTPException(404, f"Workflow {workflow_id} not found")
    del workflows[workflow_id]
    save_workflows(workflows)
    return {"message": f"Workflow {workflow_id} deleted"}


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    if job_id not in processing_jobs:
        raise HTTPException(404, f"Job {job_id} not found")
    return processing_jobs[job_id]


@app.get("/jobs")
async def list_jobs():
    return {"jobs": list(processing_jobs.values())}


@app.get("/datasets")
async def list_datasets():
    datasets = []
    if OUTPUT_DIR.exists():
        for d in OUTPUT_DIR.iterdir():
            if d.is_dir():
                images_dir = d / "images"
                sparse_dir = d / "sparse" / "0"
                num_images = len(list(images_dir.glob("*"))) if images_dir.exists() else 0
                has_sparse = sparse_dir.exists() and (sparse_dir / "cameras.txt").exists()
                datasets.append({
                    "name": d.name,
                    "num_images": num_images,
                    "has_reconstruction": has_sparse,
                })
    return {"datasets": datasets}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
