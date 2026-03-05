"""
cuVSLAM Processing Service
Drop-in replacement for COLMAP service using NVIDIA cuVSLAM for visual SLAM.
Processes stereo frame pairs to produce COLMAP-compatible sparse reconstruction.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict
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


def write_colmap_sparse(sparse_dir: Path, image_names: List[str],
                        poses: List, fx: float, fy: float,
                        cx: float, cy: float, width: int, height: int,
                        left_dir: Path = None):
    """Write COLMAP-format sparse reconstruction files (text format).
    
    Creates cameras.txt, images.txt, points3D.txt in the sparse directory.
    If left_dir is provided, triangulates sparse 3D points from consecutive frames.
    """
    sparse_dir.mkdir(parents=True, exist_ok=True)

    # Triangulate sparse 3D points if image directory is available
    points3d = {}
    image_points2d = {}
    if left_dir is not None and left_dir.exists():
        try:
            points3d, image_points2d = triangulate_sparse_points(
                left_dir, image_names, poses, fx, fy, cx, cy
            )
            logger.info(f"Triangulated {len(points3d)} sparse 3D points")
        except Exception as e:
            logger.warning(f"Triangulation failed, writing empty points3D: {e}")

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
                          camera_params: dict = None):
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

    # Write COLMAP-format sparse output (with triangulated 3D points)
    sparse_dir = output_dir / "sparse" / "0"
    write_colmap_sparse(sparse_dir, image_names, poses,
                        fx, fy, cx, cy, w, h, left_dir=left_dir)

    return valid_poses, num_pairs


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

    # Save video to temp
    video_path = TEMP_DIR / f"{dataset_id}_{file.filename}"
    content = await file.read()
    with open(video_path, "wb") as f:
        f.write(content)

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
            processing_jobs[job_id]["message"] = "Matching features..."
            update_workflow(workflow_id, {"progress": 0.5})

            # Feature matching
            if matcher == "exhaustive":
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

            result = await asyncio.to_thread(subprocess.run, cmd_match, capture_output=True, text=True, timeout=1800, env=env)
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

            # Poll training status until complete
            if training_job_id:
                max_wait = 3600
                elapsed = 0
                async with httpx.AsyncClient(timeout=30.0) as client:
                    while elapsed < max_wait:
                        await asyncio.sleep(10)
                        elapsed += 10

                        try:
                            resp = await client.get(
                                f"{training_url}/jobs/{training_job_id}"
                            )
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
                                    logger.info(f"[{workflow_id}] Training complete")
                                    break
                                elif tstatus == "failed":
                                    raise Exception(f"Training failed: {tmessage}")
                        except httpx.ConnectError:
                            logger.warning(f"[{workflow_id}] Training status check failed, retrying...")

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
            images_dir.mkdir(exist_ok=True, parents=True)
            images_right_dir.mkdir(exist_ok=True, parents=True)

            num_left = 0
            num_right = 0
            camera_params = {}

            for file in files:
                filename_lower = file.filename.lower()

                if filename_lower.endswith('.zip'):
                    content = await file.read()
                    zip_path = output_dir / "upload.zip"
                    with open(zip_path, "wb") as f:
                        f.write(content)

                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        for zip_info in zip_ref.filelist:
                            zname = zip_info.filename.lower()
                            if not zname.endswith(('.jpg', '.jpeg', '.png')):
                                continue

                            extracted = zip_ref.read(zip_info.filename)

                            # Route based on directory in ZIP
                            if 'right' in zip_info.filename.lower() or 'images_right' in zip_info.filename.lower():
                                img_path = images_right_dir / f"frame_{num_right:04d}.jpg"
                                with open(img_path, "wb") as img_f:
                                    img_f.write(extracted)
                                num_right += 1
                            else:
                                img_path = images_dir / f"frame_{num_left:04d}.jpg"
                                with open(img_path, "wb") as img_f:
                                    img_f.write(extracted)
                                num_left += 1

                        # Check for camera_params.json in ZIP
                        try:
                            params_data = zip_ref.read("camera_params.json")
                            camera_params = json.loads(params_data)
                            logger.info(f"[{workflow_id}] Loaded camera params from ZIP: {camera_params}")
                        except (KeyError, json.JSONDecodeError):
                            pass

                    zip_path.unlink()

                elif filename_lower.endswith(('.jpg', '.jpeg', '.png')):
                    content = await file.read()
                    if 'right' in file.filename.lower():
                        img_path = images_right_dir / f"frame_{num_right:04d}.jpg"
                        with open(img_path, "wb") as f:
                            f.write(content)
                        num_right += 1
                    else:
                        img_path = images_dir / f"frame_{num_left:04d}.jpg"
                        with open(img_path, "wb") as f:
                            f.write(content)
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
            if filter_blur.lower() == "true":
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

            if num_left < 3:
                raise Exception(f"Need at least 3 left images, got {num_left}")

            if not CUVSLAM_AVAILABLE:
                raise Exception("cuVSLAM library not available in this container")

            if num_right == 0:
                raise Exception(
                    "No right stereo frames found. cuVSLAM requires stereo pairs. "
                    "Include images_right/ directory in ZIP or files with 'right' in name."
                )

            # Run cuVSLAM
            job_id = f"cuvslam_{dataset_id}_{datetime.now().strftime('%H%M%S')}"
            processing_jobs[job_id] = {
                "job_id": job_id,
                "dataset_id": dataset_id,
                "status": "processing",
                "progress": 0.0,
                "message": "Starting cuVSLAM processing",
                "started_at": datetime.now().isoformat()
            }
            update_workflow(workflow_id, {
                "colmap_job_id": job_id,
                "current_step": "Running cuVSLAM reconstruction",
                "progress": 0.3
            })

            valid_poses, total_frames = run_cuvslam_on_frames(
                images_dir, images_right_dir, output_dir,
                workflow_id, camera_params
            )

            processing_jobs[job_id]["status"] = "completed"
            processing_jobs[job_id]["progress"] = 1.0
            processing_jobs[job_id]["message"] = "cuVSLAM processing complete"
            processing_jobs[job_id]["num_images"] = valid_poses
            processing_jobs[job_id]["completed_at"] = datetime.now().isoformat()

            update_workflow(workflow_id, {
                "current_step": f"cuVSLAM complete ({valid_poses}/{total_frames} frames)",
                "progress": 0.7
            })

            logger.info(f"[{workflow_id}] cuVSLAM processing complete")

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

            # Poll training status until complete
            if training_job_id:
                max_wait = 3600  # 1 hour max
                elapsed = 0
                async with httpx.AsyncClient(timeout=30.0) as client:
                    while elapsed < max_wait:
                        await asyncio.sleep(10)
                        elapsed += 10

                        try:
                            resp = await client.get(
                                f"{training_url}/jobs/{training_job_id}"
                            )
                            if resp.status_code == 200:
                                tdata = resp.json()
                                tstatus = tdata.get("status", "")
                                tprogress = tdata.get("progress", 0)
                                tmessage = tdata.get("message", "")

                                # Map training progress (0-1) to our range (0.75-0.95)
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
                                    logger.info(f"[{workflow_id}] Training complete")
                                    break
                                elif tstatus == "failed":
                                    raise Exception(f"Training failed: {tmessage}")
                        except httpx.ConnectError:
                            logger.warning(f"[{workflow_id}] Training status check failed, retrying...")

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
