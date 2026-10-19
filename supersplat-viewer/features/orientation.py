"""Viewer orientation for PLYs without trained cameras (Inria/COLMAP-style "point_cloud_*.ply").

The SuperSplat viewer always loads splats rotated (0, 0, 180), which is right for fVDB exports whose up
is PLY -y. Scenes without cameras may use any up axis: measured 2026-10-18, Frankfurt street and the Graz
tower are both Z-up, so the default orbit looked straight down (Graz) or sideways (Frankfurt).

Up is estimated from the splat itself, with two independent cues that must agree (<= MAX_DISAGREE_DEG):
  1. surface normals: the thinnest axis of flat Gaussians (min scale 3x smaller than the middle one);
     the normal structure tensor's dominant eigenvector is the ground/roof direction;
  2. extent: the smallest principal axis of the (5-95 % trimmed) Gaussian centres (scenes are wider
     than tall).
Sign: an axis within 30 deg of PLY Z is taken as +Z (georeferenced/ENU exports are Z-up; Graz is a hilltop
whose height distribution misleads); otherwise the longer tail of the height distribution points up
(buildings and trees rise above the ground). If the cues disagree, the viewer default is kept.

Result: rotation R (viewer = R @ ply) as a quaternion for the patched viewer's ?orient= parameter. With
up -y (fVDB) the result equals the viewer default (0, 0, 180).
"""
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np

logger = logging.getLogger("supersplat-viewer.orientation")

DEFAULT = np.diag([-1.0, -1.0, 1.0])  # viewer default: Euler (0, 0, 180)
SAMPLES = 300_000
MAX_DISAGREE_DEG = 30.0
FLAT_RATIO_LOG = np.log(3.0)
Z_ALIGNED = np.cos(np.radians(30))


def _quat_to_normals(q: np.ndarray, log_scales: np.ndarray) -> np.ndarray:
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    R = np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
                  np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
                  np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], 1)
    return R[np.arange(len(R)), :, np.argmin(log_scales, 1)]


def estimate_up(vertex) -> tuple[np.ndarray | None, dict]:
    """Up axis (PLY frame) from Gaussian centres + rotations/scales, or None with the reason."""
    step = max(1, len(vertex) // SAMPLES)
    xyz = np.stack([np.asarray(vertex[k][::step], np.float64) for k in "xyz"], 1)
    ok = np.isfinite(xyz).all(1)
    lo, hi = np.percentile(xyz[ok], 5, 0), np.percentile(xyz[ok], 95, 0)
    inside = ok & np.all((xyz >= lo) & (xyz <= hi), 1)
    P = xyz[inside]
    pts_axis = np.linalg.eigh(np.cov((P - P.mean(0)).T))[1][:, 0]
    info: dict = {"samples": int(inside.sum())}
    names = vertex.dtype.names
    if not {"rot_0", "rot_1", "rot_2", "rot_3", "scale_0", "scale_1", "scale_2"} <= set(names):
        return None, {**info, "reason": "no rotation/scale properties"}
    q = np.stack([np.asarray(vertex[f"rot_{i}"][::step], np.float64) for i in range(4)], 1)[inside]
    s = np.stack([np.asarray(vertex[f"scale_{i}"][::step], np.float64) for i in range(3)], 1)[inside]
    srt = np.sort(s, 1)
    flat = (srt[:, 1] - srt[:, 0]) > FLAT_RATIO_LOG
    if flat.sum() < 1000:
        return None, {**info, "reason": f"only {int(flat.sum())} flat Gaussians"}
    n = _quat_to_normals(q[flat], s[flat])
    nrm_axis = np.linalg.eigh(n.T @ n / len(n))[1][:, -1]
    angle = float(np.degrees(np.arccos(min(1.0, abs(float(nrm_axis @ pts_axis))))))
    info.update(flat_gaussians=int(flat.sum()), cue_angle_deg=round(angle, 1))
    if angle > MAX_DISAGREE_DEG:
        return None, {**info, "reason": f"normal and extent cues disagree by {angle:.0f} deg"}
    up = nrm_axis / np.linalg.norm(nrm_axis)
    if abs(up[2]) >= Z_ALIGNED:
        # Georeferenced / ENU-aligned exports are Z-up. Measured 2026-10-18: the Graz hilltop has its dense
        # plaza on top and the slope below, so a height-distribution test picks the wrong sign there.
        up, sign_rule = up * np.sign(up[2]), "PLY +Z (axis within 30 deg of Z: georeferenced Z-up convention)"
    else:
        h = (P - np.median(P, 0)) @ up
        if np.percentile(h, 98) + np.percentile(h, 2) < 0:  # longer tail (buildings, trees) points up
            up = -up
        sign_rule = "longer tail of the height distribution points up"
    return up, {**info, "method": "flat-Gaussian normals + extent PCA", "sign_rule": sign_rule}


def _align(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimal rotation taking unit vector a to unit vector b (Rodrigues)."""
    v, c = np.cross(a, b), float(a @ b)
    if c > 1 - 1e-12:
        return np.eye(3)
    if c < -1 + 1e-12:  # opposite: rotate 180 deg about any axis orthogonal to a
        axis = np.cross(a, [1.0, 0, 0] if abs(a[0]) < 0.9 else [0, 1.0, 0])
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


def rotation_for_up(up: np.ndarray) -> np.ndarray:
    """viewer = R @ ply with R @ up = +Y; starts from the viewer default so fVDB scenes are unchanged."""
    return _align(DEFAULT @ up, np.array([0.0, 1.0, 0.0])) @ DEFAULT


def quat_xyzw(R: np.ndarray) -> list[float]:
    t = np.trace(R)
    if t > 0:
        s = 2 * np.sqrt(t + 1)
        q = [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, s / 4]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2 * np.sqrt(1 + R[i, i] - R[j, j] - R[k, k])
        q = [0.0] * 4
        q[i], q[j], q[k] = s / 4, (R[j, i] + R[i, j]) / s, (R[k, i] + R[i, k]) / s
        q[3] = (R[k, j] - R[j, k]) / s
    q = np.asarray(q)
    return (q / np.linalg.norm(q)).round(6).tolist()


@lru_cache(maxsize=16)
def _orientation(path: str, mtime_ns: int, has_cameras: bool):
    if has_cameras:
        return None  # fVDB export: the viewer default is correct
    from .flythrough import _read_elements  # noqa: PLC0415  (shared PLY reader)
    up, info = estimate_up(_read_elements(Path(path))[0]["vertex"])
    if up is None:
        logger.info(f"Orientation for {Path(path).name}: viewer default ({info.get('reason')})")
        return None
    R = rotation_for_up(up)
    logger.info(f"Orientation for {Path(path).name}: up {np.round(up, 3).tolist()} ({info})")
    return {"quat": quat_xyzw(R), "matrix": R.tolist(), "up_ply": up.tolist(), **info}


def orientation(path: Path, has_cameras: bool):
    """Cached per (file, mtime). None -> keep the viewer default."""
    return _orientation(str(path), path.stat().st_mtime_ns, has_cameras)
