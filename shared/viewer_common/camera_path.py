"""Smooth, constant-speed flythrough paths built from trained camera poses.

Pure numpy so it can be shared by the fVDB viewer (:8085, GPU-rendered frames)
and the SuperSplat viewer (:8086, browser-rendered animation track).

Raw trained cameras come from handheld captures: they are unevenly spaced and
jittery, so stepping through them directly produces a choppy, variable-speed
flythrough. build_smooth_path() Gaussian-smooths positions/rotations
(edge-clamped, no wrap back to the first camera), densely resamples the spline
and parameterizes it by arc length so sample_path() moves at constant speed.
"""
import math

import numpy as np

# Minimum length of every exported flythrough video (seconds)
MIN_VIDEO_SECONDS = 30


def slerp_rotation(R1, R2, t):
    """Interpolate two 3x3 rotations; SVD re-orthogonalization keeps a proper rotation."""
    U, _, Vt = np.linalg.svd((1 - t) * R1 + t * R2)
    diag = np.array([1, 1, np.sign(np.linalg.det(U @ Vt))])
    return U @ np.diag(diag) @ Vt


def catmull_rom(p0, p1, p2, p3, t):
    """Catmull-Rom spline interpolation for smooth position curves."""
    return 0.5 * (
        2 * p1 +
        (-p0 + p2) * t +
        (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t +
        (-p0 + 3 * p1 - 3 * p2 + p3) * t * t * t
    )


def ease_in_out(u):
    """Cosine ease so a flythrough starts and stops gently instead of snapping."""
    return 0.5 - 0.5 * math.cos(math.pi * min(max(u, 0.0), 1.0))


def build_smooth_path(c2w, K, sizes, depths=None, subsamples=16):
    """Build a smoothed, densely resampled, arc-length-parameterized camera path.

    Args:
        c2w: (N, 4, 4) camera-to-world matrices (OpenCV convention).
        K: (N, 3, 3) projection (intrinsics) matrices in pixels.
        sizes: (N, 2) source image sizes as (height, width).
        depths: optional (N,) median scene depth per camera (used as look-at distance).
    Returns:
        dict with dense arrays "pos", "rot", "K_norm" (intrinsics normalized by
        image size), "depth" and normalized cumulative arc length "arc" in [0, 1].
    """
    c2w = np.asarray(c2w, dtype=np.float64)
    K_src = np.asarray(K, dtype=np.float64)
    sizes = np.asarray(sizes, dtype=np.float64)
    depth_src = None if depths is None else np.asarray(depths, dtype=np.float64).reshape(-1)

    # Drop degenerate/non-finite poses so one bad camera can't break the whole path
    with np.errstate(all='ignore'):
        valid = np.isfinite(c2w).all(axis=(1, 2)) & (np.abs(np.linalg.det(c2w[:, :3, :3]) - 1) < 0.1)
        if depth_src is not None:
            valid &= np.isfinite(depth_src) & (depth_src > 0)
    if valid.sum() < 2:
        raise ValueError("fewer than 2 valid trained cameras for flythrough path")
    c2w, K_src, sizes = c2w[valid], K_src[valid], sizes[valid]
    if depth_src is not None:
        depth_src = depth_src[valid]
    n = c2w.shape[0]
    pos, rot = c2w[:, :3, 3], c2w[:, :3, :3]

    # Gaussian smoothing window (sigma ~n/30 cameras), clamped at the path ends.
    # Wider windows cut corners on orbit-style captures and pull the camera
    # inside the subject; narrower ones keep handheld jitter.
    sigma = max(1.0, n / 30.0)
    offsets = np.arange(-int(3 * sigma), int(3 * sigma) + 1)
    weights = np.exp(-0.5 * (offsets / sigma) ** 2)
    weights /= weights.sum()
    win = np.clip(np.arange(n)[:, None] + offsets[None, :], 0, n - 1)
    pos_s = np.einsum('w,nwc->nc', weights, pos[win])
    # Chordal rotation mean: weighted matrix average projected back onto SO(3)
    U, _, Vt = np.linalg.svd(np.einsum('w,nwij->nij', weights, rot[win]))
    D = np.ones((n, 3))
    D[:, 2] = np.sign(np.linalg.det(U @ Vt))
    rot_s = (U * D[:, None, :]) @ Vt
    depth_s = None if depth_src is None else np.einsum('w,nw->n', weights, depth_src[win])

    # Intrinsics normalized by source image size so they can be blended and rescaled
    K_norm = K_src.copy()
    K_norm[:, 0, :] /= sizes[:, 1:2]
    K_norm[:, 1, :] /= sizes[:, 0:1]

    # Densely resample the smoothed spline (Catmull-Rom + SLERP) so arc length is
    # measured on the actual curve; uniform Catmull-Rom alone varies speed inside
    # segments when camera spacing is uneven.
    d_pos, d_rot, d_K, d_depth = [], [], [], []
    for i in range(n - 1):
        i0, i3 = max(i - 1, 0), min(i + 2, n - 1)
        for s in range(subsamples):
            f = s / subsamples
            d_pos.append(catmull_rom(pos_s[i0], pos_s[i], pos_s[i + 1], pos_s[i3], f))
            d_rot.append(slerp_rotation(rot_s[i], rot_s[i + 1], f))
            d_K.append((1 - f) * K_norm[i] + f * K_norm[i + 1])
            if depth_s is not None:
                d_depth.append((1 - f) * depth_s[i] + f * depth_s[i + 1])
    d_pos.append(pos_s[-1])
    d_rot.append(rot_s[-1])
    d_K.append(K_norm[-1])
    if depth_s is not None:
        d_depth.append(depth_s[-1])
    pos_s, rot_s = np.array(d_pos), np.array(d_rot)

    # Arc length mixes translation and rotation so pure pans still advance smoothly
    seg_pos = np.linalg.norm(np.diff(pos_s, axis=0), axis=1)
    rel = np.einsum('nji,njk->nik', rot_s[:-1], rot_s[1:])
    seg_rot = np.arccos(np.clip((np.trace(rel, axis1=1, axis2=2) - 1) / 2, -1, 1))

    def _normalize(d):
        total = d.sum()
        return d / total if total > 1e-9 else np.zeros_like(d)

    seg = 0.5 * _normalize(seg_pos) + 0.5 * _normalize(seg_rot)
    if seg.sum() < 1e-9:
        seg = np.ones(len(pos_s) - 1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    arc /= arc[-1]

    return {
        "pos": pos_s,
        "rot": rot_s,
        "K_norm": np.array(d_K),
        "depth": np.array(d_depth) if d_depth else None,
        "arc": arc,
        "num_cameras": n,
        "sigma": sigma,
    }


def sample_path(path, u):
    """Sample the path at normalized arc position u in [0, 1] (constant speed).

    Returns (position (3,), rotation (3, 3), normalized intrinsics (3, 3), depth or None).
    """
    arc = path["arc"]
    i1 = int(np.clip(np.searchsorted(arc, u, side='right') - 1, 0, len(arc) - 2))
    i2 = i1 + 1
    span = arc[i2] - arc[i1]
    frac = float((u - arc[i1]) / span) if span > 1e-12 else 0.0
    # Dense samples are close together, so lerp/SLERP between neighbours is exact enough
    pos = (1 - frac) * path["pos"][i1] + frac * path["pos"][i2]
    rot = slerp_rotation(path["rot"][i1], path["rot"][i2], frac)
    K_norm = (1 - frac) * path["K_norm"][i1] + frac * path["K_norm"][i2]
    depth = None
    if path.get("depth") is not None:
        depth = float((1 - frac) * path["depth"][i1] + frac * path["depth"][i2])
    return pos, rot, K_norm, depth


def frames_for_duration(duration_s, fps):
    """Frame count for a video of the given duration, enforcing the minimum length."""
    return int(round(max(float(duration_s), MIN_VIDEO_SECONDS) * fps))
