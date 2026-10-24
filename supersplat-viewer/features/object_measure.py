"""3D size (model units) of a SAM3-segmented object, for scale calibration against its catalogue size.

The Gaussians projecting into the mask include whatever is behind the object (shelf back, wall). The object
is the front-most depth band: depths from the 5th percentile up to that plus BAND_WIDTHS x the object's width
at that depth (mask width in pixels x depth / fx). Its height is the 1st-99th percentile extent along the
viewer's up axis (+Y in the viewer = R^T e_y in the PLY, R from the splat entity's model matrix), width likewise
along the camera's right axis. Points p/q on the centre line give dist.spatial = height exactly.

Uncertainty: 2 px silhouette blur at the object's depth per end (same assumption as Blueprints' object method).
"""
import numpy as np

from .flythrough import _read_elements

BAND_WIDTHS = 1.0
MIN_POINTS = 50
EDGE_PX = 2.0


def _mat(m) -> np.ndarray:
    return np.asarray(m, dtype=np.float64).reshape(4, 4).T  # PlayCanvas column-major -> row-major


def measure(model_path, mask: np.ndarray, camera: dict) -> dict:
    while mask.ndim > 2:
        mask = mask[0]
    binary = mask > 0.5
    H, W = binary.shape
    ys, xs = np.nonzero(binary)
    if len(xs) < 25:
        raise ValueError("mask is empty")
    view, proj = _mat(camera["view"]), _mat(camera["proj"])
    model = _mat(camera["model"]) if camera.get("model") else np.eye(4)
    try:
        v = _read_elements(model_path)[0]["vertex"]
    except (KeyError, ValueError) as e:  # e.g. SuperSplat compressed PLY (packed 'chunk' element, list properties)
        raise ValueError(f"unsupported PLY layout for 3D measuring ({e}); use a standard 3DGS PLY") from None
    if not {"x", "y", "z"} <= set(v.dtype.names or ()):
        raise ValueError("PLY vertices have no x/y/z")
    P = np.stack([np.asarray(v[k], dtype=np.float64) for k in "xyz"], axis=1)
    op = 1 / (1 + np.exp(-np.asarray(v["opacity"], dtype=np.float64))) if "opacity" in v.dtype.names else np.ones(len(P))
    keep = np.isfinite(P).all(axis=1) & (op > 0.3)            # faint floaters do not define a silhouette
    Ph = np.c_[P, np.ones(len(P))]
    cam = Ph @ (view @ model).T                                  # camera space (PlayCanvas: looks down -Z)
    clip = cam @ proj.T
    w = clip[:, 3]
    front = keep & (w > 1e-8) & (cam[:, 2] < 0)
    sw = np.where(front, w, 1.0)
    u = (clip[:, 0] / sw * 0.5 + 0.5) * (W - 1)
    vv = (1.0 - (clip[:, 1] / sw * 0.5 + 0.5)) * (H - 1)
    on = front & (u >= 0) & (u <= W - 1) & (vv >= 0) & (vv <= H - 1)
    idx = np.flatnonzero(on)
    inside = idx[binary[vv[idx].astype(int), u[idx].astype(int)]]
    if len(inside) < MIN_POINTS:
        raise ValueError(f"only {len(inside)} splats project into the mask")
    depth = -cam[inside, 2]
    fx = proj[0, 0] * W / 2
    d0 = float(np.percentile(depth, 5))
    width_world = (xs.max() - xs.min() + 1) * d0 / fx
    band = inside[depth <= d0 + BAND_WIDTHS * width_world]
    if len(band) < MIN_POINTS:
        raise ValueError(f"only {len(band)} splats in the object's depth band")
    R = model[:3, :3]
    up = R.T @ np.array([0.0, 1.0, 0.0])                         # viewer +Y in PLY coordinates
    right = R.T @ (np.linalg.inv(view)[:3, :3] @ np.array([1.0, 0.0, 0.0]))
    pts = P[band]
    h, r = pts @ up, pts @ right
    h_lo, h_hi = np.percentile(h, [1, 99])
    r_lo, r_hi = np.percentile(r, [1, 99])
    c = pts.mean(axis=0)
    p = c + (h_lo - c @ up) * up
    q = c + (h_hi - c @ up) * up
    depth_band = float(np.median(-cam[band, 2]))
    return {"height_mu": float(h_hi - h_lo), "width_mu": float(r_hi - r_lo), "p": p.tolist(), "q": q.tolist(),
            "sigma_mu": float(EDGE_PX * depth_band / fx), "n_points": int(len(band)), "n_in_mask": int(len(inside)),
            "up_ply": up.tolist(), "depth": depth_band}
