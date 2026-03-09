#!/usr/bin/env python3
"""
Automated Gaussian Splat cleanup - removes floaters and artifacts.

Inspired by clean-gs (github.com/smlab-niser/clean-gs) Stage 3 outlier removal,
plus additional opacity and scale-based pruning.

This module operates on standard 3DGS PLY files and requires no semantic masks.

Techniques:
1. Low-opacity pruning: Remove near-invisible Gaussians
2. Large-scale pruning: Remove oversized smeared blobs
3. Spatial outlier removal: Remove Gaussians far from scene center
4. k-NN neighbor isolation: Remove scattered/isolated noise Gaussians
"""

import logging
import numpy as np
from pathlib import Path
from plyfile import PlyData, PlyElement

logger = logging.getLogger(__name__)


def clean_gaussians(
    input_ply: str,
    output_ply: str,
    opacity_threshold: float = 0.005,
    scale_percentile: float = 99.5,
    spatial_percentile: float = 99.0,
    neighbor_percentile: float = 95.0,
    k_neighbors: int = 10,
) -> dict:
    """
    Clean a Gaussian Splat PLY by removing floaters and artifacts.

    Args:
        input_ply: Path to input PLY file
        output_ply: Path to write cleaned PLY file
        opacity_threshold: Remove Gaussians with opacity below this (default 0.005)
        scale_percentile: Remove Gaussians with scale above this percentile (default 99.5)
        spatial_percentile: Remove Gaussians beyond this distance percentile from center (default 99.0)
        neighbor_percentile: Remove Gaussians with avg k-NN distance above this percentile (default 95.0)
        k_neighbors: Number of neighbors for k-NN isolation detection (default 10)

    Returns:
        dict with cleanup statistics
    """
    logger.info(f"Loading Gaussians from {input_ply}...")
    plydata = PlyData.read(input_ply)
    vertex = plydata['vertex']
    num_original = len(vertex)

    if num_original == 0:
        logger.warning("Empty PLY file, nothing to clean")
        return {"original": 0, "final": 0, "removed": 0}

    # Extract positions
    xyz = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1)

    # Track which Gaussians to keep
    keep_mask = np.ones(num_original, dtype=bool)
    stats = {"original": num_original}

    # --- Stage 1: Low-opacity pruning ---
    # Opacity is stored as sigmoid-inverse, so apply sigmoid
    if 'opacity' in vertex.data.dtype.names:
        raw_opacity = np.array(vertex['opacity'])
        opacity = 1.0 / (1.0 + np.exp(-raw_opacity))  # sigmoid
        low_opacity = opacity < opacity_threshold
        num_low_opacity = int(np.sum(low_opacity))
        keep_mask &= ~low_opacity
        stats["low_opacity_removed"] = num_low_opacity
        logger.info(f"  [Stage 1] Low-opacity pruning: removed {num_low_opacity:,} "
                     f"(opacity < {opacity_threshold})")
    else:
        stats["low_opacity_removed"] = 0
        logger.info("  [Stage 1] No opacity field found, skipping")

    # --- Stage 2: Large-scale pruning ---
    # Scale stored as log-space in scale_0, scale_1, scale_2
    scale_names = [f'scale_{i}' for i in range(3)]
    if all(name in vertex.data.dtype.names for name in scale_names):
        log_scales = np.stack([np.array(vertex[s]) for s in scale_names], axis=1)
        scales = np.exp(log_scales)
        max_scale = np.max(scales, axis=1)
        # Only compute threshold on currently-kept Gaussians
        kept_scales = max_scale[keep_mask]
        if len(kept_scales) > 0:
            scale_thresh = np.percentile(kept_scales, scale_percentile)
            large_scale = max_scale > scale_thresh
            num_large = int(np.sum(large_scale & keep_mask))
            keep_mask &= ~large_scale
            stats["large_scale_removed"] = num_large
            stats["scale_threshold"] = float(scale_thresh)
            logger.info(f"  [Stage 2] Large-scale pruning: removed {num_large:,} "
                         f"(max_scale > {scale_thresh:.4f}, {scale_percentile}th percentile)")
    else:
        stats["large_scale_removed"] = 0
        logger.info("  [Stage 2] No scale fields found, skipping")

    # --- Stage 3: Spatial outlier removal ---
    kept_xyz = xyz[keep_mask]
    if len(kept_xyz) > 100:
        center = np.mean(kept_xyz, axis=0)
        distances = np.linalg.norm(xyz - center, axis=1)
        kept_distances = distances[keep_mask]
        dist_thresh = np.percentile(kept_distances, spatial_percentile)
        spatial_outliers = distances > dist_thresh
        num_spatial = int(np.sum(spatial_outliers & keep_mask))
        keep_mask &= ~spatial_outliers
        stats["spatial_removed"] = num_spatial
        stats["spatial_threshold"] = float(dist_thresh)
        logger.info(f"  [Stage 3] Spatial outlier removal: removed {num_spatial:,} "
                     f"(distance > {dist_thresh:.2f}, {spatial_percentile}th percentile)")
    else:
        stats["spatial_removed"] = 0
        logger.info("  [Stage 3] Too few Gaussians for spatial filtering")

    # --- Stage 4: k-NN neighbor-based isolation removal ---
    kept_indices = np.where(keep_mask)[0]
    kept_xyz = xyz[keep_mask]
    if len(kept_xyz) > k_neighbors + 1:
        try:
            from sklearn.neighbors import NearestNeighbors
            n_neighbors = min(k_neighbors, len(kept_xyz) - 1)
            nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(kept_xyz)
            distances_knn, _ = nbrs.kneighbors(kept_xyz)
            avg_neighbor_dist = np.mean(distances_knn[:, 1:], axis=1)  # skip self
            knn_thresh = np.percentile(avg_neighbor_dist, neighbor_percentile)
            isolated = avg_neighbor_dist > knn_thresh
            isolated_indices = kept_indices[isolated]
            keep_mask[isolated_indices] = False
            num_isolated = len(isolated_indices)
            stats["knn_isolated_removed"] = num_isolated
            stats["knn_threshold"] = float(knn_thresh)
            logger.info(f"  [Stage 4] k-NN isolation removal: removed {num_isolated:,} "
                         f"(avg_dist > {knn_thresh:.4f}, {neighbor_percentile}th percentile)")
        except ImportError:
            stats["knn_isolated_removed"] = 0
            logger.warning("  [Stage 4] scikit-learn not available, skipping k-NN filtering")
    else:
        stats["knn_isolated_removed"] = 0
        logger.info("  [Stage 4] Too few Gaussians for k-NN filtering")

    # --- Save cleaned PLY ---
    num_final = int(np.sum(keep_mask))
    num_removed = num_original - num_final
    pct_removed = 100.0 * num_removed / num_original if num_original > 0 else 0

    stats["final"] = num_final
    stats["removed"] = num_removed
    stats["pct_removed"] = round(pct_removed, 1)

    logger.info(f"  Final: {num_final:,} / {num_original:,} kept "
                 f"({num_removed:,} removed, {pct_removed:.1f}%)")

    # Write cleaned PLY preserving ALL original elements, comments, and metadata
    # Only the vertex element is filtered; all others (camera matrices, etc.) are kept as-is
    vertex_kept = vertex[keep_mask]
    el = PlyElement.describe(vertex_kept, 'vertex')
    elements = [el] + [e for e in plydata.elements if e.name != 'vertex']
    out = PlyData(elements, text=plydata.text)
    out.comments = plydata.comments
    out.obj_info = plydata.obj_info
    out.write(output_ply)

    file_size_mb = Path(output_ply).stat().st_size / (1024 * 1024)
    stats["file_size_mb"] = round(file_size_mb, 1)
    logger.info(f"  Saved cleaned PLY: {output_ply} ({file_size_mb:.1f} MB)")

    return stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Clean Gaussian Splat PLY files")
    parser.add_argument("--input", required=True, help="Input PLY file")
    parser.add_argument("--output", required=True, help="Output cleaned PLY file")
    parser.add_argument("--opacity_threshold", type=float, default=0.005)
    parser.add_argument("--scale_percentile", type=float, default=99.5)
    parser.add_argument("--spatial_percentile", type=float, default=99.0)
    parser.add_argument("--neighbor_percentile", type=float, default=95.0)
    parser.add_argument("--k_neighbors", type=int, default=10)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    stats = clean_gaussians(
        args.input, args.output,
        opacity_threshold=args.opacity_threshold,
        scale_percentile=args.scale_percentile,
        spatial_percentile=args.spatial_percentile,
        neighbor_percentile=args.neighbor_percentile,
        k_neighbors=args.k_neighbors,
    )
    print(f"\nCleanup stats: {stats}")
