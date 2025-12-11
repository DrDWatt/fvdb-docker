"""
Simple USD Writer - Programmatic USD Scene Creation
Creates valid USDA (USD ASCII) files without requiring pxr package
Works on ARM64 and doesn't need OpenUSD binaries
"""

import logging
import numpy as np
from pathlib import Path
from plyfile import PlyData
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def write_usd_point_cloud(ply_path: Path, usd_path: Path, subsample_factor: int = 10) -> bool:
    """
    Write a USD (USDA) point cloud file programmatically
    Creates valid USD ASCII format that can be opened in any USD viewer
    
    Args:
        ply_path: Input PLY file path
        usd_path: Output USD file path (will create .usda file)
        subsample_factor: Downsample factor for performance (1 = full resolution)
    
    Returns:
        True if successful
    """
    try:
        logger.info(f"Converting {ply_path} to USD...")
        
        # Read PLY data
        plydata = PlyData.read(str(ply_path))
        vertex_data = plydata['vertex']
        
        # Extract positions
        x = np.array(vertex_data['x'])
        y = np.array(vertex_data['y'])
        z = np.array(vertex_data['z'])
        
        # Subsample for performance
        if subsample_factor > 1:
            x = x[::subsample_factor]
            y = y[::subsample_factor]
            z = z[::subsample_factor]
        
        num_points = len(x)
        logger.info(f"Processing {num_points:,} points...")
        
        # Extract colors
        colors = None
        try:
            if 'red' in vertex_data.data.dtype.names:
                r = np.array(vertex_data['red'])[::subsample_factor] / 255.0
                g = np.array(vertex_data['green'])[::subsample_factor] / 255.0
                b = np.array(vertex_data['blue'])[::subsample_factor] / 255.0
                colors = np.column_stack([r, g, b])
                logger.info("Extracted RGB colors")
            elif 'f_dc_0' in vertex_data.data.dtype.names:
                # Spherical harmonics
                sh_r = np.array(vertex_data['f_dc_0'])[::subsample_factor]
                sh_g = np.array(vertex_data['f_dc_1'])[::subsample_factor]
                sh_b = np.array(vertex_data['f_dc_2'])[::subsample_factor]
                C0 = 0.28209479177387814
                colors = np.column_stack([
                    sh_r / C0 + 0.5,
                    sh_g / C0 + 0.5,
                    sh_b / C0 + 0.5
                ])
                colors = np.clip(colors, 0, 1)
                logger.info("Converted spherical harmonics to RGB")
        except Exception as e:
            logger.warning(f"Could not extract colors: {e}")
        
        # Build USD ASCII content
        usd_content = f"""#usda 1.0
(
    defaultPrim = "GaussianSplatCloud"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "GaussianSplatCloud" (
    kind = "component"
)
{{
    def Points "points" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {{
        float3[] extent = [({x.min():.6f}, {y.min():.6f}, {z.min():.6f}), ({x.max():.6f}, {y.max():.6f}, {z.max():.6f})]
        point3f[] points = [
"""
        
        # Write points (in batches for better formatting)
        batch_size = 10
        for i in range(0, num_points, batch_size):
            end_idx = min(i + batch_size, num_points)
            batch_points = [
                f"({x[j]:.6f}, {y[j]:.6f}, {z[j]:.6f})"
                for j in range(i, end_idx)
            ]
            usd_content += "            " + ", ".join(batch_points)
            if end_idx < num_points:
                usd_content += ",\n"
            else:
                usd_content += "\n"
        
        usd_content += "        ]\n"
        
        # Add colors if available
        if colors is not None:
            usd_content += "        color3f[] primvars:displayColor = [\n"
            for i in range(0, num_points, batch_size):
                end_idx = min(i + batch_size, num_points)
                batch_colors = [
                    f"({colors[j,0]:.6f}, {colors[j,1]:.6f}, {colors[j,2]:.6f})"
                    for j in range(i, end_idx)
                ]
                usd_content += "            " + ", ".join(batch_colors)
                if end_idx < num_points:
                    usd_content += ",\n"
                else:
                    usd_content += "\n"
            usd_content += "        ]\n"
        
        # Add point widths for better visualization
        usd_content += f"""        float[] widths = [{', '.join(['0.01'] * num_points)}]
    }}
}}
"""
        
        # Write to file
        usd_file = usd_path.with_suffix('.usda')
        with open(usd_file, 'w') as f:
            f.write(usd_content)
        
        file_size_mb = usd_file.stat().st_size / (1024 * 1024)
        logger.info(f"✅ Created USD file: {usd_file}")
        logger.info(f"   Points: {num_points:,}")
        logger.info(f"   Colors: {'Yes' if colors is not None else 'No'}")
        logger.info(f"   Size: {file_size_mb:.2f} MB")
        logger.info(f"   Format: USDA (ASCII)")
        
        return True
        
    except Exception as e:
        logger.error(f"USD conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def write_usd_mesh(ply_path: Path, usd_path: Path) -> bool:
    """
    Write a basic USD mesh (for future implementation)
    Currently writes point cloud - mesh reconstruction can be added
    """
    logger.info("Mesh export - using point cloud format for now")
    return write_usd_point_cloud(ply_path, usd_path, subsample_factor=5)


if __name__ == "__main__":
    # Test conversion - HIGH QUALITY
    test_ply = Path("/workspace/data/models/counter_registry_test.ply")
    test_usd = Path("/workspace/data/outputs/counter_high_quality.usda")
    
    if test_ply.exists():
        logger.info("=" * 60)
        logger.info("Testing HIGH QUALITY PLY to USD conversion...")
        logger.info("=" * 60)
        success = write_usd_point_cloud(test_ply, test_usd, subsample_factor=2)  # Keep 50% of points
        if success:
            logger.info("=" * 60)
            logger.info(f"✅ SUCCESS! HIGH QUALITY USD file created")
            logger.info(f"📁 Output: {test_usd}")
            logger.info(f"🎯 Ready for Blender, SuperSplat, Omniverse")
            logger.info("=" * 60)
        else:
            logger.error("❌ Conversion failed")
    else:
        logger.warning(f"Test file not found: {test_ply}")
