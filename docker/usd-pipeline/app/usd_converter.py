"""
USD Conversion using Open3D
Programmatic USD scene creation for ARM64
"""

import logging
import numpy as np
from pathlib import Path
from plyfile import PlyData
import open3d as o3d

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ply_to_pointcloud_usd(ply_path: Path, usd_path: Path) -> bool:
    """
    Convert PLY to USD using Open3D
    Creates a USD point cloud scene programmatically
    """
    try:
        logger.info(f"Converting {ply_path} to USD format...")
        
        # Load PLY with plyfile
        plydata = PlyData.read(str(ply_path))
        vertex_data = plydata['vertex']
        
        # Extract positions
        points = np.column_stack([
            vertex_data['x'],
            vertex_data['y'],
            vertex_data['z']
        ])
        
        # Extract colors if available
        colors = None
        try:
            if 'red' in vertex_data.data.dtype.names:
                colors = np.column_stack([
                    vertex_data['red'],
                    vertex_data['green'],
                    vertex_data['blue']
                ]) / 255.0
            elif 'f_dc_0' in vertex_data.data.dtype.names:
                # Spherical harmonics - convert DC component to RGB
                sh_r = np.array(vertex_data['f_dc_0'])
                sh_g = np.array(vertex_data['f_dc_1'])
                sh_b = np.array(vertex_data['f_dc_2'])
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
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Estimate normals for better USD visualization
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        
        logger.info(f"Created point cloud with {len(points)} points")
        
        # Export to USD using Open3D
        # Open3D supports USD export natively
        success = o3d.io.write_point_cloud(str(usd_path), pcd)
        
        if success:
            logger.info(f"✅ Successfully exported to USD: {usd_path}")
            logger.info(f"   Points: {len(points):,}")
            if colors is not None:
                logger.info(f"   Colors: Yes")
            logger.info(f"   Normals: Yes")
            return True
        else:
            logger.error("Failed to write USD file")
            return False
            
    except Exception as e:
        logger.error(f"USD conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def ply_to_mesh_usd(ply_path: Path, usd_path: Path, voxel_size: float = 0.01) -> bool:
    """
    Convert PLY to USD mesh using surface reconstruction
    Better for visualization than raw points
    """
    try:
        logger.info(f"Converting {ply_path} to USD mesh...")
        
        # Load with Open3D directly
        pcd = o3d.io.read_point_cloud(str(ply_path))
        logger.info(f"Loaded {len(pcd.points)} points")
        
        # Downsample for faster processing
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        logger.info(f"Downsampled to {len(pcd_down.points)} points")
        
        # Estimate normals
        pcd_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
        )
        
        # Surface reconstruction using Poisson
        logger.info("Running Poisson surface reconstruction...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd_down, depth=9
        )
        
        # Remove low-density vertices
        vertices_to_remove = densities < np.quantile(densities, 0.01)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        
        logger.info(f"Created mesh with {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
        
        # Export mesh to USD
        success = o3d.io.write_triangle_mesh(str(usd_path), mesh)
        
        if success:
            logger.info(f"✅ Successfully exported mesh to USD: {usd_path}")
            return True
        else:
            logger.error("Failed to write USD mesh")
            return False
            
    except Exception as e:
        logger.error(f"USD mesh conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Test conversion
    test_ply = Path("/workspace/data/models/counter_registry_test.ply")
    test_usd = Path("/workspace/data/outputs/counter_pointcloud.usd")
    
    if test_ply.exists():
        logger.info("Testing PLY to USD conversion...")
        success = ply_to_pointcloud_usd(test_ply, test_usd)
        if success:
            logger.info(f"✅ Test successful! USD file: {test_usd}")
        else:
            logger.error("❌ Test failed")
    else:
        logger.warning(f"Test file not found: {test_ply}")
