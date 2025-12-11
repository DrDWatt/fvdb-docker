"""
Standalone COLMAP processing utilities
"""

import subprocess
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def run_colmap_automatic(workspace_path: Path, quality: str = "medium") -> bool:
    # Set Qt to offscreen mode for headless execution
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    """
    Run COLMAP automatic reconstructor
    
    Args:
        workspace_path: Path containing images/ subfolder
        quality: low, medium, high, extreme
    
    Returns:
        True if successful
    """
    try:
        quality_map = {
            "low": "low",
            "medium": "medium",
            "high": "high",
            "extreme": "extreme"
        }
        
        cmd = [
            "colmap", "automatic_reconstructor",
            "--workspace_path", str(workspace_path),
            "--image_path", str(workspace_path / "images"),
            "--quality", quality_map.get(quality, "medium"),
            "--sparse", "1",
            "--dense", "0"  # Skip dense reconstruction
        ]
        
        # Set environment for headless Qt
        env = os.environ.copy()
        env['QT_QPA_PLATFORM'] = 'offscreen'
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,
            env=env
        )
        
        return result.returncode == 0
        
    except Exception as e:
        logger.error(f"COLMAP automatic reconstruction failed: {e}")
        return False

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python process_colmap.py <workspace_path> [quality]")
        sys.exit(1)
    
    workspace = Path(sys.argv[1])
    quality = sys.argv[2] if len(sys.argv) > 2 else "medium"
    
    success = run_colmap_automatic(workspace, quality)
    sys.exit(0 if success else 1)
