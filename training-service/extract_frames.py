#!/usr/bin/env python3
"""
Video Frame Extraction Utility for fVDB Training
Extracts frames from video files for 3D Gaussian Splatting reconstruction
"""

import subprocess
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_frames_from_video(
    video_path: str,
    output_dir: str,
    fps: float = 2.0,
    quality: int = 2,
    max_frames: int = None,
    start_time: float = None,
    end_time: float = None
) -> dict:
    """
    Extract frames from a video file using ffmpeg
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        fps: Frames per second to extract (default: 2 = one frame every 0.5 seconds)
        quality: JPEG quality (1-31, lower is better, default: 2)
        max_frames: Maximum number of frames to extract (None = all)
        start_time: Start extraction at this time in seconds (None = from beginning)
        end_time: End extraction at this time in seconds (None = until end)
    
    Returns:
        dict with extraction results
    """
    
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build ffmpeg command
    cmd = ["ffmpeg"]
    
    # Input file
    if start_time is not None:
        cmd.extend(["-ss", str(start_time)])
    
    cmd.extend(["-i", str(video_path)])
    
    # End time
    if end_time is not None:
        duration = end_time - (start_time or 0)
        cmd.extend(["-t", str(duration)])
    
    # Frame rate
    cmd.extend(["-vf", f"fps={fps}"])
    
    # Quality
    cmd.extend(["-q:v", str(quality)])
    
    # Output pattern
    output_pattern = output_dir / "frame_%05d.jpg"
    cmd.append(str(output_pattern))
    
    # Limit frames if specified
    if max_frames:
        cmd.extend(["-frames:v", str(max_frames)])
    
    logger.info(f"Extracting frames from {video_path.name}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    # Run ffmpeg
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Count extracted frames
        frames = list(output_dir.glob("frame_*.jpg"))
        num_frames = len(frames)
        
        logger.info(f"Successfully extracted {num_frames} frames")
        
        return {
            "success": True,
            "num_frames": num_frames,
            "output_dir": str(output_dir),
            "frames": sorted([f.name for f in frames])
        }
        
    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg failed: {e.stderr}")
        return {
            "success": False,
            "error": e.stderr
        }


def get_video_info(video_path: str) -> dict:
    """
    Get information about a video file using ffprobe
    
    Args:
        video_path: Path to video file
    
    Returns:
        dict with video information
    """
    
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
        "-of", "default=noprint_wrappers=1",
        str(video_path)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse output
        info = {}
        for line in result.stdout.strip().split('\n'):
            if '=' in line:
                key, value = line.split('=', 1)
                info[key] = value
        
        return {
            "success": True,
            "duration": float(info.get("duration", 0)),
            "width": int(info.get("width", 0)),
            "height": int(info.get("height", 0)),
            "frame_rate": info.get("r_frame_rate", "unknown"),
            "total_frames": int(info.get("nb_frames", 0)) if info.get("nb_frames") else None
        }
        
    except subprocess.CalledProcessError as e:
        logger.error(f"ffprobe failed: {e.stderr}")
        return {
            "success": False,
            "error": e.stderr
        }


def recommend_extraction_params(video_duration: float, target_frames: int = 50) -> dict:
    """
    Recommend extraction parameters based on video duration
    
    Args:
        video_duration: Video duration in seconds
        target_frames: Target number of frames to extract
    
    Returns:
        dict with recommended parameters
    """
    
    # Calculate fps to get target frames
    fps = target_frames / video_duration if video_duration > 0 else 2.0
    
    # Clamp to reasonable values
    fps = max(0.5, min(fps, 5.0))  # Between 0.5 and 5 fps
    
    estimated_frames = int(video_duration * fps)
    
    return {
        "recommended_fps": round(fps, 2),
        "estimated_frames": estimated_frames,
        "video_duration": video_duration,
        "note": f"Extracting at {fps:.2f} fps will yield ~{estimated_frames} frames"
    }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python extract_frames.py <video_file> <output_dir> [fps] [max_frames]")
        print("\nExample:")
        print("  python extract_frames.py video.mp4 ./frames 2.0")
        print("  python extract_frames.py video.mp4 ./frames 1.5 60")
        sys.exit(1)
    
    video_file = sys.argv[1]
    output_dir = sys.argv[2]
    fps = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
    max_frames = int(sys.argv[4]) if len(sys.argv) > 4 else None
    
    # Get video info
    print("Analyzing video...")
    info = get_video_info(video_file)
    if info["success"]:
        print(f"Duration: {info['duration']:.2f}s")
        print(f"Resolution: {info['width']}x{info['height']}")
        print(f"Frame rate: {info['frame_rate']}")
        
        # Get recommendations
        rec = recommend_extraction_params(info['duration'])
        print(f"\nRecommendation: {rec['note']}")
    
    # Extract frames
    print(f"\nExtracting frames at {fps} fps...")
    result = extract_frames_from_video(
        video_file,
        output_dir,
        fps=fps,
        max_frames=max_frames
    )
    
    if result["success"]:
        print(f"\n✓ Extracted {result['num_frames']} frames to {result['output_dir']}")
    else:
        print(f"\n✗ Extraction failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)
