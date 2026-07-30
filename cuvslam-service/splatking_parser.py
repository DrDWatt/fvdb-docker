"""
SplatKing ZIP Parser

Parses ZIP files exported from the SplatKing iOS app (Gaussian Splatting capture tool).
Extracts images, quality flags, and camera metadata for use in COLMAP/cuVSLAM pipelines.

SplatKing ZIP structure:
  - capture_started.json  (session metadata)
  - photo_series.json     (per-capture analysis and metadata)
  - quality_flags.csv     (quality score per image: filename, quality_band, quality_score, stream, format)
  - splatpack.json        (full manifest with pairs, camera metadata, app info)
  - ultra_*.jpg / ultra_*.json  (ultra-wide camera images + per-image metadata)
  - wide_*.jpg / wide_*.json    (wide camera images + per-image metadata)
"""

import csv
import json
import logging
import struct
import zipfile
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def is_splatking_zip(zip_ref: zipfile.ZipFile) -> bool:
    """Detect whether a ZIP file is a SplatKing export by checking for signature files."""
    names = [n.split("/")[-1] for n in zip_ref.namelist()]
    # SplatKing ZIPs always contain splatpack.json and quality_flags.csv
    return "splatpack.json" in names and "quality_flags.csv" in names


def parse_quality_flags(zip_ref: zipfile.ZipFile, prefix: str = "") -> Dict[str, dict]:
    """Parse quality_flags.csv from the ZIP.
    
    Returns dict keyed by filename with fields:
      quality_band (str): 'good', 'medium', 'poor'
      quality_score (float): 0.0 - 1.0
      stream (str): 'ultra' or 'wide'
    """
    csv_path = _find_file(zip_ref, "quality_flags.csv", prefix)
    if not csv_path:
        return {}

    content = zip_ref.read(csv_path).decode("utf-8")
    reader = csv.DictReader(content.strip().splitlines())
    
    quality_map = {}
    for row in reader:
        filename = row.get("filename", "").strip().strip('"')
        quality_map[filename] = {
            "quality_band": row.get("quality_band", "").strip().strip('"'),
            "quality_score": float(row.get("quality_score", "0").strip().strip('"')),
            "stream": row.get("stream", "").strip().strip('"'),
        }
    return quality_map


def parse_splatpack(zip_ref: zipfile.ZipFile, prefix: str = "") -> dict:
    """Parse splatpack.json manifest from the ZIP."""
    sp_path = _find_file(zip_ref, "splatpack.json", prefix)
    if not sp_path:
        return {}
    content = zip_ref.read(sp_path).decode("utf-8")
    return json.loads(content)


def parse_image_metadata(zip_ref: zipfile.ZipFile, json_filename: str) -> dict:
    """Parse per-image JSON metadata file from the ZIP.
    
    Extracts camera intrinsics (focal length, sensor info) and quality metrics.
    """
    try:
        content = zip_ref.read(json_filename).decode("utf-8")
        return json.loads(content)
    except (KeyError, json.JSONDecodeError) as e:
        logger.warning(f"Could not parse {json_filename}: {e}")
        return {}


def extract_camera_intrinsics(image_meta: dict, pixel_width: int, pixel_height: int) -> dict:
    """Derive camera intrinsics (fx, fy, cx, cy) from SplatKing per-image metadata.
    
    Uses EXIF FocalLength (mm) and sensor crop factor to compute pixel focal length.
    iPhone ultra-wide: 13mm equiv (2.22mm actual), sensor ~5.7mm wide
    iPhone wide: 24-26mm equiv (6.76mm actual), sensor ~5.7mm wide
    
    Falls back to FocalLenIn35mmFilm for a rough estimate if sensor size unknown.
    """
    # Try nested metadata format (per-image JSON)
    exif = image_meta.get("metadata", {}).get("{Exif}", {})
    if not exif:
        # Fallback: photo_series.json format
        exif = image_meta.get("metadata", {}).get("exif", {})
    
    focal_mm = exif.get("FocalLength")
    focal_35mm = exif.get("FocalLenIn35mmFilm")
    pixel_x = exif.get("PixelXDimension", pixel_width)
    pixel_y = exif.get("PixelYDimension", pixel_height)

    if focal_mm and focal_35mm and focal_35mm > 0:
        # crop_factor = 35mm_equiv / actual_focal_length
        # sensor_width_mm = 36 / crop_factor  (36mm is full-frame width)
        crop_factor = focal_35mm / focal_mm
        sensor_width_mm = 36.0 / crop_factor
        # fx_pixels = focal_mm * pixel_width / sensor_width_mm
        fx = focal_mm * pixel_x / sensor_width_mm
        fy = focal_mm * pixel_y / (sensor_width_mm * pixel_y / pixel_x)
    elif focal_35mm:
        # Rough estimate: fx ≈ focal_35mm_equiv * pixel_width / 36
        fx = focal_35mm * pixel_x / 36.0
        fy = fx
    else:
        # Default: assume ~70% of image width as focal length (common for phones)
        fx = pixel_x * 0.7
        fy = fx
    
    cx = pixel_x / 2.0
    cy = pixel_y / 2.0

    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy}


def extract_splatking_images(
    zip_ref: zipfile.ZipFile,
    output_dir: Path,
    quality_threshold: float = 0.35,
    preferred_stream: str = "ultra",
    workflow_id: str = ""
) -> Tuple[int, int, dict]:
    """Extract images from a SplatKing ZIP into left/right directories for cuVSLAM processing.
    
    Uses the ultra-wide stream as primary (left) and wide stream as secondary (right).
    Filters out images below quality_threshold using quality_flags.csv.
    
    Args:
        zip_ref: Open ZipFile instance
        output_dir: Base output directory (will create images/ and images_right/ subdirs)
        quality_threshold: Minimum quality_score to include an image (0.0-1.0)
        preferred_stream: Which stream to use as primary left images ('ultra' or 'wide')
        workflow_id: For logging
    
    Returns:
        (num_left, num_right, camera_params) tuple
    """
    # Detect subfolder prefix (SplatKing ZIPs have a root folder)
    prefix = _detect_prefix(zip_ref)
    
    # Parse quality flags
    quality_map = parse_quality_flags(zip_ref, prefix)
    logger.info(f"[{workflow_id}] SplatKing: {len(quality_map)} quality entries loaded")
    
    # Parse splatpack for pair ordering
    splatpack = parse_splatpack(zip_ref, prefix)
    
    # Determine stream roles
    if preferred_stream == "wide":
        left_stream = "wide"
        right_stream = "ultra"
    else:
        left_stream = "ultra"
        right_stream = "wide"
    
    # Create output directories
    images_dir = output_dir / "images"
    images_right_dir = output_dir / "images_right"
    images_dir.mkdir(parents=True, exist_ok=True)
    images_right_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metadata for downstream use
    metadata_dir = output_dir / "splatking_metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    # Build ordered list of image pairs from splatpack
    pairs = splatpack.get("pairs", [])
    
    num_left = 0
    num_right = 0
    skipped_quality = 0
    camera_params = {}
    per_image_intrinsics = []
    
    for pair_idx, pair in enumerate(pairs):
        # Check pair-level quality
        pair_quality = pair.get("quality", {})
        pair_score = pair_quality.get("score", 1.0)
        
        if pair_score < quality_threshold:
            skipped_quality += 1
            continue
        
        streams = pair.get("streams", [])
        left_file = None
        right_file = None
        left_meta = None
        
        for stream_info in streams:
            stream_name = stream_info.get("extraMetadata", {}).get("stream", "")
            image_file = stream_info.get("imageFile", "")
            
            # Also check per-image quality from quality_flags
            img_quality = quality_map.get(image_file, {})
            img_score = img_quality.get("quality_score", pair_score)
            
            if img_score < quality_threshold:
                continue
            
            if stream_name == left_stream:
                left_file = image_file
                left_meta = stream_info
            elif stream_name == right_stream:
                right_file = image_file
        
        if not left_file:
            skipped_quality += 1
            continue
        
        # Extract left image (convert HEIC→JPEG if needed)
        left_zip_path = f"{prefix}{left_file}" if prefix else left_file
        try:
            img_data = zip_ref.read(left_zip_path)
            img_path = images_dir / f"frame_{num_left:04d}.jpg"
            _save_image_as_jpeg(img_data, img_path)
        except KeyError:
            logger.warning(f"[{workflow_id}] Left image not found in ZIP: {left_zip_path}")
            continue
        
        # Extract right image (if available)
        if right_file:
            right_zip_path = f"{prefix}{right_file}" if prefix else right_file
            try:
                img_data = zip_ref.read(right_zip_path)
                img_path = images_right_dir / f"frame_{num_left:04d}.jpg"
                _save_image_as_jpeg(img_data, img_path)
                num_right += 1
            except KeyError:
                logger.warning(f"[{workflow_id}] Right image not found: {right_zip_path}")
        
        # Extract camera intrinsics from first valid image metadata
        if not camera_params and left_meta:
            exif = left_meta.get("metadata", {}).get("exif", {})
            pixel_x = exif.get("PixelXDimension", 4224)
            pixel_y = exif.get("PixelYDimension", 2376)
            camera_params = extract_camera_intrinsics(left_meta, pixel_x, pixel_y)
            logger.info(f"[{workflow_id}] SplatKing camera intrinsics: {camera_params}")
        
        # Store per-image intrinsics for COLMAP output
        if left_meta:
            exif = left_meta.get("metadata", {}).get("exif", {})
            pixel_x = exif.get("PixelXDimension", 4224)
            pixel_y = exif.get("PixelYDimension", 2376)
            intrinsics = extract_camera_intrinsics(left_meta, pixel_x, pixel_y)
            per_image_intrinsics.append({
                "frame": f"frame_{num_left:04d}.jpg",
                "intrinsics": intrinsics,
                "quality_score": pair_score,
            })
        
        num_left += 1
    
    # Save metadata files for reference
    if per_image_intrinsics:
        with open(metadata_dir / "per_image_intrinsics.json", "w") as f:
            json.dump(per_image_intrinsics, f, indent=2)
    
    if quality_map:
        with open(metadata_dir / "quality_flags.json", "w") as f:
            json.dump(quality_map, f, indent=2)
    
    # Save splatpack summary
    summary = {
        "source": "splatking",
        "schema_version": splatpack.get("schemaVersion"),
        "capture_type": splatpack.get("captureType"),
        "app_version": splatpack.get("app", {}).get("shortVersion"),
        "total_pairs": splatpack.get("pairCount", 0),
        "extracted_left": num_left,
        "extracted_right": num_right,
        "skipped_quality": skipped_quality,
        "quality_threshold": quality_threshold,
        "preferred_stream": preferred_stream,
        "camera_params": camera_params,
    }
    with open(metadata_dir / "extraction_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(
        f"[{workflow_id}] SplatKing extraction: {num_left} left, {num_right} right "
        f"(skipped {skipped_quality} below quality {quality_threshold})"
    )
    
    return num_left, num_right, camera_params


def _save_image_as_jpeg(image_data: bytes, output_path: Path) -> bool:
    """Save image data to JPEG, converting from HEIC/HEIF if needed.
    Falls back to raw write for standard formats (JPEG, PNG)."""
    if len(image_data) >= 12:
        ftype = image_data[8:12]
        if image_data[4:8] == b'ftyp' and ftype in (b'heic', b'heix', b'hevc', b'hevx', b'heim', b'heis', b'mif1'):
            try:
                import pillow_heif
                pillow_heif.register_heif_opener()
                from PIL import Image
                import io
                img = Image.open(io.BytesIO(image_data))
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(str(output_path), 'JPEG', quality=95)
                return True
            except ImportError:
                logger.error(f"HEIC image detected but pillow-heif not installed: {output_path}")
            except Exception as e:
                logger.error(f"HEIC conversion failed for {output_path}: {e}")
    # Standard format — write directly
    with open(output_path, "wb") as f:
        f.write(image_data)
    return True


def _detect_prefix(zip_ref: zipfile.ZipFile) -> str:
    """Detect the root folder prefix in a SplatKing ZIP."""
    for name in zip_ref.namelist():
        parts = name.split("/")
        if len(parts) > 1 and parts[-1] == "splatpack.json":
            return "/".join(parts[:-1]) + "/"
    # Check if files are directly at root
    if "splatpack.json" in zip_ref.namelist():
        return ""
    # Try first directory entry
    for name in zip_ref.namelist():
        if "/" in name:
            return name.split("/")[0] + "/"
    return ""


def _find_file(zip_ref: zipfile.ZipFile, filename: str, prefix: str = "") -> Optional[str]:
    """Find a file in the ZIP, trying with and without prefix."""
    candidates = [
        f"{prefix}{filename}",
        filename,
    ]
    for candidate in candidates:
        if candidate in zip_ref.namelist():
            return candidate
    # Search all entries
    for name in zip_ref.namelist():
        if name.endswith(f"/{filename}") or name == filename:
            return name
    return None


# ===========================================================================
# Video-based SplatKing ZIP support (streaming ZIPs from video capture mode)
#
# SplatKing video exports use data descriptors (bit 3 flag) and lack a
# central directory, which means Python's standard zipfile module cannot
# read them. Structure:
#   - splatpack.json  (manifest: camera intrinsics, streams, schema)
#   - ultra.mov       (ultra-wide camera video)
#   - wide.mov        (wide camera video)
# ===========================================================================

# Local file header signature
_LFH_SIG = b'PK\x03\x04'
# Data descriptor signature (optional)
_DD_SIG = b'PK\x07\x08'


def extract_streaming_zip(zip_path: Path, extract_dir: Path) -> Dict[str, Path]:
    """
    Extract files from a streaming ZIP (no central directory).

    Returns a dictionary mapping filename to extracted file path.
    """
    extract_dir.mkdir(parents=True, exist_ok=True)
    extracted = {}

    with open(zip_path, 'rb') as f:
        while True:
            # Read local file header (minimum 30 bytes)
            header = f.read(30)
            if len(header) < 30:
                break

            # Check for local file header signature
            if header[:4] != _LFH_SIG:
                # Try to find next header (may have padding)
                f.seek(-30, 1)
                chunk = f.read(4096)
                idx = chunk.find(_LFH_SIG)
                if idx == -1:
                    break
                f.seek(-(len(chunk) - idx), 1)
                header = f.read(30)
                if header[:4] != _LFH_SIG:
                    break

            # Parse local file header fields
            flags = struct.unpack('<H', header[6:8])[0]
            comp_method = struct.unpack('<H', header[8:10])[0]
            comp_size = struct.unpack('<I', header[18:22])[0]
            fname_len = struct.unpack('<H', header[26:28])[0]
            extra_len = struct.unpack('<H', header[28:30])[0]

            # Read filename and extra field
            fname_bytes = f.read(fname_len)
            fname = fname_bytes.decode('utf-8', errors='replace')
            f.read(extra_len)  # skip extra field

            has_data_descriptor = bool(flags & 0x08)

            if has_data_descriptor and comp_size == 0:
                # Streaming entry: scan for next PK header to find data boundary
                file_data = _read_until_next_entry(f)
            elif comp_size > 0:
                file_data = f.read(comp_size)
            else:
                file_data = b''

            # Skip data descriptor if present
            if has_data_descriptor:
                dd_peek = f.read(4)
                if dd_peek == _DD_SIG:
                    f.read(12)  # crc32 + comp_size + uncomp_size
                elif len(dd_peek) == 4:
                    # No signature, just crc32 followed by sizes
                    f.read(8)  # comp_size + uncomp_size

            # Only store (skip compression - SplatKing uses method 0 = stored)
            if comp_method != 0:
                logger.warning(f"Skipping compressed entry (method={comp_method}): {fname}")
                continue

            # Strip leading directory from path for cleaner extraction
            basename = Path(fname).name
            if not basename:
                # Directory entry
                continue

            out_path = extract_dir / basename
            with open(out_path, 'wb') as out_f:
                out_f.write(file_data)

            extracted[basename] = out_path
            logger.info(f"Extracted: {basename} ({len(file_data)} bytes)")

    return extracted


def _read_until_next_entry(f: BinaryIO) -> bytes:
    """
    Read file data until the next PK local/central header or data descriptor.
    Uses buffered reads for efficiency with large video files.
    """
    chunks = []
    buf_size = 1024 * 1024  # 1MB read buffer

    while True:
        chunk = f.read(buf_size)
        if not chunk:
            break

        # Search for next PK header in this chunk
        search_data = chunk
        pk_pos = -1
        search_start = 0

        while True:
            idx = search_data.find(b'PK', search_start)
            if idx == -1:
                break
            # Verify it's a real header
            if idx + 4 <= len(search_data):
                sig_type = search_data[idx+2:idx+4]
                if sig_type in (b'\x03\x04', b'\x01\x02'):
                    pk_pos = idx
                    break
                elif sig_type == b'\x07\x08':
                    # Data descriptor - the data ends here
                    pk_pos = idx
                    break
            search_start = idx + 1

        if pk_pos != -1:
            # Found boundary
            if search_data[pk_pos+2:pk_pos+4] == b'\x07\x08':
                # Data descriptor: include nothing after it
                chunks.append(chunk[:pk_pos])
                # Seek back to right after the data descriptor (16 bytes)
                remaining = len(chunk) - pk_pos - 16
                if remaining > 0:
                    f.seek(-remaining, 1)
                elif remaining < 0:
                    # Data descriptor spans into next read
                    f.read(-remaining)
            else:
                # Next local/central header
                chunks.append(chunk[:pk_pos])
                remaining = len(chunk) - pk_pos
                f.seek(-remaining, 1)
            break
        else:
            # No header found - keep all but last 4 bytes (boundary safety)
            if len(chunk) == buf_size:
                chunks.append(chunk[:-4])
                f.seek(-4, 1)
            else:
                chunks.append(chunk)
                break

    return b''.join(chunks)


def parse_video_splatpack(extracted_files: Dict[str, Path]) -> Optional[Dict]:
    """
    Parse splatpack.json from an extracted video-capture ZIP.

    Returns the parsed splatpack dict, or None if not a SplatKing zip.
    """
    splatpack_path = extracted_files.get('splatpack.json')
    if not splatpack_path or not splatpack_path.exists():
        return None

    try:
        with open(splatpack_path) as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to parse splatpack.json: {e}")
        return None


def get_video_camera_intrinsics(splatpack: Dict) -> List[Dict]:
    """
    Extract camera intrinsics from splatpack.json video streams.

    Returns list of dicts with keys:
      - camera: str (e.g. "wide", "ultra")
      - video_file: str (e.g. "wide.mov")
      - width / height: int
      - field_of_view: float (degrees)
      - frame_count: int
    """
    intrinsics = []
    for stream in splatpack.get('streams', []):
        device_info = stream.get('deviceInfo', {})
        active_format = device_info.get('activeFormat', {})

        intrinsics.append({
            'camera': stream.get('camera', 'unknown'),
            'video_file': stream.get('rawVideoFile', ''),
            'width': active_format.get('width', 0),
            'height': active_format.get('height', 0),
            'field_of_view': active_format.get('fieldOfView', 0),
            'frame_count': stream.get('frameCount', 0),
        })

    return intrinsics


def find_video_files(extracted_files: Dict[str, Path]) -> List[Path]:
    """Find all video files (.mov, .mp4, .avi, .mkv) in extracted files."""
    video_exts = {'.mov', '.mp4', '.avi', '.mkv'}
    return [
        path for name, path in extracted_files.items()
        if Path(name).suffix.lower() in video_exts
    ]
