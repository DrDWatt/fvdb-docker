"""
SplatKing ZIP Parser

Parses streaming ZIP files created by the Gaussian Splat King iOS app.
These ZIPs use data descriptors (bit 3 flag) and lack a central directory,
which means Python's standard zipfile module cannot read them.

SplatKing ZIP structure:
  - capture_events.ndjson   (recording events)
  - capture_started.json    (capture metadata)
  - frame_timecodes.csv     (per-frame timestamps for each stream)
  - metadata.json           (frame time arrays)
  - splatpack.json          (manifest: camera intrinsics, streams, schema)
  - ultra.mov               (ultra-wide camera video)
  - wide.mov                (wide camera video)
"""

import struct
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, BinaryIO

logger = logging.getLogger(__name__)

# Local file header signature
_LFH_SIG = b'PK\x03\x04'
# Data descriptor signature (optional)
_DD_SIG = b'PK\x07\x08'
# Central directory header signature
_CDH_SIG = b'PK\x01\x02'


def _find_next_header(data: bytes, start: int) -> int:
    """Find the next PK signature after start position."""
    pos = start
    while pos < len(data) - 4:
        if data[pos:pos+2] == b'PK' and data[pos+2:pos+4] in (b'\x03\x04', b'\x01\x02', b'\x07\x08'):
            return pos
        pos += 1
    return len(data)


def extract_streaming_zip(zip_path: Path, extract_dir: Path) -> Dict[str, Path]:
    """
    Extract files from a streaming ZIP (no central directory).

    Args:
        zip_path: Path to the streaming ZIP file
        extract_dir: Directory to extract files into

    Returns:
        Dictionary mapping filename to extracted file path
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
            uncomp_size = struct.unpack('<I', header[22:26])[0]
            fname_len = struct.unpack('<H', header[26:28])[0]
            extra_len = struct.unpack('<H', header[28:30])[0]

            # Read filename and extra field
            fname_bytes = f.read(fname_len)
            fname = fname_bytes.decode('utf-8', errors='replace')
            f.read(extra_len)  # skip extra field

            has_data_descriptor = bool(flags & 0x08)

            if has_data_descriptor and comp_size == 0:
                # Streaming entry: scan for next PK header to find data boundary
                data_start = f.tell()
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
        # Check overlapping boundary from previous chunk
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


def parse_splatpack(extracted_files: Dict[str, Path]) -> Optional[Dict]:
    """
    Parse splatpack.json to get SplatKing capture metadata.

    Returns:
        Parsed splatpack dict, or None if not a SplatKing zip.
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


def get_camera_intrinsics(splatpack: Dict) -> List[Dict]:
    """
    Extract camera intrinsics from splatpack.json streams.

    Returns list of dicts with keys:
      - camera: str (e.g. "wide", "ultra")
      - video_file: str (e.g. "wide.mov")
      - width: int
      - height: int
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
    """Find all video files (.mov, .mp4, .avi) in extracted files."""
    video_exts = {'.mov', '.mp4', '.avi', '.mkv'}
    return [
        path for name, path in extracted_files.items()
        if Path(name).suffix.lower() in video_exts
    ]
