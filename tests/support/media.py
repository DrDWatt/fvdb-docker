"""Image / video helpers: synthetic frames and MP4 inspection via the bundled ffmpeg."""
import io
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


def jpeg_bytes(width=320, height=180, shade=0, quality=90) -> bytes:
    """A JPEG with a moving square so consecutive frames differ."""
    img = np.full((height, width, 3), 30, dtype=np.uint8)
    x = int(shade) % max(1, width - 40)
    img[height // 3: height // 3 + 40, x: x + 40] = (230, 60, 40)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def image_size(data: bytes):
    return Image.open(io.BytesIO(data)).size


def probe_mp4(data_or_path) -> dict:
    """Duration, codec, profile, pixel format, size and fps of an MP4, plus a full
    decode check (ffmpeg exits non-zero on corrupt streams)."""
    import imageio_ffmpeg

    if isinstance(data_or_path, (str, Path)):
        path = Path(data_or_path)
    else:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(data_or_path)
        path = Path(tmp.name)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    info = subprocess.run([ffmpeg, "-hide_banner", "-i", str(path)], capture_output=True, text=True).stderr
    # full decode: errors go to stderr, decoded frame count to the -progress stream
    decode = subprocess.run([ffmpeg, "-v", "error", "-nostats", "-progress", "pipe:1",
                             "-i", str(path), "-f", "null", "-"], capture_output=True, text=True)
    dur = re.search(r"Duration: (\d+):(\d+):([\d.]+)", info)
    video = re.search(r"Video: (\w+) \((\w+)\)[^,]*, (\w+)[^,]*, (\d+)x(\d+).*?, ([\d.]+) fps", info)
    nframes = re.findall(r"^frame=(\d+)", decode.stdout, re.M)
    raw = path.read_bytes()
    return {
        "duration": int(dur[1]) * 3600 + int(dur[2]) * 60 + float(dur[3]) if dur else 0.0,
        "codec": video[1] if video else None,
        "profile": video[2] if video else None,
        "pix_fmt": video[3] if video else None,
        "width": int(video[4]) if video else 0,
        "height": int(video[5]) if video else 0,
        "fps": float(video[6]) if video else 0.0,
        "frames": int(nframes[-1]) if nframes else 0,
        "decodes_cleanly": decode.returncode == 0 and not decode.stderr.strip(),
        # moov before mdat => "faststart" (streams/plays before fully downloaded)
        "faststart": 0 <= raw.find(b"moov") < raw.find(b"mdat"),
        "raw_info": info,
    }


def assert_universal_mp4(info: dict, min_seconds: float, fps: int, size=None):
    """The export contract: H.264 High, yuv420p, CFR, >= min length, clean decode."""
    assert info["codec"] == "h264", info["raw_info"]
    assert info["profile"] == "High", info["raw_info"]
    assert info["pix_fmt"] == "yuv420p", info["raw_info"]
    assert info["duration"] >= min_seconds - 0.05, info["raw_info"]
    assert abs(info["fps"] - fps) < 0.01, info["raw_info"]
    assert info["frames"] >= round(min_seconds * fps), info
    assert info["decodes_cleanly"]
    assert info["faststart"]
    if size:
        assert (info["width"], info["height"]) == tuple(size)
