"""H.264 MP4 writer shared by both viewers' MP4 export."""
import numpy as np
import pytest

from support.media import assert_universal_mp4, jpeg_bytes, probe_mp4
from viewer_common.video import H264Writer


def test_rgb_frames_encode_to_universal_mp4(tmp_path):
    out = tmp_path / "rgb.mp4"
    writer = H264Writer(out, 320, 240, 30)
    for i in range(30 * 30):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        frame[:, (i * 3) % 320] = 255
        writer.write(frame.tobytes())
    writer.close()
    assert_universal_mp4(probe_mp4(out), min_seconds=30, fps=30, size=(320, 240))


def test_jpeg_frames_encode_and_are_scaled_to_requested_size(tmp_path):
    out = tmp_path / "jpeg.mp4"
    writer = H264Writer(out, 256, 144, 24, input_format="jpeg")
    for i in range(24 * 30):
        writer.write(jpeg_bytes(320, 180, shade=i))    # source frames larger than output
    writer.close()
    assert_universal_mp4(probe_mp4(out), min_seconds=30, fps=24, size=(256, 144))


def test_odd_dimensions_are_rounded_to_even_for_yuv420p(tmp_path):
    writer = H264Writer(tmp_path / "odd.mp4", 321, 181, 30)
    assert (writer.width, writer.height) == (320, 180)
    writer.abort()


def test_corrupt_input_raises_with_ffmpeg_message(tmp_path):
    writer = H264Writer(tmp_path / "bad.mp4", 64, 64, 30, input_format="jpeg")
    writer.write(b"not a jpeg")
    with pytest.raises(RuntimeError, match="ffmpeg encode failed"):
        writer.close()


def test_unknown_input_format_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        H264Writer(tmp_path / "x.mp4", 64, 64, 30, input_format="png")
