"""Universally playable MP4 writer (H.264 High, yuv420p, CFR, faststart).

Output plays smoothly in QuickTime, Windows Media Player, browsers and phones.
Frames are piped to the static ffmpeg bundled with imageio-ffmpeg because the
OpenCV FFmpeg builds in these containers lack libx264.
"""
import logging
import subprocess

logger = logging.getLogger(__name__)


class H264Writer:
    """Stream frames into an H.264 MP4.

    input_format:
        "rgb24" - write() takes raw RGB bytes of exactly width*height*3
        "jpeg"  - write() takes one encoded JPEG image per call
    """

    def __init__(self, output_path, width, height, fps, input_format="rgb24", crf=18):
        import imageio_ffmpeg

        # yuv420p requires even dimensions
        self.width, self.height = int(width) - int(width) % 2, int(height) - int(height) % 2
        self.fps = int(fps)
        self.output_path = str(output_path)
        self.frames = 0
        if input_format == "rgb24":
            source = ['-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{self.width}x{self.height}']
        elif input_format == "jpeg":
            source = ['-f', 'image2pipe', '-c:v', 'mjpeg']
        else:
            raise ValueError(f"unsupported input_format: {input_format}")
        cmd = [
            imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error',
            *source, '-framerate', str(self.fps), '-i', '-',
            # scale guards against odd/mismatched input sizes (JPEG path)
            '-vf', f'scale={self.width}:{self.height}:flags=lanczos',
            '-c:v', 'libx264', '-preset', 'medium', '-crf', str(crf),
            '-profile:v', 'high', '-level', '4.1', '-pix_fmt', 'yuv420p',
            '-r', str(self.fps), '-fps_mode', 'cfr', '-g', str(self.fps * 2),
            '-movflags', '+faststart', self.output_path,
        ]
        logger.info(f"H.264 encode start: {self.width}x{self.height}@{self.fps}fps -> {self.output_path}")
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, data: bytes):
        self._proc.stdin.write(data)
        self.frames += 1

    def close(self):
        """Finish encoding. Raises RuntimeError with ffmpeg's message on failure."""
        if self._proc.stdin and not self._proc.stdin.closed:
            self._proc.stdin.close()
        stderr = self._proc.stderr.read().decode(errors='replace')
        self._proc.wait()
        if self._proc.returncode != 0:
            raise RuntimeError(f"ffmpeg encode failed: {stderr[-500:]}")
        logger.info(f"H.264 encode done: {self.frames} frames -> {self.output_path}")

    def abort(self):
        """Stop encoding without producing a valid file."""
        try:
            self._proc.kill()
            self._proc.wait(timeout=5)
        except Exception:
            pass
