# Custom WebRTC Streaming Server for Omniverse Web Viewer
# Streams 3D Gaussian Splat models via WebRTC

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for video/graphics
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libopencv-dev \
    python3-opencv \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \
    aiortc==1.9.0 \
    aiohttp==3.10.0 \
    websockets==13.0 \
    numpy==1.26.0 \
    opencv-python-headless==4.9.0.80 \
    pillow==10.4.0 \
    plyfile==1.0.3

WORKDIR /app

# Copy server code
COPY streaming_server.py /app/
COPY test-viewer.html /app/

# Create directories
RUN mkdir -p /app/models

# Expose ports
EXPOSE 49100 8090

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8090/health || exit 1

CMD ["python", "streaming_server.py"]
