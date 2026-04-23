# COLMAP Processing Service for ARM64
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV QT_QPA_PLATFORM=offscreen

# Install COLMAP from Ubuntu repositories (easier than building from source)
# This installs COLMAP 3.8 which is stable and works well
# Also install xvfb for virtual display
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3-pip \
    python3-dev \
    colmap \
    ffmpeg \
    curl \
    wget \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN python3 -m pip install --break-system-packages \
    fastapi \
    uvicorn[standard] \
    python-multipart \
    aiofiles \
    pydantic \
    pillow \
    opencv-python-headless \
    httpx

WORKDIR /app

# Copy service code
COPY colmap_service.py /app/
COPY process_colmap.py /app/

# Create directories
RUN mkdir -p /app/uploads /app/processing /app/outputs /app/temp

EXPOSE 8003

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8003/health || exit 1

# Start xvfb and then uvicorn
CMD Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 & export DISPLAY=:99 && uvicorn colmap_service:app --host 0.0.0.0 --port 8003
