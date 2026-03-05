# cuVSLAM Service - GPU-accelerated Visual SLAM for aarch64 (Blackwell/Jetson)
# Uses PyCuVSLAM with CUDA 12.6 runtime
FROM nvcr.io/nvidia/cuda:12.6.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies + Python 3.10 (default on Ubuntu 22.04)
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    libcusolver-12-6 \
    libgl1 \
    libglib2.0-0 \
    curl \
    ffmpeg \
    colmap \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy PyCuVSLAM aarch64 binaries and install
COPY pycuvslam/bin/aarch64 /app/pycuvslam-aarch64
RUN pip3 install --no-cache-dir -e /app/pycuvslam-aarch64

# Install FastAPI service dependencies
RUN pip3 install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    python-multipart \
    aiofiles \
    pydantic \
    pillow \
    opencv-python-headless \
    numpy \
    scipy \
    httpx \
    sharp-frames

# Copy the FastAPI service wrapper
COPY cuvslam_service.py /app/

# Create working directories
RUN mkdir -p /app/uploads /app/processing /app/outputs /app/temp

EXPOSE 8003

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8003/health || exit 1

CMD ["uvicorn", "cuvslam_service:app", "--host", "0.0.0.0", "--port", "8003"]
