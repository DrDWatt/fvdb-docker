# cuSFM Service - GPU-accelerated Structure from Motion with TensorRT
# Uses NVIDIA cuSFM (CUDA 13) for Blackwell GPUs
FROM nvcr.io/nvidia/tensorrt:25.09-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies (from official cuSFM Dockerfile)
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    git \
    git-lfs \
    cmake \
    build-essential \
    libabsl-dev \
    libprotobuf-dev \
    protobuf-compiler \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for FastAPI service wrapper
RUN pip3 install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    python-multipart \
    aiofiles \
    pydantic \
    pillow \
    opencv-python-headless

WORKDIR /app

# Clone pyCuSFM repository with LFS support
RUN git lfs install && \
    git clone https://github.com/nvidia-isaac/PyCuSFM /app/pycusfm

# Setup cuSFM for CUDA 13 (Blackwell GPU support)
WORKDIR /app/pycusfm
RUN ./setup.bash cuda13

# Install pyCuSFM in development mode
RUN pip3 install --editable .

# Add cusfm_cli to PATH
ENV PATH="/root/.local/bin:/app/pycusfm:$PATH"
ENV PYCUSFM_DIR="/app/pycusfm/pycusfm"

WORKDIR /app

# Copy the FastAPI service wrapper
COPY cusfm_service.py /app/

# Create working directories
RUN mkdir -p /app/uploads /app/processing /app/outputs /app/temp

EXPOSE 8014

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8014/health || exit 1

CMD ["uvicorn", "cusfm_service:app", "--host", "0.0.0.0", "--port", "8014"]
