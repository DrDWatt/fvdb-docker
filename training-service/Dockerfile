# Multi-architecture Dockerfile for fVDB Reality Capture Training Service
# ARM64 with CUDA support using NVIDIA PyTorch container with Python 3.11+

FROM nvcr.io/nvidia/pytorch:24.11-py3

# Build arguments for multi-arch
ARG TARGETPLATFORM
ARG BUILDPLATFORM

# Set environment
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install additional system dependencies including build tools for point-cloud-utils
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    ninja-build \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    unzip \
    libeigen3-dev \
    libboost-all-dev \
    libtbb-dev \
    python3-dev \
    libpython3-dev \
    pybind11-dev \
    && rm -rf /var/lib/apt/lists/*

# PyTorch is already installed in the base image with CUDA support
# Note: Skip verification during build due to library conflicts - will verify at runtime
# RUN python3 -c "import torch; print(f'✅ PyTorch {torch.__version__}')"

# fVDB will be mounted from host at /host-fvdb
# No need to install in container - use host Python with fVDB

# Install FastAPI, Uvicorn, and other web dependencies
RUN python -m pip install --break-system-packages \
    fastapi \
    uvicorn[standard] \
    python-multipart \
    aiofiles \
    pydantic \
    requests \
    pillow \
    opencv-python-headless \
    matplotlib \
    PyMuPDF==1.24.0

# Create app directory
WORKDIR /app

# Copy service code
COPY training_service.py /app/
COPY requirements.txt /app/

# Create data directories
RUN mkdir -p /app/data /app/uploads /app/outputs /app/models

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the service - container Python with host fVDB via PYTHONPATH
CMD ["python3", "-m", "uvicorn", "training_service:app", "--host", "0.0.0.0", "--port", "8000"]
