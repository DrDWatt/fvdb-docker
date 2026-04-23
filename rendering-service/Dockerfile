# Multi-architecture Dockerfile for fVDB Rendering Service
# Supports: linux/arm64, linux/amd64

FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu24.04

ARG TARGETPLATFORM
ARG BUILDPLATFORM

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python symlinks
RUN ln -sf /usr/bin/python3.12 /usr/bin/python && \
    ln -sf /usr/bin/python3.12 /usr/bin/python3

# Install Python packages
RUN python -m pip install --break-system-packages torch torchvision --index-url https://download.pytorch.org/whl/cu126

RUN python -m pip install --break-system-packages fvdb fvdb-reality-capture

RUN python -m pip install --break-system-packages \
    fastapi \
    uvicorn[standard] \
    python-multipart \
    jinja2 \
    aiofiles \
    pydantic \
    pillow \
    numpy \
    PyMuPDF==1.24.0

WORKDIR /app

# Copy service code
COPY rendering_service.py /app/
COPY templates/ /app/templates/
COPY static/ /app/static/

# Create directories
RUN mkdir -p /app/models /app/outputs /app/cache

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

CMD ["uvicorn", "rendering_service:app", "--host", "0.0.0.0", "--port", "8001"]
