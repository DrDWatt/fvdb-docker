# GARField Service - 3D Asset Extraction
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    git \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA 12.4 support
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install segment-anything for mask generation
RUN pip install --no-cache-dir git+https://github.com/facebookresearch/segment-anything.git

# Copy service code
COPY garfield_service.py .
COPY static/ ./static/

EXPOSE 8006

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8006/health || exit 1

CMD ["python", "-m", "uvicorn", "garfield_service:app", "--host", "0.0.0.0", "--port", "8006"]
