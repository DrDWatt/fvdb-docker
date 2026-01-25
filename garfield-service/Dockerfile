# GARField Service - 3D Asset Extraction
# Use NGC PyTorch image for proper ARM64 CUDA support
FROM nvcr.io/nvidia/pytorch:25.04-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install additional system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

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
