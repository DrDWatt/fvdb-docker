#!/bin/bash
# Build using the host's Python environment (bind mount approach)
# This creates minimal containers that use the host's fVDB installation

set -e

echo "======================================================================"
echo "Building fVDB Docker Images Using Host Installation"
echo "======================================================================"
echo ""

# Detect fVDB installation
if [ -d "$HOME/miniforge3/envs/fvdb" ]; then
    echo "✅ Found fVDB conda environment"
    PYTHON_ENV="$HOME/miniforge3/envs/fvdb"
    PYTHON_BIN="$PYTHON_ENV/bin/python"
    PYTHON_SITE="$PYTHON_ENV/lib/python3.12/site-packages"
elif python3 -c "import fvdb" 2>/dev/null; then
    echo "✅ Found fVDB in system Python"
    PYTHON_BIN="python3"
    PYTHON_SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")
else
    echo "❌ fVDB not found"
    echo "Please install fVDB first or activate the conda environment"
    exit 1
fi

echo "Python binary: $PYTHON_BIN"
echo "Python packages: $PYTHON_SITE"
echo ""

# Get fVDB version
FVDB_VERSION=$($PYTHON_BIN -c "import fvdb; print(fvdb.__version__)" 2>/dev/null || echo "unknown")
echo "fVDB version: $FVDB_VERSION"
echo ""

# Configuration
REGISTRY="${REGISTRY:-localhost:7000}"
VERSION="${VERSION:-latest}"

# Create bind-mount docker-compose
cat > docker-compose.host.yml <<EOF
version: '3.8'

# Docker Compose using host's Python installation
# This avoids building fVDB in containers

services:
  training:
    build:
      context: ./training-service
      dockerfile: Dockerfile.host
    image: ${REGISTRY}/fvdb-training:${VERSION}
    container_name: fvdb-training
    ports:
      - "8000:8000"
    volumes:
      # Mount host Python packages (read-only)
      - ${PYTHON_SITE}:/host-python:ro
      # Data volumes
      - training-data:/app/data
      - training-uploads:/app/uploads
      - training-outputs:/app/outputs
      - shared-models:/app/models
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
      - PYTHONPATH=/host-python:\${PYTHONPATH}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    restart: unless-stopped

  rendering:
    build:
      context: ./rendering-service
      dockerfile: Dockerfile.host
    image: ${REGISTRY}/fvdb-rendering:${VERSION}
    container_name: fvdb-rendering
    ports:
      - "8001:8001"
    volumes:
      # Mount host Python packages (read-only)
      - ${PYTHON_SITE}:/host-python:ro
      # Data volumes
      - shared-models:/app/models
      - rendering-outputs:/app/outputs
      - rendering-cache:/app/cache
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
      - PYTHONPATH=/host-python:\${PYTHONPATH}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    restart: unless-stopped
    depends_on:
      - training

volumes:
  training-data:
  training-uploads:
  training-outputs:
  shared-models:
  rendering-outputs:
  rendering-cache:

networks:
  default:
    name: fvdb-network
EOF

# Create lightweight Dockerfile for host mount
cat > training-service/Dockerfile.host <<EOF
FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=\$CUDA_HOME/bin:\$PATH
ENV LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH

# Install minimal dependencies
RUN apt-get update && apt-get install -y \\
    python3.12 \\
    python3-pip \\
    libgl1 \\
    libglib2.0-0 \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Python symlinks
RUN ln -sf /usr/bin/python3.12 /usr/bin/python && \\
    ln -sf /usr/bin/python3.12 /usr/bin/python3

# Install only API dependencies (not fVDB - using host mount)
RUN python -m pip install --break-system-packages \\
    fastapi \\
    uvicorn[standard] \\
    python-multipart \\
    aiofiles \\
    pydantic \\
    requests

WORKDIR /app

COPY training_service.py /app/

RUN mkdir -p /app/data /app/uploads /app/outputs /app/models

EXPOSE 8000

CMD ["uvicorn", "training_service:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

# Copy for rendering service
cp training-service/Dockerfile.host rendering-service/Dockerfile.host
sed -i 's/training_service.py/rendering_service.py/g' rendering-service/Dockerfile.host
sed -i 's/8000/8001/g' rendering-service/Dockerfile.host

echo "📦 Building containers..."
docker compose -f docker-compose.host.yml build

echo ""
echo "======================================================================"
echo "✅ Build Complete!"
echo "======================================================================"
echo ""
echo "📋 Setup Summary:"
echo "  - Containers built with minimal dependencies"
echo "  - Host Python packages mounted at: /host-python"
echo "  - fVDB version: $FVDB_VERSION"
echo ""
echo "🚀 To start services:"
echo "   docker compose -f docker-compose.host.yml up -d"
echo ""
echo "📝 Note: Containers use your host's fVDB installation"
echo "   Any updates to host fVDB will be reflected in containers"
echo ""
