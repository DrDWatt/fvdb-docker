# SpectacularAI Gaussian Splatting Service
# Provides SLAM-based 3DGS training using SpectacularAI SDK
FROM nvcr.io/nvidia/cuda:12.2.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    git \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Install SpectacularAI SDK with full dependencies
RUN pip3 install --no-cache-dir "spectacularAI[full]"

# Install Nerfstudio for Gaussian Splatting training
RUN pip3 install --no-cache-dir nerfstudio gsplat

# Copy application code
COPY . .

# Create directories for data
RUN mkdir -p /app/uploads /app/processing /app/outputs /app/models

# Expose API port
EXPOSE 8004

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8004/health || exit 1

# Run the service
CMD ["python3", "-m", "uvicorn", "spectacularai_service:app", "--host", "0.0.0.0", "--port", "8004"]
