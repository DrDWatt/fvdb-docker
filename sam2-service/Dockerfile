# SAM-2 Segmentation Service
# GPU-accelerated object segmentation using Segment Anything Model 2
FROM nvcr.io/nvidia/pytorch:25.04-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Install SAM-2 from Meta's repository
# Use --no-build-isolation to use existing PyTorch from NGC container (avoids version conflict)
RUN pip3 install --no-build-isolation git+https://github.com/facebookresearch/segment-anything-2.git

# Create directories
RUN mkdir -p /app/uploads /app/outputs /app/models /app/cache

# Copy service code
COPY sam2_service.py .
COPY static/ ./static/

# Download SAM-2 checkpoints on first run (handled in service)
ENV SAM2_CHECKPOINT_DIR=/app/models/sam2
ENV TORCH_HOME=/app/cache

EXPOSE 8004

HEALTHCHECK --interval=10s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8004/health || exit 1

CMD ["python3", "-m", "uvicorn", "sam2_service:app", "--host", "0.0.0.0", "--port", "8004"]
