###############################################################################
# SuperSplat Viewer – GPU-accelerated Gaussian Splatting viewer with SAM3
#
# Architecture:
#   - Frontend: SuperSplat viewer (WebGPU/WebGL, self-hosted static build)
#   - Backend:  FastAPI (SAM3 segmentation, GARField extraction, Trellis proxy)
#
# GPU: Requires NVIDIA GB10 (Blackwell sm_121) — uses PyTorch nightly + CUDA 13.0
# Model downloads happen at BUILD time so containers start instantly.
###############################################################################
FROM nvidia/cuda:12.6.3-devel-ubuntu24.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies (includes build tools for pycocotools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3-pip python3.12-dev \
    build-essential \
    git curl wget ca-certificates \
    nodejs npm \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Python virtual environment
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install PyTorch nightly with CUDA 13.0 (required for NVIDIA GB10 sm_121 Blackwell)
RUN pip install --no-cache-dir --pre \
    torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu130

# Install SAM3 dependencies first (einops, pycocotools, etc.), then SAM3 itself
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

RUN git clone https://github.com/facebookresearch/sam3.git /opt/sam3 && \
    cd /opt/sam3 && pip install --no-cache-dir -e .

# Build SuperSplat viewer from source
RUN git clone https://github.com/playcanvas/supersplat-viewer.git /opt/supersplat-viewer && \
    cd /opt/supersplat-viewer && \
    npm install && \
    npm run build

# Copy built viewer static files and apply patches
COPY patch_viewer.py /tmp/patch_viewer.py
RUN mkdir -p /app/static/viewer && \
    cp -r /opt/supersplat-viewer/public/* /app/static/viewer/ && \
    python3 /tmp/patch_viewer.py /app/static/viewer && \
    rm /tmp/patch_viewer.py

# Download SAM3 model at build time (requires HF_TOKEN build arg + gated access)
# If download fails (token missing or access not yet granted), model downloads at runtime.
ARG HF_TOKEN=""
ENV HF_TOKEN=${HF_TOKEN}
RUN mkdir -p /app/models/sam3 && \
    if [ -n "$HF_TOKEN" ]; then \
      python3 -c "from huggingface_hub import login, snapshot_download; \
                  login(token='${HF_TOKEN}'); \
                  snapshot_download(repo_id='facebook/sam3', local_dir='/app/models/sam3')" \
      || echo 'WARNING: SAM3 model download failed (gated repo?). Will download at runtime.'; \
    else \
      echo 'WARNING: HF_TOKEN not set. SAM3 model will download on first use.'; \
    fi

# Copy application code
COPY supersplat_service.py /app/supersplat_service.py

# Create model and output directories
RUN mkdir -p /app/splat-models /app/outputs /app/cache

EXPOSE 8086

CMD ["python3", "supersplat_service.py"]
