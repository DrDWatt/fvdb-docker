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
# ---------------------------------------------------------------------------
# Stage 1: build the SuperSplat viewer (Node 22; Ubuntu 24.04's apt nodejs is v18,
# too old for viewer >= 1.30)
# ---------------------------------------------------------------------------
FROM node:22.23.3-slim AS viewer-build
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# SuperSplat viewer, pinned to v1.27.1+2 (e9fd177, PlayCanvas engine 2.20.6): the build that loads in the
# users' browsers. v1.37.0 (cd0e316, engine 2.23.0) builds and passes the headless tests here but stayed
# on its loading splash in macOS Chrome (2026-10-16); try it with
#   --build-arg SUPERSPLAT_VIEWER_REF=cd0e316e397929ffa87abd4833f73f8a0bd1fd2b
# patch_viewer.py supports both (getCameraMatrices in the bundle for v1.27, via window.app for >= v1.32,
# plus ?orient= for PLYs without cameras).
ARG SUPERSPLAT_VIEWER_REF=e9fd1775de56d3b96b38945fe21f9307c119b053
RUN git clone https://github.com/playcanvas/supersplat-viewer.git /opt/supersplat-viewer && \
    cd /opt/supersplat-viewer && \
    git checkout ${SUPERSPLAT_VIEWER_REF} && \
    npm ci --no-audit --no-fund && \
    npm run build

# ---------------------------------------------------------------------------
# Stage 2: service image
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.6.3-devel-ubuntu24.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies (includes build tools for pycocotools). Node is only needed to build the
# viewer, which now happens in stage 1.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3-pip python3.12-dev \
    build-essential \
    git curl wget ca-certificates \
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

# Pinned to the SAM3 commit validated on GB10 with the torch nightly above
ARG SAM3_REF=46957e47805eaa273f4aa7bbbd25a88bca9108ce
RUN git clone https://github.com/facebookresearch/sam3.git /opt/sam3 && \
    cd /opt/sam3 && git checkout ${SAM3_REF} && pip install --no-cache-dir -e .

# Copy the built viewer static files (stage 1) and apply patches
COPY --from=viewer-build /opt/supersplat-viewer/public /app/static/viewer
COPY patch_viewer.py /tmp/patch_viewer.py
RUN rm -f /app/static/viewer/*.map && \
    python3 /tmp/patch_viewer.py /app/static/viewer && \
    rm /tmp/patch_viewer.py

# Download SAM3 model at build time (requires the hf_token build secret + gated access).
# The token is a BuildKit secret, so it never lands in an image layer or `docker history`;
# at runtime the service gets HF_TOKEN from compose. If download fails (token missing or
# access not yet granted), the model downloads at runtime.
RUN --mount=type=secret,id=hf_token,required=false \
    mkdir -p /app/models/sam3 && \
    if [ -s /run/secrets/hf_token ]; then \
      python3 -c "from huggingface_hub import snapshot_download; \
                  snapshot_download(repo_id='facebook/sam3', local_dir='/app/models/sam3', \
                                    token=open('/run/secrets/hf_token').read().strip())" \
      || echo 'WARNING: SAM3 model download failed (gated repo?). Will download at runtime.'; \
    else \
      echo 'WARNING: hf_token build secret not set. SAM3 model will download on first use.'; \
    fi

# Copy application code (features/ = flythrough + metadata/RAG routers, web/ = their UI).
# ../shared/viewer_common is outside this build context and is mounted by compose.
COPY supersplat_service.py /app/supersplat_service.py
COPY features /app/features
COPY web /app/web

# Create model and output directories
RUN mkdir -p /app/splat-models /app/outputs /app/cache

EXPOSE 8086

CMD ["python3", "supersplat_service.py"]
