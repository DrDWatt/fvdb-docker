# TRELLIS.2 3D Reconstruction Service
# Requires NVIDIA GPU with 24GB+ VRAM (A100, H100, RTX 4090)
# Build time is ~30min due to CUDA package compilation
FROM nvidia/cuda:13.0.0-devel-ubuntu22.04 AS base
# Detect architecture at build time
ARG TARGETARCH
# HuggingFace token for downloading gated models (DINOv3) at build time
ARG HF_TOKEN

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda-13.0
ENV OPENCV_IO_ENABLE_OPENEXR=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    cmake \
    ninja-build \
    build-essential \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libegl1-mesa \
    libgles2-mesa \
    libglfw3 \
    libopenexr-dev \
    && rm -rf /var/lib/apt/lists/*

# Install conda for TRELLIS.2 environment management
# Detect arch: aarch64 for ARM64 (Jetson/GB10), x86_64 for AMD64
RUN ARCH=$(uname -m) && \
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-${ARCH}.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh
ENV PATH=/opt/conda/bin:$PATH

# Clone TRELLIS.2
WORKDIR /opt
RUN git clone -b main https://github.com/microsoft/TRELLIS.2.git --recursive && \
    cd TRELLIS.2

WORKDIR /opt/TRELLIS.2

# Use bash for all subsequent RUN commands
SHELL ["/bin/bash", "-c"]

# Accept conda TOS for default channels (required by newer Miniconda)
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

# Create conda environment using conda-forge (faster/more reliable than defaults)
RUN conda create -y -n trellis2 -c conda-forge --override-channels python=3.10 || \
    (sleep 10 && conda create -y -n trellis2 -c conda-forge --override-channels python=3.10)

# Install xformers first (brings compatible torch + CUDA bindings)
# TRELLIS.2 supports ATTN_BACKEND=xformers as alternative to flash-attn
ENV ATTN_BACKEND=xformers
ENV MAX_JOBS=4
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0;10.0;12.0"
RUN conda run -n trellis2 \
    pip install xformers

# Install basic dependencies (replicates setup.sh --basic)
RUN apt-get update && apt-get install -y libjpeg-dev libgl-dev libegl-dev && rm -rf /var/lib/apt/lists/*
RUN conda run -n trellis2 \
    pip install imageio imageio-ffmpeg tqdm easydict opencv-python-headless \
    ninja trimesh transformers gradio==6.0.1 tensorboard pandas lpips zstandard \
    kornia timm Pillow \
    git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

# Install nvdiffrast (replicates setup.sh --nvdiffrast)
RUN mkdir -p /tmp/extensions && \
    git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast && \
    conda run -n trellis2 \
    pip install /tmp/extensions/nvdiffrast --no-build-isolation

# Install nvdiffrec (replicates setup.sh --nvdiffrec)
RUN git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git /tmp/extensions/nvdiffrec && \
    conda run -n trellis2 \
    pip install /tmp/extensions/nvdiffrec --no-build-isolation

# Install CuMesh (replicates setup.sh --cumesh)
RUN git clone https://github.com/JeffreyXiang/CuMesh.git /tmp/extensions/CuMesh --recursive && \
    conda run -n trellis2 \
    pip install /tmp/extensions/CuMesh --no-build-isolation

# Install FlexGEMM (replicates setup.sh --flexgemm)
RUN git clone https://github.com/JeffreyXiang/FlexGEMM.git /tmp/extensions/FlexGEMM --recursive && \
    conda run -n trellis2 \
    pip install /tmp/extensions/FlexGEMM --no-build-isolation

# Install O-Voxel from the TRELLIS.2 repo (replicates setup.sh --o-voxel)
RUN cp -r o-voxel /tmp/extensions/o-voxel && \
    conda run -n trellis2 \
    pip install /tmp/extensions/o-voxel --no-build-isolation

# TRELLIS.2 is not pip-installable - add to PYTHONPATH
ENV PYTHONPATH=/opt/TRELLIS.2:$PYTHONPATH

# Install FastAPI service dependencies
RUN conda run -n trellis2 \
    pip install fastapi "uvicorn[standard]" python-multipart aiofiles pydantic einops

# Clean up build artifacts
RUN rm -rf /tmp/extensions

# Patch TRELLIS.2 configs to use non-gated model mirrors:
# - DINOv3: facebook/dinov3-vitl16-pretrain-lvd1689m → PIA-SPACE-LAB/dinov3-vitl-pretrain-lvd1689m
# - RMBG:   briaai/RMBG-2.0 → ZhengPeng7/BiRefNet (same BiRefNet architecture)
RUN find /opt/TRELLIS.2/configs -name "*.json" -exec \
    sed -i 's|facebook/dinov3-vitl16-pretrain-lvd1689m|PIA-SPACE-LAB/dinov3-vitl-pretrain-lvd1689m|g' {} +

# Patch BiRefNet to cast model to float32 (ZhengPeng7/BiRefNet loads as fp16
# but receives fp32 image tensors, causing dtype mismatch)
RUN python3 -c "\
f = '/opt/TRELLIS.2/trellis2/pipelines/rembg/BiRefNet.py'; \
txt = open(f).read(); \
txt = txt.replace('self.model.eval()', 'self.model.float()\n        self.model.eval()'); \
open(f, 'w').write(txt); \
print('Patched BiRefNet to cast model to float32')"

# Patch DINOv3 feature extractor: transformers DINOv3ViTModel wraps encoder
# in .model attribute, so .layer is at .model.model.layer not .model.layer
RUN python3 -c "\
f = '/opt/TRELLIS.2/trellis2/modules/image_feature_extractor.py'; \
txt = open(f).read(); \
txt = txt.replace('self.model.layer', 'self.model.model.layer'); \
open(f, 'w').write(txt); \
print('Patched DINOv3 feature extractor: model.layer -> model.model.layer')"

# Pre-download model weights at build time for instant startup
ENV HF_HOME=/app/model_cache
RUN conda run -n trellis2 \
    python -c "\
import os; os.environ['HF_HOME'] = '/app/model_cache'; \
from huggingface_hub import snapshot_download; \
print('Downloading TRELLIS.2-4B model weights...'); \
snapshot_download('microsoft/TRELLIS.2-4B'); \
print('Downloading DINOv3 (non-gated mirror)...'); \
snapshot_download('PIA-SPACE-LAB/dinov3-vitl-pretrain-lvd1689m'); \
print('Downloading BiRefNet background remover (non-gated)...'); \
snapshot_download('ZhengPeng7/BiRefNet'); \
print('All models downloaded successfully.')"

# Patch the downloaded pipeline.json to use non-gated model repos
RUN find /app/model_cache -name "pipeline.json" -exec \
    sed -i \
    -e 's|facebook/dinov3-vitl16-pretrain-lvd1689m|PIA-SPACE-LAB/dinov3-vitl-pretrain-lvd1689m|g' \
    -e 's|briaai/RMBG-2.0|ZhengPeng7/BiRefNet|g' \
    {} +

WORKDIR /app

# Copy service code
COPY trellis_service.py /app/
COPY static/ /app/static/

# Create directories
RUN mkdir -p /app/outputs /app/cache

EXPOSE 8013

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8013/health || exit 1

# Run with trellis2 conda environment
CMD ["/opt/conda/envs/trellis2/bin/python", "-m", "uvicorn", "trellis_service:app", "--host", "0.0.0.0", "--port", "8013"]
