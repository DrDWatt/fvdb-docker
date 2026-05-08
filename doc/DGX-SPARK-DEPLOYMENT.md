# Reality Engine - DGX Spark / GB10 Deployment Guide

## Overview

This guide deploys the complete Reality Engine Gaussian Splatting Platform on an NVIDIA DGX Spark (Grace Blackwell GB10) workstation. The platform provides end-to-end 3D Gaussian Splat creation from video/photos with AI-powered segmentation, 3D extraction, and robotics integration.

## Prerequisites

### Hardware Requirements
- **NVIDIA DGX Spark** with GB10 (Grace Blackwell) SoC
- **128GB Unified Memory** (shared CPU/GPU)
- **1TB+ NVMe Storage** (images total ~120GB + data/models)
- **Network**: 10GbE or faster recommended for data transfer

### Software Requirements
- **Ubuntu 24.04** (ships with DGX Spark)
- **NVIDIA Driver**: 570+ (pre-installed on DGX Spark)
- **Docker**: 24.0+ with NVIDIA Container Toolkit
- **Docker Compose**: v2.20+
- **AWS CLI v2** (for pulling images from ECR)

### Verify GPU Access
```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi
```

---

## Step 1: Install AWS CLI and Authenticate

```bash
# Install AWS CLI v2 (aarch64)
curl "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" -o /tmp/awscliv2.zip
unzip /tmp/awscliv2.zip -d /tmp/aws-install
sudo /tmp/aws-install/aws/install
rm -rf /tmp/awscliv2.zip /tmp/aws-install

# Configure credentials
aws configure
# Enter:
#   AWS Access Key ID: <provided by admin>
#   AWS Secret Access Key: <provided by admin>
#   Default region: us-east-1
#   Output format: json

# Verify access
aws sts get-caller-identity
```

---

## Step 2: Login to ECR and Pull Images

```bash
# Authenticate Docker with ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin 172176423313.dkr.ecr.us-east-1.amazonaws.com

# Set registry variable
export ECR=172176423313.dkr.ecr.us-east-1.amazonaws.com/reality-engine

# Pull all service images
docker pull $ECR:training-gpu
docker pull $ECR:rendering
docker pull $ECR:usd-pipeline
docker pull $ECR:colmap
docker pull $ECR:sam2
docker pull $ECR:garfield
docker pull $ECR:svo-converter
docker pull $ECR:isaac-sim
docker pull $ECR:isaac-lab
docker pull $ECR:isaac-viewer
docker pull $ECR:streaming-server
docker pull $ECR:trellis

# Pull third-party images
docker pull ollama/ollama:latest
docker pull curlimages/curl:latest
docker pull registry:2
```

---

## Step 3: Retag Images for Local Use

```bash
export ECR=172176423313.dkr.ecr.us-east-1.amazonaws.com/reality-engine

docker tag $ECR:training-gpu fvdb-training-gpu:latest
docker tag $ECR:rendering fvdb-rendering-minimal:latest
docker tag $ECR:usd-pipeline fvdb-docker-usd-pipeline:latest
docker tag $ECR:colmap colmap-service:latest
docker tag $ECR:sam2 sam2-service:latest
docker tag $ECR:garfield garfield-service:latest
docker tag $ECR:svo-converter svo-rosbag-converter:latest
docker tag $ECR:isaac-sim isaac-sim-service:latest
docker tag $ECR:isaac-lab isaac-lab-service:latest
docker tag $ECR:isaac-viewer isaac-viewer:latest
docker tag $ECR:streaming-server omniverse-streaming-server:latest
docker tag $ECR:trellis trellis-service:latest
```

---

## Step 4: Clone the Repository

```bash
cd ~
git clone https://github.com/DrDWatt/fvdb-docker.git
cd fvdb-docker
git checkout main
```

---

## Step 5: Create Data Directories

```bash
cd ~/fvdb-docker

# Create all required data directories
mkdir -p colmap-data/{uploads,processing,outputs,temp}
mkdir -p sam2-data/{uploads,outputs,models,cache}
mkdir -p garfield-data/{outputs,cache}
mkdir -p isaac-data/{svo-uploads,rosbags,scenes,sim-outputs,viewer-outputs,frames,checkpoints,logs,lab-outputs}
mkdir -p trellis-data/{outputs,cache}
mkdir -p ollama-data
mkdir -p usd-outputs
mkdir -p models uploads outputs
mkdir -p cache/torch
```

---

## Step 6: Set Up Python Environment for Training

The training service and fVDB viewer require a host Python environment with CUDA-accelerated packages.

```bash
# Install miniforge (if not present)
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh -b -p $HOME/miniforge3
eval "$($HOME/miniforge3/bin/conda shell.bash hook)"

# Create fvdb environment
conda create -n fvdb python=3.12 -y
conda activate fvdb

# Install PyTorch with CUDA 12.6 (match DGX Spark driver)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install Gaussian Splatting dependencies
pip install plyfile numpy scipy pillow tqdm
pip install fastapi uvicorn python-multipart aiofiles requests
pip install gsplat nerfstudio

# Clone and install fvdb-reality-capture
cd ~
git clone https://github.com/DrDWatt/fvdb-reality-capture.git
```

---

## Step 7: Configure Environment Variables

Create `.env` file in the project root:

```bash
cat > ~/fvdb-docker/.env << 'EOF'
# HuggingFace token (required for TRELLIS model download)
HF_TOKEN=<your_huggingface_token>

# CUDA configuration (DGX Spark GB10)
CUDA_HOME=/usr/local/cuda
TORCH_CUDA_ARCH_LIST=10.0

# Service URLs (internal Docker network)
TRAINING_SERVICE_URL=http://fvdb-training-gpu:8000
RENDERING_SERVICE_URL=http://fvdb-rendering:8001
SAM2_SERVICE_URL=http://sam2-segmentation:8004
OLLAMA_URL=http://ollama-llm:11434
EOF
```

---

## Step 8: Update Docker Compose Paths

Edit `docker-compose.master.yml` to match DGX Spark paths:

```bash
# Replace host Python environment path
sed -i "s|/home/dwatkins3/miniforge3/envs/fvdb|$HOME/miniforge3/envs/fvdb|g" docker-compose.master.yml

# Replace reality-capture path
sed -i "s|/home/dwatkins3/fvdb-reality-capture|$HOME/fvdb-reality-capture|g" docker-compose.master.yml

# Replace CUDA path (DGX Spark uses /usr/local/cuda)
sed -i "s|/usr/local/cuda-13.0|/usr/local/cuda|g" docker-compose.master.yml
```

---

## Step 9: Launch the Platform

```bash
cd ~/fvdb-docker

# Start all services
docker compose -f docker-compose.master.yml up -d

# Monitor startup (wait for all services to be healthy)
watch docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

Expected startup time: ~3-5 minutes for all services to reach healthy state.

---

## Step 10: Verify Deployment

```bash
# Check all services are healthy
docker ps --format "table {{.Names}}\t{{.Status}}" | grep -c healthy

# Test API endpoints
curl -s http://localhost:8000/health | jq .
curl -s http://localhost:8085/health | jq .
curl -s http://localhost:8012/health | jq .
curl -s http://localhost:8003/health | jq .
curl -s http://localhost:8004/health | jq .
```

---

## Service Access

| Service | URL | Description |
|---------|-----|-------------|
| **fVDB Viewer** | http://localhost:8085 | 3D Gaussian Splat Viewer with segmentation |
| **Training Workflow** | http://localhost:8000/workflow | Video/Photo → Gaussian Splat pipeline |
| **ISAAC Viewer** | http://localhost:8012 | SVO → Gaussian Splat workflow |
| **Training API** | http://localhost:8000/api | Training service Swagger docs |
| **COLMAP API** | http://localhost:8003/docs | Structure from Motion API |
| **SAM-2 API** | http://localhost:8004/docs | Segmentation API |
| **GARField API** | http://localhost:8006/docs | 3D extraction API |
| **TRELLIS Viewer** | http://localhost:8013/api | Image → 3D mesh API |
| **Ollama** | http://localhost:11435 | LLM for RAG queries |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DGX Spark / GB10 (aarch64)                        │
│                    128GB Unified Memory                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │
│  │ fVDB Viewer │  │  Training   │  │   ISAAC     │  ← Customer UIs │
│  │   :8085     │  │  Workflow   │  │  Viewer     │                 │
│  │             │  │ :8000/wflow │  │   :8012     │                 │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                │
│         │                 │                 │                        │
│  ┌──────┴─────────────────┴─────────────────┴──────┐               │
│  │              Docker Bridge Network               │               │
│  │            (fvdb-workflow-net)                    │               │
│  └──────┬────────┬────────┬────────┬────────┬──────┘               │
│         │        │        │        │        │                       │
│  ┌──────┴──┐ ┌───┴───┐ ┌──┴──┐ ┌───┴──┐ ┌──┴───┐                 │
│  │ Training│ │COLMAP │ │SAM-2│ │GARFld│ │Ollama│  ← AI/ML         │
│  │  :8000  │ │ :8003 │ │:8004│ │:8006 │ │:11434│                  │
│  └─────────┘ └───────┘ └─────┘ └──────┘ └──────┘                  │
│                                                                     │
│  ┌─────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐         │
│  │Rendering│ │  USD   │ │TRELLIS │ │  SVO   │ │Streaming│ ← Infra │
│  │  :8001  │ │ :8002  │ │ :8013  │ │ :8009  │ │ :8090   │         │
│  └─────────┘ └────────┘ └────────┘ └────────┘ └─────────┘         │
│                                                                     │
│  ┌───────────────────────────────────────────────────────┐         │
│  │              Shared Volumes (NVMe)                     │         │
│  │  ./models  ./colmap-data  ./isaac-data  ./cache       │         │
│  └───────────────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## GPU Memory Usage (Typical)

| Service | VRAM (Approx) | Notes |
|---------|---------------|-------|
| Training | 8-16 GB | During active training |
| SAM-2 | 4-6 GB | When loaded |
| GARField | 4-6 GB | During extraction |
| Ollama (nemotron-mini) | 4-6 GB | When model loaded |
| TRELLIS | 6-10 GB | During reconstruction |
| fVDB Viewer | 2-4 GB | For rendering |
| **Total Peak** | **~40 GB** | All services active simultaneously |

The GB10's 128GB unified memory ensures all services can run concurrently without OOM issues.

---

## Maintenance Commands

```bash
# Restart a single service
docker restart <container-name>

# View logs
docker logs -f fvdb-viewer --tail 100

# Stop everything
docker compose -f docker-compose.master.yml down

# Update from ECR (after new images are pushed)
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin 172176423313.dkr.ecr.us-east-1.amazonaws.com
docker compose -f docker-compose.master.yml pull
docker compose -f docker-compose.master.yml up -d

# Clean unused images
docker image prune -f
```

---

## Troubleshooting

### Service won't start
```bash
docker logs <container-name> --tail 50
```

### GPU not accessible
```bash
# Verify NVIDIA runtime
docker info | grep -i runtime
# Should show: nvidia

# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi
```

### ECR login expired (tokens last 12 hours)
```bash
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin 172176423313.dkr.ecr.us-east-1.amazonaws.com
```

### Out of disk space
```bash
docker system df
docker system prune -a --volumes  # WARNING: removes all unused data
```

---

## ECR Image Tags Reference

All images are stored in: `172176423313.dkr.ecr.us-east-1.amazonaws.com/reality-engine`

| Tag | Service | Size |
|-----|---------|------|
| `training-gpu` | Training + fVDB Viewer | ~19 GB |
| `rendering` | PLY Model Rendering | ~441 MB |
| `usd-pipeline` | PLY → USD Conversion | ~368 MB |
| `colmap` | Structure from Motion | ~1.56 GB |
| `sam2` | SAM-2 Segmentation | ~23.3 GB |
| `garfield` | GARField 3D Extraction | ~22.9 GB |
| `svo-converter` | SVO → ROSBAG | ~12.3 GB |
| `isaac-sim` | ISAAC Sim | ~191 MB |
| `isaac-lab` | ISAAC Lab | ~191 MB |
| `isaac-viewer` | ISAAC Viewer UI | ~759 MB |
| `streaming-server` | WebRTC Streaming | ~2 GB |
| `trellis` | TRELLIS.2 3D Reconstruction | ~37 GB |

**Total storage required**: ~120 GB
