# 🚀 fVDB Gaussian Splatting Platform - Quick Start Guide

## One-Command Setup

```bash
# Clone and start everything
git clone <repository-url> fvdb-docker
cd fvdb-docker
docker compose -f docker-compose.master.yml up -d
```

This single command will:
- ✅ Download all required AI models (SAM-2, etc.)
- ✅ Start all 8+ microservices
- ✅ Configure GPU access automatically
- ✅ Set up shared volumes for data persistence

## Prerequisites

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **GPU** | NVIDIA with 8GB VRAM | NVIDIA with 16GB+ VRAM |
| **Docker** | 24.0+ | Latest |
| **NVIDIA Driver** | 535+ | 545+ |
| **Container Toolkit** | nvidia-container-toolkit | Latest |
| **RAM** | 16GB | 32GB+ |
| **Disk** | 50GB free | 100GB+ free |

### Install NVIDIA Container Toolkit

```bash
# Ubuntu/Debian
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## Service URLs & Swagger API Documentation

Once running, access all services at:

| Service | Port | Web UI | Swagger API | Description |
|---------|------|--------|-------------|-------------|
| **🖥️ Viewer** | 8085 | http://localhost:8085 | http://localhost:8085/docs | Main 3D viewer with segmentation |
| **🎓 Training** | 8000 | http://localhost:8000 | http://localhost:8000/api | Video/Photos → Gaussian Splat |
| **🎨 Rendering** | 8001 | http://localhost:8001 | http://localhost:8001/api | PLY model management |
| **🎬 USD** | 8002 | http://localhost:8002 | http://localhost:8002/api | PLY → USD for Omniverse |
| **📸 COLMAP** | 8003 | http://localhost:8003 | http://localhost:8003/docs | Camera pose estimation |
| **🔬 SAM-2** | 8004 | http://localhost:8004 | http://localhost:8004/docs | Object segmentation |
| **🎯 GARField** | 8006 | http://localhost:8006 | http://localhost:8006/docs | 3D object extraction |
| **🌐 Streaming** | 8080 | http://localhost:8080/test | N/A | WebRTC 3D streaming |

## Typical Workflow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Upload    │ ──► │   COLMAP    │ ──► │  Training   │ ──► │   Viewer    │
│   Video     │     │   (8003)    │     │   (8000)    │     │   (8085)    │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                                              │
                         ┌────────────────────┼────────────────────┐
                         ▼                    ▼                    ▼
                   ┌───────────┐        ┌───────────┐        ┌───────────┐
                   │   SAM-2   │        │  GARField │        │    USD    │
                   │ Segment   │        │  Extract  │        │  Convert  │
                   └───────────┘        └───────────┘        └───────────┘
```

### Step-by-Step

1. **Upload Video/Photos** → http://localhost:8000
   - Upload your video or photo collection
   - System extracts frames and runs COLMAP

2. **Train Gaussian Splat** → Automatic after COLMAP
   - Creates .ply model file
   - ~30 minutes for typical scene

3. **View & Interact** → http://localhost:8085
   - Load trained model
   - Rotate, zoom, explore 3D scene

4. **Segment Objects** → Double-click in Viewer
   - Use SAM-2 for object segmentation
   - Add per-object summaries and training data

5. **Export** → http://localhost:8002
   - Convert to USD for Omniverse
   - Extract individual objects

## Common Commands

```bash
# Start all services
docker compose -f docker-compose.master.yml up -d

# Check service status
docker compose -f docker-compose.master.yml ps

# View logs (all services)
docker compose -f docker-compose.master.yml logs -f

# View logs (specific service)
docker compose -f docker-compose.master.yml logs -f fvdb-viewer

# Restart a service
docker compose -f docker-compose.master.yml restart fvdb-viewer

# Stop all services
docker compose -f docker-compose.master.yml down

# Stop and remove volumes (fresh start)
docker compose -f docker-compose.master.yml down -v

# Rebuild all images
docker compose -f docker-compose.master.yml build --no-cache
```

## Directory Structure

```
fvdb-docker/
├── docker-compose.master.yml    # Main compose file (use this!)
├── models/                      # Trained .ply models (shared)
├── uploads/                     # Uploaded videos/photos
├── outputs/                     # Training outputs
├── colmap-data/                 # COLMAP processing data
├── sam2-data/                   # SAM-2 models and outputs
├── garfield-data/               # GARField extraction outputs
├── usd-outputs/                 # Converted USD files
└── docs/                        # Documentation
    └── ARCHITECTURE_DIAGRAM.md  # Full system diagram
```

## Troubleshooting

### GPU Not Detected
```bash
# Verify NVIDIA driver
nvidia-smi

# Verify Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

### Service Won't Start
```bash
# Check logs for specific service
docker compose -f docker-compose.master.yml logs <service-name>

# Restart with fresh build
docker compose -f docker-compose.master.yml up -d --build <service-name>
```

### Out of GPU Memory
- Reduce number of concurrent services
- Use `docker-compose.workflow.yml` for minimal setup

## Architecture Diagram

See `docs/ARCHITECTURE_DIAGRAM.md` for the complete multi-container architecture with all Swagger API URLs.

## Support

- **Documentation**: See `docs/` folder
- **API Docs**: Each service has Swagger at `/api` or `/docs`
- **Health Checks**: All services expose `/health` endpoint
