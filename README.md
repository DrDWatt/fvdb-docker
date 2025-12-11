# fVDB Reality Capture Docker Services

Multi-architecture Docker containers for training and rendering Gaussian Splat models with fVDB Reality Capture.

## 🎯 Features

- **Multi-Architecture Support**: ARM64 (DGX Spark, Apple Silicon) and AMD64 (x86_64)
- **Two Services**:
  - **Training Service** (port 8000): Upload datasets, train Gaussian splats
  - **Rendering Service** (port 8001): Render and visualize trained models
- **REST API**: Full Swagger/OpenAPI documentation
- **GPU Acceleration**: NVIDIA CUDA 12.6 support
- **Tutorial Links**: Direct access to fVDB tutorials from Swagger UI

## 📦 Services

### Training Service (http://localhost:8000)

FastAPI service for Gaussian Splat training:
- Upload datasets (ZIP with COLMAP data)
- Download datasets from URLs
- Train models with configurable steps
- Monitor training progress
- Export to PLY format

**Swagger UI**: http://localhost:8000/

### Rendering Service (http://localhost:8001)

Web-based rendering and visualization:
- Upload trained PLY models
- Render images from models
- 3D web viewer
- Depth map generation

**Swagger UI**: http://localhost:8001/api

## 🚀 Quick Start

### Prerequisites

- Docker with BuildKit support
- NVIDIA Docker runtime (for GPU support)
- NVIDIA GPU with CUDA support

### Build Multi-Architecture Images

```bash
# Enable BuildKit
export DOCKER_BUILDKIT=1

# Build for current architecture
docker compose build

# Or build for specific architecture
docker buildx build --platform linux/arm64 -t fvdb-training:arm64 ./training-service
docker buildx build --platform linux/amd64 -t fvdb-training:amd64 ./training-service
```

### Run Services

```bash
# Start both services
docker compose up -d

# View logs
docker compose logs -f

# Stop services
docker compose down
```

### Access Services

- **Training API**: http://localhost:8000 (Swagger UI)
- **Rendering API**: http://localhost:8001/api (Swagger UI)
- **Rendering Home**: http://localhost:8001/

## 📚 Tutorials

Both services include direct links to fVDB tutorials:

1. **Gaussian Splat Radiance Field Reconstruction**
   - https://fvdb.ai/reality-capture/tutorials/radiance_field_and_mesh_reconstruction.html

2. **FRGS Tutorial**
   - https://fvdb.ai/reality-capture/tutorials/frgs.html

3. **fVDB Documentation**
   - https://fvdb.ai/

## 🎓 Usage Examples

### 1. Upload Dataset

```bash
# Upload ZIP containing COLMAP data
curl -X POST "http://localhost:8000/datasets/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@dataset.zip" \
  -F "dataset_name=my_dataset"
```

### 2. Start Training

```bash
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "my_dataset",
    "num_training_steps": 62200,
    "output_name": "my_model"
  }'
```

### 3. Check Training Status

```bash
curl "http://localhost:8000/jobs/{job_id}"
```

### 4. Download Trained Model

```bash
curl -O "http://localhost:8000/outputs/{job_id}/my_model.ply"
```

### 5. Upload to Rendering Service

```bash
curl -X POST "http://localhost:8001/models/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@my_model.ply" \
  -F "model_id=my_model"
```

### 6. View in Browser

```bash
open http://localhost:8001/viewer/my_model
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│         Client (Browser/API)            │
└────────────┬──────────────┬─────────────┘
             │              │
     Port 8000│              │Port 8001
             │              │
┌────────────▼──────────┐  ┌▼──────────────────┐
│  Training Service     │  │ Rendering Service │
│  ─────────────────    │  │ ─────────────────  │
│  - FastAPI            │  │ - FastAPI          │
│  - Swagger UI         │  │ - Web Viewer       │
│  - Dataset Upload     │  │ - Model Rendering  │
│  - Splat Training     │  │ - Image Export     │
│  - Progress Monitor   │  │ - Depth Maps       │
└───────────┬───────────┘  └┬──────────────────┘
            │               │
            │  Shared       │
            └──Volume───────┘
            (Trained Models)
```

## 📁 Volume Structure

```
fvdb-docker/
├── training-data/      # Uploaded datasets
├── training-uploads/   # Temporary upload storage
├── training-outputs/   # Training job outputs
├── shared-models/      # Trained models (shared)
├── rendering-outputs/  # Rendered images
└── rendering-cache/    # Rendering cache
```

## 🔧 Configuration

### Environment Variables

**Training Service:**
```bash
NVIDIA_VISIBLE_DEVICES=all
NVIDIA_DRIVER_CAPABILITIES=compute,utility
```

**Rendering Service:**
```bash
NVIDIA_VISIBLE_DEVICES=all
NVIDIA_DRIVER_CAPABILITIES=compute,utility
```

### Resource Limits

Edit `docker-compose.yml` to adjust GPU allocation:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1  # Number of GPUs
          capabilities: [gpu]
```

## 🧪 Testing

### Test Training Service

```bash
# Health check
curl http://localhost:8000/health

# List datasets
curl http://localhost:8000/datasets

# Get tutorials
curl http://localhost:8000/tutorials
```

### Test Rendering Service

```bash
# Health check
curl http://localhost:8001/health

# List models
curl http://localhost:8001/models

# Get tutorials
curl http://localhost:8001/tutorials
```

## 🐛 Troubleshooting

### GPU Not Detected

```bash
# Check NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi

# Verify GPU access in container
docker compose exec training nvidia-smi
```

### Service Won't Start

```bash
# Check logs
docker compose logs training
docker compose logs rendering

# Restart services
docker compose restart
```

### Port Already in Use

```bash
# Change ports in docker-compose.yml
ports:
  - "8080:8000"  # Training on 8080
  - "8081:8001"  # Rendering on 8081
```

## 📊 Performance

### Training Times (Approximate)

| Dataset Size | Steps  | ARM64 (DGX Spark) | x86_64 (GPU) |
|--------------|--------|-------------------|--------------|
| Small (50)   | 10,000 | ~10 min          | ~8 min       |
| Medium (200) | 30,000 | ~30 min          | ~25 min      |
| Large (300+) | 62,200 | ~2.5 hours       | ~2 hours     |

### Resource Usage

- **Training Service**: 8-16GB GPU RAM, 4-8GB System RAM
- **Rendering Service**: 2-4GB GPU RAM, 2-4GB System RAM

## 🔒 Security Notes

- Services run with GPU access - use in trusted environments
- No authentication by default - add auth layer for production
- Upload size limits not enforced - configure nginx/proxy if needed

## 📝 License

Uses fVDB Reality Capture under its license terms.
NVIDIA CUDA images subject to NVIDIA license.

## 🤝 Contributing

This is a containerized deployment of fVDB Reality Capture.
For fVDB issues, see: https://github.com/fvdb

## 🆘 Support

- **fVDB Documentation**: https://fvdb.ai/
- **Tutorials**: https://fvdb.ai/reality-capture/tutorials/
- **Docker Issues**: Check logs with `docker compose logs`

## ✨ Features Roadmap

- [ ] WebSocket support for live training updates
- [ ] Interactive 3D viewer with Three.js
- [ ] Batch rendering support
- [ ] Model comparison tools
- [ ] Export to multiple formats (USDZ, etc.)
- [ ] Authentication/authorization
- [ ] Cloud storage integration

---

**Built with ❤️ using fVDB Reality Capture**
