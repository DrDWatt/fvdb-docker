# fVDB Rendering Container Build - ARM64 with Streaming

## Current Status: ⏳ BUILDING

The ARM64-compatible rendering service with WebRTC streaming support is currently building.

### Build Progress

```bash
# Check build status
docker build logs

# When complete, you'll see:
# Successfully built <image_id>
# Successfully tagged fvdb-rendering-arm64:latest
```

## What's Being Built

### Dockerfile: `rendering-service/Dockerfile.arm64`

**Key Features:**
- ✅ ARM64 (aarch64) compatible
- ✅ NVIDIA CUDA 12.6 runtime
- ✅ PyTorch with CUDA support
- ✅ WebRTC streaming (aiortc, websockets)
- ✅ FastAPI REST API
- ✅ Omniverse Web Viewer integration
- ❌ Skips problematic dependencies (point-cloud-utils, pye57)

### Service: `rendering_service_streaming.py`

**Capabilities:**
- 📦 Model management (upload, list, info)
- 🌐 WebRTC streaming to Omniverse Web Viewer
- 🎨 Web UI for browsing models
- 📊 REST API with OpenAPI docs
- 💾 Model download support
- 🔄 Streaming session management

## After Build Completes

### 1. Verify Build

```bash
# Check image exists
docker images | grep fvdb-rendering-arm64

# Expected output:
# fvdb-rendering-arm64   latest   <id>   <time>   ~2GB
```

### 2. Start the Service

```bash
cd /home/dwatkins3/fvdb-docker

# Option A: Using docker-compose
docker compose -f docker-compose.streaming.yml up -d

# Option B: Manual run
docker run -d \
  --name fvdb-rendering \
  --gpus all \
  -p 8001:8001 \
  -p 49100:49100 \
  -p 8890:8890 \
  -v $(pwd)/test-models:/app/models:ro \
  --network bridge \
  --restart unless-stopped \
  fvdb-rendering-arm64:latest
```

### 3. Verify Service is Running

```bash
# Check container status
docker ps | grep fvdb-rendering

# Check health
curl http://localhost:8001/health

# Expected response:
# {
#   "status": "healthy",
#   "service": "fvdb-rendering",
#   "streaming_enabled": true,
#   "models_available": 6,
#   "active_streams": 0
# }
```

### 4. Access the Services

| Service | URL | Purpose |
|---------|-----|---------|
| **Rendering API** | http://localhost:8001 | Web UI + API |
| **API Docs** | http://localhost:8001/api | Interactive API documentation |
| **Web Viewer** | http://localhost:5173 | Omniverse WebRTC client |
| **Health Check** | http://localhost:8001/health | Service status |

### 5. Test with Sample Models

```bash
# List available models
curl http://localhost:8001/models

# Start streaming a model
curl -X POST http://localhost:8001/stream/start/bunny.ply

# Check streaming status
curl http://localhost:8001/stream/status
```

### 6. View in Omniverse Web Viewer

1. Open http://localhost:8001 in browser
2. See list of available test models:
   - bunny.ply
   - elephant.ply
   - sofa.ply
   - icosahedron.ply
3. Click "Start Streaming" on any model
4. Open http://localhost:5173 (Omniverse Web Viewer)
5. Viewer connects to `fvdb-rendering:49100`
6. View your 3D model!

## Test Models Included

Located in `/home/dwatkins3/fvdb-docker/test-models/`:

1. **bunny.ply** - Stanford Bunny (35KB)
2. **elephant.ply** - Elephant mesh (72KB)
3. **sofa.ply** - Sofa point cloud (45KB)
4. **icosahedron.ply** - Simple geometry (1KB)
5. **sofa_ascii.ply** - ASCII version (198KB)
6. **icosahedron_ascii.ply** - ASCII version (5KB)

## Adding Your Trained Models

### From Training Service

```bash
# Copy from training outputs
docker cp fvdb-training:/app/outputs/your_model.ply \
  /home/dwatkins3/fvdb-docker/test-models/

# Restart rendering service to pick up new model
docker restart fvdb-rendering
```

### Direct Upload

```bash
# Upload via API
curl -X POST "http://localhost:8001/models/upload" \
  -F "file=@/path/to/your_model.ply"
```

### Mount Volume

```bash
# Add to docker-compose.streaming.yml:
volumes:
  - /path/to/your/trained/models:/app/models:ro
```

## Network Configuration

### Container Network

Both containers are on Docker's default `bridge` network:

```
omniverse-web-viewer-dev (port 5173)
    ↕ bridge network
fvdb-rendering (ports 8001, 49100, 8890)
```

DNS resolution allows web-viewer to reach `fvdb-rendering` by name.

### Ports

| Port | Service | Protocol | Purpose |
|------|---------|----------|---------|
| 8001 | REST API | HTTP | Web UI, model management |
| 49100 | Streaming | WebRTC | Signaling for Omniverse |
| 8890 | Viewer | HTTP | Alternative fVDB viewer |
| 5173 | Web Viewer | HTTP | Omniverse React client |

## Troubleshooting

### Build Fails

```bash
# Check build logs
docker build -f rendering-service/Dockerfile.arm64 \
  -t fvdb-rendering-arm64:latest \
  rendering-service/ 2>&1 | tee build.log

# Common issues:
# - Network timeout: Retry build
# - Disk space: docker system prune
# - Memory: Increase Docker memory limit
```

### Service Won't Start

```bash
# Check logs
docker logs fvdb-rendering

# Check GPU access
docker exec fvdb-rendering nvidia-smi

# Restart service
docker restart fvdb-rendering
```

### Web Viewer Can't Connect

```bash
# Verify network
docker network inspect bridge

# Test connectivity
docker exec omniverse-web-viewer-dev ping -c 2 fvdb-rendering

# Check port 49100 is accessible
docker exec fvdb-rendering netstat -tuln | grep 49100
```

## Build Timeline

Typical build time on DGX Spark ARM64:

1. **Base image pull**: ~2 minutes (CUDA runtime)
2. **System packages**: ~1 minute (apt-get install)
3. **PyTorch download**: ~3-5 minutes (~700MB)
4. **Python packages**: ~2 minutes (fastapi, etc.)
5. **WebRTC libraries**: ~1 minute (aiortc, websockets)
6. **Total**: ~8-12 minutes

## What's Different from Original

### Removed (ARM64 incompatible):
- ❌ `fvdb-reality-capture` full package
- ❌ `point-cloud-utils` (C++ compilation)
- ❌ `pye57` (missing xercesc)
- ❌ Complex reconstruction features

### Added (streaming support):
- ✅ `aiortc` (WebRTC support)
- ✅ `websockets` (signaling)
- ✅ `aiohttp` (async HTTP)
- ✅ CORS middleware
- ✅ Streaming session management
- ✅ Model upload endpoint

### Simplified:
- ✅ Core rendering only
- ✅ PLY file support
- ✅ REST API focused
- ✅ Web UI for browsing
- ✅ Multiple download options

## Success Criteria

Build is successful when:

- ✅ Docker image tagged: `fvdb-rendering-arm64:latest`
- ✅ Image size: ~1.5-2.5 GB
- ✅ Container starts without errors
- ✅ Health endpoint returns 200 OK
- ✅ Models directory is accessible
- ✅ Ports 8001, 49100, 8890 are listening

## Next Steps After Build

1. ✅ Start service: `docker compose -f docker-compose.streaming.yml up -d`
2. ✅ Check health: `curl http://localhost:8001/health`
3. ✅ Browse models: Open http://localhost:8001
4. ✅ Start streaming: Click "Start Streaming" on a model
5. ✅ View in browser: Open http://localhost:5173
6. ✅ Connect and interact with 3D models!

## Documentation

- **Setup Guide**: `STREAMING_SETUP.md`
- **Simple Guide**: `SIMPLE_SETUP_GUIDE.md`
- **Viewing Options**: `VIEWING_GUIDE.md`
- **This Status**: `BUILD_STATUS.md`

---

**Current Status**: Building... check progress with:

```bash
docker ps -a | grep build
# or
docker images | grep fvdb-rendering-arm64
```

When you see `fvdb-rendering-arm64:latest` in the images list, the build is complete!
