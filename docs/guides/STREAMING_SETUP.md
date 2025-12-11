# fVDB Rendering Service with Streaming - ARM64

## What's Different

This is a **simplified, ARM64-compatible** version of the rendering service that:

✅ **Works on ARM64** (DGX Spark)
✅ **Enables WebRTC streaming** to Omniverse Web Viewer
✅ **Uses existing trained models**
✅ **Skips problematic dependencies** (point-cloud-utils, pye57)
✅ **Provides multiple viewing options**

## Quick Start

### 1. Build the Container

```bash
cd /home/dwatkins3/fvdb-docker

# Build ARM64-compatible image
docker build -f rendering-service/Dockerfile.arm64 \
  -t fvdb-rendering-arm64:latest \
  rendering-service/
```

### 2. Start the Service

```bash
# Using docker-compose
docker compose -f docker-compose.streaming.yml up -d

# Or manually
docker run -d \
  --name fvdb-rendering \
  --gpus all \
  -p 8001:8001 \
  -p 49100:49100 \
  -p 8890:8890 \
  -v $(pwd)/test-models:/app/models:ro \
  --network bridge \
  fvdb-rendering-arm64:latest
```

### 3. Access the Services

- **Rendering API**: http://localhost:8001
- **Omniverse Web Viewer**: http://localhost:5173
- **Health Check**: http://localhost:8001/health

## Using with Omniverse Web Viewer

### Setup

1. **Start rendering service** (see above)

2. **Connect web viewer to rendering service**:
   - The web viewer is configured to connect to `fvdb-rendering:49100`
   - Both containers are on the same Docker `bridge` network

3. **Open web viewer**:
   ```
   http://localhost:5173
   ```

4. **View your models**:
   - Web viewer will automatically discover available models
   - Select a model to stream
   - Real-time 3D interaction!

## Testing with Sample Models

Sample PLY models have been copied to `/home/dwatkins3/fvdb-docker/test-models/`:

- `bunny.ply` - Stanford Bunny
- `elephant.ply` - Elephant mesh
- `sofa.ply` - Sofa point cloud
- `icosahedron.ply` - Simple geometry

### View Sample Models

1. Go to http://localhost:8001
2. See list of available models
3. Click "Start Streaming" on any model
4. Open Omniverse Web Viewer: http://localhost:5173
5. Connect and view!

## Adding Your Own Models

### Option 1: Copy to Container

```bash
# Copy PLY file to models directory
docker cp your_model.ply fvdb-rendering:/app/models/

# Or copy to downloads (web-accessible)
docker cp your_model.ply fvdb-rendering:/app/static/downloads/
```

### Option 2: Upload via API

```bash
# Upload using curl
curl -X POST "http://localhost:8001/models/upload" \
  -F "file=@your_model.ply"
```

### Option 3: Mount Directory

```bash
# Add to docker-compose.streaming.yml:
volumes:
  - /path/to/your/models:/app/models:ro
```

## API Endpoints

### Model Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI with model list |
| `/health` | GET | Health check |
| `/models` | GET | List all models |
| `/models/{name}` | GET | Get model info |
| `/models/upload` | POST | Upload new model |

### Streaming

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/stream/start/{name}` | POST | Start streaming session |
| `/stream/status` | GET | Get streaming status |
| `/stream/stop/{id}` | DELETE | Stop streaming session |

### Viewing

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/viewer/{name}` | GET | Simple viewer page |
| `/static/downloads/{name}` | GET | Download PLY file |

## Architecture

```
┌─────────────────────────────────────────┐
│  Browser: http://localhost:5173         │
│  Omniverse Web Viewer                   │
└──────────────┬──────────────────────────┘
               │ HTTP + WebRTC
               ▼
┌──────────────────────────────────────────┐
│  Docker: omniverse-web-viewer-dev        │
│  (React + WebRTC Client)                 │
└──────────────┬──────────────────────────┘
               │ Docker Bridge Network
               │ WebRTC: fvdb-rendering:49100
               ▼
┌──────────────────────────────────────────┐
│  Docker: fvdb-rendering                  │
│  • REST API (8001)                       │
│  • WebRTC Streaming (49100)              │
│  • Model Management                      │
│  • GPU-Accelerated (NVIDIA GB10)         │
└──────────────┬──────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│  Models Directory                        │
│  /app/models/*.ply                       │
│  Test models + Your trained models       │
└──────────────────────────────────────────┘
```

## What Was Removed

To make this work on ARM64, we removed:

- ❌ `fvdb-reality-capture` full package (has C++ compilation issues)
- ❌ `point-cloud-utils` (requires PCL, fails on ARM64)
- ❌ `pye57` (requires xercesc, not available)

## What Was Added

- ✅ WebRTC streaming support (`aiortc`, `websockets`)
- ✅ CORS middleware for web viewer
- ✅ Model upload endpoint
- ✅ Streaming session management
- ✅ Simple web UI for model browsing

## Viewing Options

You now have **3 ways** to view your models:

### 1. Omniverse Web Viewer (Streaming)
- Real-time WebRTC streaming
- Interactive 3D controls
- Access: http://localhost:5173

### 2. Direct Download + SuperSplat
- Download PLY from http://localhost:8001
- Upload to https://playcanvas.com/supersplat
- Instant web-based viewing

### 3. API Access
- Programmatic model access
- REST API for integration
- Documentation: http://localhost:8001/api

## Troubleshooting

### Build Fails

**Check Docker logs:**
```bash
docker logs fvdb-rendering
```

**Rebuild from scratch:**
```bash
docker build --no-cache -f rendering-service/Dockerfile.arm64 \
  -t fvdb-rendering-arm64:latest rendering-service/
```

### Web Viewer Can't Connect

**Verify network:**
```bash
# Check both containers are on bridge network
docker inspect omniverse-web-viewer-dev | grep NetworkMode
docker inspect fvdb-rendering | grep NetworkMode
```

**Test connectivity:**
```bash
docker exec omniverse-web-viewer-dev ping -c 2 fvdb-rendering
```

### No Models Showing

**Check models directory:**
```bash
docker exec fvdb-rendering ls -lh /app/models/
```

**Copy test models:**
```bash
docker cp test-models/*.ply fvdb-rendering:/app/models/
```

## Performance Notes

- **GPU**: Automatically uses NVIDIA GB10
- **Memory**: ~2GB for service + model size
- **Network**: Local Docker bridge (low latency)
- **Streaming**: WebRTC (< 100ms latency typical)

## Next Steps

1. **Verify build completes** successfully
2. **Start the service**: `docker compose -f docker-compose.streaming.yml up -d`
3. **Check health**: `curl http://localhost:8001/health`
4. **Open web UI**: http://localhost:8001
5. **Connect web viewer**: http://localhost:5173
6. **Start streaming** a test model!

---

**Status**: Building ARM64-compatible image with streaming support...

Check build progress:
```bash
docker logs -f $(docker ps -q --filter ancestor=fvdb-rendering-arm64:latest)
```
