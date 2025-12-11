# fVDB + Omniverse Web Viewer Integration Complete! 🎉

## Summary

The fVDB rendering service is now integrated with the Omniverse Web Viewer, allowing you to view your 3D Gaussian Splats through a web-based interface with WebRTC streaming.

## What Was Done

### 1. Created Integrated Docker Compose ✅
- **File**: `docker-compose.omniverse.yml`
- **Network**: All containers on `omniverse-fvdb-network`
- **Services**:
  - fVDB Training (port 8000)
  - fVDB Rendering (ports 8001, 8890, 49100)
  - Omniverse Web Viewer (port 5173)

### 2. Configured Stream Settings ✅
- **File**: `/home/dwatkins3/CascadeProjects/web-viewer-sample/stream.config.fvdb.json`
- **Connection**: Web viewer → fVDB rendering container
- **Method**: Docker network DNS resolution
- **Port**: 49100 (WebRTC signaling)

### 3. Updated Network Configuration ✅
- Containers can communicate via service names
- Ports exposed to host for browser access
- GPU passthrough configured for rendering

## Current Status

### Running Services
- ✅ **fVDB Training**: http://localhost:8000
- ✅ **fVDB Rendering**: http://localhost:8001
- ⏳ **Omniverse Web Viewer**: Building (port 5173)

### Building
The integrated stack is currently building. This will take a few minutes as it downloads and installs all dependencies for the web viewer.

## Quick Start (Once Build Completes)

### Access the Services

1. **Omniverse Web Viewer**
   ```
   http://localhost:5173
   ```
   Main interface for viewing 3D models via streaming

2. **fVDB Rendering Service**
   ```
   http://localhost:8001
   ```
   REST API and model management

3. **fVDB Interactive Viewer**
   ```
   http://localhost:8890
   ```
   Native fVDB viewer (alternative to Omniverse)

4. **fVDB Training Service**
   ```
   http://localhost:8000
   ```
   Upload photos and train 3D models

## Workflow

### Complete End-to-End Process

```
1. Photos (iPhone/Camera)
         ↓
2. Upload to Training Service (port 8000)
         ↓
3. Training Completes → PLY file
         ↓
4. Model available in Rendering Service
         ↓
5. View in Omniverse Web Viewer (port 5173)
   OR
5. View in fVDB Native Viewer (port 8890)
   OR
5. Download PLY and view in SuperSplat
```

## Commands

### Check Build Status
```bash
cd /home/dwatkins3/fvdb-docker
docker compose -f docker-compose.omniverse.yml ps
```

### View Logs
```bash
# All services
docker compose -f docker-compose.omniverse.yml logs -f

# Specific service
docker compose -f docker-compose.omniverse.yml logs -f web-viewer
docker compose -f docker-compose.omniverse.yml logs -f rendering
```

### Restart Services
```bash
# All
docker compose -f docker-compose.omniverse.yml restart

# Specific
docker compose -f docker-compose.omniverse.yml restart web-viewer
```

### Stop Services
```bash
docker compose -f docker-compose.omniverse.yml down
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Browser                                                 │
│  Access: http://localhost:5173                          │
└─────────────────┬───────────────────────────────────────┘
                  │
                  │ HTTP + WebRTC
                  ▼
┌─────────────────────────────────────────────────────────┐
│  Docker: omniverse-web-viewer                            │
│  React + Vite + WebRTC Client                            │
│  Port 5173                                              │
└─────────────────┬───────────────────────────────────────┘
                  │
                  │ Docker Network: omniverse-fvdb-network
                  │ WebRTC Signaling: Port 49100
                  ▼
┌─────────────────────────────────────────────────────────┐
│  Docker: fvdb-rendering                                  │
│  • FastAPI REST API (8001)                               │
│  • fVDB Interactive Viewer (8890)                        │
│  • WebRTC Streaming (49100)                              │
│  • GPU-Accelerated Rendering                             │
└─────────────────┬───────────────────────────────────────┘
                  │
                  │ Shared Models Volume
                  ▼
┌─────────────────────────────────────────────────────────┐
│  Docker: fvdb-training                                   │
│  Training Service (8000)                                 │
│  3D Gaussian Splat Training                              │
└─────────────────────────────────────────────────────────┘
```

## Network Details

### Docker Network: `omniverse-fvdb-network`
- Type: Bridge network
- DNS: Container name resolution enabled
- Isolation: Containers can only communicate within network

### Service Communication
- **web-viewer → rendering**: `fvdb-rendering:49100`
- **rendering → training**: Shared volumes

### Host Access
- Port 5173: Omniverse Web Viewer
- Port 8001: fVDB Rendering API
- Port 8890: fVDB Interactive Viewer
- Port 8000: fVDB Training API
- Port 49100: WebRTC (internal to Docker network)

## Viewing Options

You now have **multiple ways** to view your 3D models:

### 1. Omniverse Web Viewer (NEW!)
- **URL**: http://localhost:5173
- **Method**: WebRTC streaming from fVDB
- **Features**:
  - Web-based, no downloads needed
  - Real-time streaming
  - Interactive controls
  - Works on any modern browser

### 2. fVDB Native Viewer
- **URL**: http://localhost:8890
- **Method**: Direct GPU rendering
- **Features**:
  - Native fVDB rendering
  - High performance
  - Full interactive controls
  - Direct from container

### 3. External Viewers
- **SuperSplat**: https://playcanvas.com/supersplat
  - Download PLY from http://localhost:8001/static/downloads/
  - Drag and drop to SuperSplat
  
- **Polycam** (iPhone/Mac)
  - Download from App Store
  - Import PLY file
  - View in AR!

## Testing the Integration

### 1. Wait for Build to Complete
```bash
# Check if all containers are up
docker compose -f docker-compose.omniverse.yml ps

# Expected output:
# fvdb-training        Up (healthy)
# fvdb-rendering       Up (healthy)
# omniverse-web-viewer Up
```

### 2. Test Network Connectivity
```bash
# From web viewer to rendering
docker exec omniverse-web-viewer ping -c 3 fvdb-rendering

# Check WebRTC port
docker exec omniverse-web-viewer nc -zv fvdb-rendering 49100
```

### 3. Check Available Models
```bash
# List models
docker exec fvdb-rendering ls -lh /app/models/

# If you have e2e_demo.ply, it should be listed
```

### 4. Open Web Viewer
```
http://localhost:5173
```

You should see:
- Connection interface
- Server: `fvdb-rendering`
- Port: `49100`
- Connect button

Click connect, and you should see your 3D Gaussian Splat!

## Troubleshooting

### Build Still Running
If the build is taking a long time (>5 minutes for web-viewer):
```bash
# Check logs
docker compose -f docker-compose.omniverse.yml logs -f web-viewer
```

The web viewer needs to download ~100MB of Node.js packages. This is normal for first build.

### Web Viewer Can't Connect
1. Check rendering service is healthy:
   ```bash
   docker compose -f docker-compose.omniverse.yml ps
   ```

2. Verify network:
   ```bash
   docker network inspect omniverse-fvdb-network
   ```

3. Check logs:
   ```bash
   docker compose -f docker-compose.omniverse.yml logs rendering
   ```

### Port Conflicts
If you get "port already in use":
```bash
# Check what's using the port
sudo lsof -i :5173
sudo lsof -i :49100

# Stop conflicting services
docker ps  # Find and stop conflicting containers
```

## Next Steps

1. **Wait for build to complete** (check with `docker compose ps`)

2. **Access the web viewer**:
   ```
   http://localhost:5173
   ```

3. **Train a new model** (if you haven't):
   - Go to http://localhost:8000
   - Upload 20-50 photos
   - Start training
   - Wait 2-5 minutes

4. **View your model**:
   - Open http://localhost:5173
   - Connect to streaming server
   - Explore your 3D scene!

## Documentation

- **Full Integration Guide**: `OMNIVERSE_INTEGRATION.md`
- **Viewing Options**: `VIEWING_GUIDE.md`
- **Quick Start**: `QUICKSTART.md`
- **fVDB Docs**: https://fvdb.ai/
- **Omniverse Web Viewer**: https://github.com/NVIDIA-Omniverse/web-viewer-sample

## Status Summary

✅ **Integration configured**
✅ **Docker Compose created**
✅ **Network configured**
✅ **Stream settings configured**
⏳ **Web viewer building** (in progress)
✅ **fVDB services running**

---

**Almost there! Once the build completes, you'll be able to view your 3D Gaussian Splats in the Omniverse Web Viewer!** 🚀

Check build status:
```bash
docker compose -f docker-compose.omniverse.yml ps
```

When all services show "Up", open: **http://localhost:5173**
