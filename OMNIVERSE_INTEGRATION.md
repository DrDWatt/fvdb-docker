# Omniverse Web Viewer Integration with fVDB

## Overview

This integration connects the Omniverse Web Viewer to the fVDB rendering service, allowing you to view 3D Gaussian Splats through a web browser with WebRTC streaming.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Browser (localhost:5173)                                        │
│  Omniverse Web Viewer - React Client                            │
└────────────────────────┬────────────────────────────────────────┘
                         │ HTTP + WebRTC
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Docker Container: omniverse-web-viewer                          │
│  Vite Dev Server + WebRTC Client                                │
└────────────────────────┬────────────────────────────────────────┘
                         │ Docker Network: omniverse-fvdb-network
                         │ WebRTC Signaling: Port 49100
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Docker Container: fvdb-rendering                                │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  fVDB Rendering Service (FastAPI)                        │ │
│  │  • Port 8001: REST API + Web UI                          │ │
│  │  • Port 8890: fVDB Interactive Viewer                    │ │
│  │  • Port 49100: WebRTC Signaling (Omniverse streaming)    │ │
│  └───────────────────────────────────────────────────────────┘ │
│  GPU-accelerated Gaussian Splat rendering                       │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│  Docker Container: fvdb-training                                 │
│  Training service for 3D reconstruction                          │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Stop Existing Containers

```bash
cd /home/dwatkins3/fvdb-docker

# Stop current containers
docker compose down

# Stop web viewer if running separately
cd /home/dwatkins3/CascadeProjects/web-viewer-sample
docker compose down
```

### 2. Launch Integrated Stack

```bash
cd /home/dwatkins3/fvdb-docker

# Start all services with Omniverse integration
docker compose -f docker-compose.omniverse.yml up -d

# View logs
docker compose -f docker-compose.omniverse.yml logs -f
```

### 3. Access Services

- **Omniverse Web Viewer**: http://localhost:5173
- **fVDB Rendering API**: http://localhost:8001
- **fVDB Interactive Viewer**: http://localhost:8890
- **Training Service**: http://localhost:8000

## Services

### 1. fVDB Training Service
- **Port**: 8000
- **Purpose**: Train 3D Gaussian Splats from photos
- **API**: Upload photos, start training jobs
- **Output**: PLY files with 3D splats

### 2. fVDB Rendering Service
- **Ports**: 
  - 8001: REST API and web interface
  - 8890: fVDB native viewer
  - 49100: WebRTC signaling for Omniverse
- **Purpose**: Render and stream 3D Gaussian Splats
- **Features**:
  - Load and render PLY files
  - Interactive 3D visualization
  - WebRTC streaming to Omniverse Web Viewer
  - REST API for rendering

### 3. Omniverse Web Viewer
- **Port**: 5173
- **Purpose**: Web-based client for viewing streamed content
- **Features**:
  - WebRTC streaming client
  - React-based UI
  - Real-time interaction with 3D models

## Configuration

### Stream Configuration

The web viewer connects to fVDB rendering via `stream.config.fvdb.json`:

```json
{
  "source": "local",
  "local": {
    "server": "fvdb-rendering",
    "signalingPort": 49100
  }
}
```

- **server**: Docker container name (DNS resolution via Docker network)
- **signalingPort**: WebRTC signaling port

### Environment Variables

**fVDB Rendering Container:**
- `OMNIVERSE_STREAMING_ENABLED=true`: Enable WebRTC streaming
- `OMNIVERSE_STREAMING_PORT=49100`: WebRTC signaling port

## Workflow

### Complete End-to-End Workflow

1. **Capture Photos** (iPhone/Camera)
   ```
   Take 20-50 photos around your subject
   ```

2. **Upload to Training Service**
   ```
   http://localhost:8000
   Upload photos → Start training
   ```

3. **Training Completes**
   ```
   Produces: e2e_demo.ply (Gaussian Splat model)
   Location: /app/models/e2e_demo.ply
   ```

4. **Load in Rendering Service**
   ```
   Model automatically available to rendering service
   ```

5. **View in Omniverse Web Viewer**
   ```
   http://localhost:5173
   Connect → See streamed 3D visualization
   ```

## Useful Commands

### View Logs

```bash
# All services
docker compose -f docker-compose.omniverse.yml logs -f

# Specific service
docker compose -f docker-compose.omniverse.yml logs -f rendering
docker compose -f docker-compose.omniverse.yml logs -f web-viewer
docker compose -f docker-compose.omniverse.yml logs -f training
```

### Restart Services

```bash
# Restart all
docker compose -f docker-compose.omniverse.yml restart

# Restart specific service
docker compose -f docker-compose.omniverse.yml restart rendering
docker compose -f docker-compose.omniverse.yml restart web-viewer
```

### Stop Services

```bash
docker compose -f docker-compose.omniverse.yml down
```

### Access Container Shell

```bash
# Rendering service
docker exec -it fvdb-rendering bash

# Web viewer
docker exec -it omniverse-web-viewer sh

# Training service
docker exec -it fvdb-training bash
```

## Testing the Integration

### 1. Check Services Are Running

```bash
docker compose -f docker-compose.omniverse.yml ps
```

Expected output:
- `fvdb-training`: Up and healthy
- `fvdb-rendering`: Up and healthy
- `omniverse-web-viewer`: Up

### 2. Verify Network Connectivity

```bash
# Test from web viewer to rendering
docker exec omniverse-web-viewer ping -c 3 fvdb-rendering

# Check if port 49100 is open
docker exec omniverse-web-viewer nc -zv fvdb-rendering 49100
```

### 3. Check Available Models

```bash
# List models in rendering service
docker exec fvdb-rendering ls -lh /app/models/
```

### 4. Access Web Viewer

Open browser to http://localhost:5173

You should see:
- Connection interface
- Option to connect to streaming server
- Once connected: 3D visualization of your Gaussian Splat

## Troubleshooting

### Web Viewer Can't Connect

**Symptoms**: "Connection failed" or "Unable to connect to streaming server"

**Solutions**:

1. Check containers are on same network:
   ```bash
   docker network inspect omniverse-fvdb-network
   ```

2. Verify rendering container is healthy:
   ```bash
   docker compose -f docker-compose.omniverse.yml ps
   ```

3. Check port 49100 is accessible:
   ```bash
   docker exec fvdb-rendering netstat -tuln | grep 49100
   ```

4. View rendering service logs:
   ```bash
   docker compose -f docker-compose.omniverse.yml logs rendering
   ```

### No Models Available

**Symptoms**: Empty model list

**Solutions**:

1. Check models directory:
   ```bash
   docker exec fvdb-rendering ls -lh /app/models/
   ```

2. Copy a model to rendering service:
   ```bash
   docker exec fvdb-training cp /app/outputs/your_model.ply /app/models/
   ```

3. Train a new model:
   - Go to http://localhost:8000
   - Upload photos
   - Start training

### Port Conflicts

**Symptoms**: "Port already in use"

**Solutions**:

1. Check what's using the ports:
   ```bash
   sudo lsof -i :5173
   sudo lsof -i :8001
   sudo lsof -i :49100
   ```

2. Stop conflicting services:
   ```bash
   # If web viewer running separately
   cd /home/dwatkins3/CascadeProjects/web-viewer-sample
   docker compose down
   ```

3. Modify ports in `docker-compose.omniverse.yml` if needed:
   ```yaml
   ports:
     - "5174:5173"  # Change host port (left side)
   ```

### GPU Not Available

**Symptoms**: CUDA errors, slow rendering

**Solutions**:

1. Check GPU is accessible in container:
   ```bash
   docker exec fvdb-rendering nvidia-smi
   ```

2. Verify nvidia-container-toolkit is installed:
   ```bash
   docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
   ```

## Advanced Configuration

### Using Alternative Viewers

You still have access to other viewing methods:

1. **fVDB Native Viewer** (Port 8890):
   - Direct GPU-accelerated viewing
   - Built-in interactive controls

2. **SuperSplat** (External):
   - Download PLY from http://localhost:8001/static/downloads/
   - Upload to https://playcanvas.com/supersplat

3. **REST API** (Port 8001):
   - Programmatic rendering
   - Custom camera views
   - See API docs at http://localhost:8001/api

### Custom Stream Configuration

Edit `/home/dwatkins3/CascadeProjects/web-viewer-sample/stream.config.fvdb.json`:

```json
{
  "source": "local",
  "local": {
    "server": "fvdb-rendering",  // Container name
    "signalingPort": 49100,      // WebRTC port
    "mediaPort": null            // Auto-select
  }
}
```

### Performance Tuning

**For Better Streaming Quality:**

1. Increase rendering resolution in fVDB service
2. Adjust camera settings in web viewer
3. Monitor GPU usage: `nvidia-smi -l 1`

**For Lower Latency:**

1. Reduce image quality in WebRTC settings
2. Use local network (avoid VPN/proxy)
3. Ensure GPU has sufficient memory

## Network Architecture

All containers are on the `omniverse-fvdb-network` Docker bridge network:

- **Container DNS**: Containers can reach each other by name
  - `fvdb-training`
  - `fvdb-rendering`
  - `omniverse-web-viewer`

- **Host Access**: Ports are exposed to host machine
  - 8000 → Training API
  - 8001 → Rendering API
  - 8890 → fVDB Viewer
  - 5173 → Omniverse Web Viewer
  - 49100 → WebRTC Signaling

## Current Status

✅ **Web Viewer**: Running (http://localhost:5173)
✅ **fVDB Rendering**: Running (http://localhost:8001)
✅ **fVDB Training**: Running (http://localhost:8000)
⏳ **WebRTC Streaming**: Ready (port 49100)

## Next Steps

1. **Test the connection**:
   ```bash
   docker compose -f docker-compose.omniverse.yml up -d
   # Open http://localhost:5173
   ```

2. **Train a model** (if you haven't):
   - Go to http://localhost:8000
   - Upload photos
   - Start training

3. **View in Omniverse**:
   - Open http://localhost:5173
   - Connect to streaming server
   - Explore your 3D model!

## Support

- **fVDB Documentation**: https://fvdb.ai/
- **Omniverse Web Viewer**: https://github.com/NVIDIA-Omniverse/web-viewer-sample
- **Docker Compose Reference**: https://docs.docker.com/compose/

---

**Ready to view your 3D Gaussian Splats in Omniverse Web Viewer!** 🎉
