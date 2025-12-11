# Omniverse Web Viewer with Custom WebRTC Streaming

## Overview

Complete setup with a **custom WebRTC streaming server** that demonstrates real-time streaming to the Omniverse Web Viewer.

## Architecture

```
┌──────────────────────────────────────────┐
│  Browser: http://localhost:5173          │
│  Omniverse Web Viewer                    │
│  (React + WebRTC Client)                 │
└──────────────┬───────────────────────────┘
               │ WebRTC
               ▼
┌──────────────────────────────────────────┐
│  Docker: streaming-server                │
│  • Port 49100/8080: Signaling            │
│  • WebRTC Video Streaming                │
│  • Real-time 3D Visualization            │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│  Shared Models Volume                    │
│  • trained_model.ply (244 MB)            │
│  • 1,086,796 Gaussians                   │
│  • 240 training images                   │
└──────────────────────────────────────────┘
```

## What's Different

### ✅ New: Custom Streaming Server
- **Real WebRTC streaming** (not just file serving)
- **Video track generation** with model visualization
- **aiortc-based** Python WebRTC implementation
- **Compatible with Omniverse Web Viewer**

### ✅ Your Trained Model
- **Removed corrupt test models**
- **Using your latest trained model** (244 MB, 1M+ Gaussians)
- **Real training data** (240 images, 30,000 steps)

## Quick Start

### 1. Build and Start Services

```bash
cd /home/dwatkins3/fvdb-docker

# Build streaming server
docker build -t omniverse-streaming-server:latest streaming-server/

# Start full stack
docker compose -f docker-compose.fullstack.yml up -d

# Check status
docker compose -f docker-compose.fullstack.yml ps
```

### 2. Access Web Viewer

```bash
# Open in browser
open http://localhost:5173
```

### 3. Connect to Stream

1. **Select UI Option**: Choose "UI for any streaming app"
2. **Click "Next"**
3. **Connection Settings**:
   - Server: `streaming-server` (or `localhost` from host)
   - Port: `8080`
4. **Click "Connect"**
5. **✅ See streaming video!**

## Services

### Streaming Server (Port 8080/49100)

**Features:**
- WebRTC signaling
- Video frame generation
- Model metadata display
- Rotating visualization
- Connection status

**Access:**
- Signaling: http://localhost:49100
- Status Page: http://localhost:8080
- Health: http://localhost:8080/health

### Web Viewer (Port 5173)

**Features:**
- WebRTC client
- Stream controls
- Real-time video playback

**Access:**
- Main UI: http://localhost:5173

### Rendering Service (Port 8001)

**Features:**
- Model downloads
- File serving
- API access

**Access:**
- Web UI: http://localhost:8001
- Download model: http://localhost:8001/download/trained_model.ply

## Verification

### Check Services

```bash
# Check all containers
docker compose -f docker-compose.fullstack.yml ps

# Should show:
# streaming-server    Up    0.0.0.0:8080->8080/tcp, 0.0.0.0:49100->8080/tcp
# omniverse-web-viewer Up   0.0.0.0:5173->5173/tcp
# fvdb-rendering       Up    0.0.0.0:8001->8001/tcp
```

### Test Streaming Server

```bash
# Check health
curl http://localhost:8080/health

# Should return:
# {
#   "status": "healthy",
#   "service": "webrtc-streaming-server",
#   "active_connections": 0,
#   "model_loaded": true,
#   "model_info": {...}
# }
```

### View Status Page

```bash
# Open streaming server status
open http://localhost:8080
```

## Model Information

Your trained model:
- **File**: `trained_model.ply`
- **Size**: 244.6 MB
- **Gaussians**: 1,086,796
- **Training Steps**: 30,000
- **Input Images**: 240
- **Device**: CUDA (GPU)
- **Channels**: 3 (RGB)

## How It Works

### WebRTC Flow

1. **Web Viewer** opens connection to `streaming-server:8080`
2. **JavaScript** sends WebRTC offer (SDP)
3. **Streaming Server** receives offer via `/offer` endpoint
4. **Server** creates video track with generated frames
5. **Server** sends back answer (SDP)
6. **WebRTC** establishes peer connection
7. **Video frames** stream from server to client
8. **Web Viewer** displays real-time video

### Video Generation

The streaming server generates frames showing:
- **Title**: "Omniverse WebRTC Streaming"
- **Model Info**: Gaussians, steps, images count
- **Visualization**: Rotating 3D representation
- **Status**: Frame count, rotation angle, connections
- **FPS**: 30 frames per second
- **Resolution**: 1920x1080

## Troubleshooting

### Can't Connect from Web Viewer

**Check network:**
```bash
docker network inspect omniverse-net

# Verify all containers are on same network
```

**Test from web viewer container:**
```bash
docker exec omniverse-web-viewer curl http://streaming-server:8080/health
```

### No Video Showing

**Check streaming server logs:**
```bash
docker logs streaming-server

# Look for:
# "Received offer from client"
# "Sending answer to client"
# "Connection state: connected"
```

### Port Conflicts

If port 49100 or 8080 is in use:
```bash
# Stop conflicting services
docker stop fvdb-rendering

# Or edit docker-compose.fullstack.yml to use different ports
```

## Advanced: Viewing from Browser Console

```javascript
// In Web Viewer (F12 console)
// Check connection status
console.log(window.peerConnection?.connectionState);

// View ICE connection state
console.log(window.peerConnection?.iceConnectionState);

// See video tracks
console.log(window.peerConnection?.getReceivers());
```

## Commands Reference

```bash
# Build
docker build -t omniverse-streaming-server:latest streaming-server/

# Start
docker compose -f docker-compose.fullstack.yml up -d

# Stop
docker compose -f docker-compose.fullstack.yml down

# Logs
docker logs streaming-server
docker logs omniverse-web-viewer
docker logs fvdb-rendering

# Restart
docker restart streaming-server

# Status
docker compose -f docker-compose.fullstack.yml ps
curl http://localhost:8080/health
```

## URLs Summary

| Service | URL | Purpose |
|---------|-----|---------|
| **Web Viewer** | http://localhost:5173 | Main streaming client |
| **Streaming Server** | http://localhost:8080 | Status/signaling |
| **Streaming Signaling** | http://localhost:49100 | WebRTC signaling |
| **Rendering Service** | http://localhost:8001 | Model downloads |
| **Health Check** | http://localhost:8080/health | Server status |

## Next Steps

1. ✅ **Build streaming server** (in progress)
2. 🎯 **Start full stack**: `docker compose -f docker-compose.fullstack.yml up -d`
3. 🌐 **Open web viewer**: http://localhost:5173
4. 🎥 **Connect to stream**: Select "any streaming app", connect to `streaming-server:8080`
5. ✨ **See live streaming** with your trained model info!

---

**Status**: Building custom WebRTC streaming server...
**Model**: Your trained model (244 MB, 1M+ Gaussians) ready
**Next**: Complete build and start streaming!
