# Advanced Rendering Containers

This setup provides two specialized containers for high-quality Gaussian Splat rendering:

## 1. USD Pipeline (Port 8002)
Converts PLY files to USD format and renders high-quality images.

**Features:**
- PLY to USD conversion (NVIDIA OpenUSD)
- High-quality offline rendering
- Export rendered images
- GPU-accelerated

**Endpoints:**
- `GET /` - Web UI
- `GET /health` - Health check
- `GET /models` - List available PLY models
- `POST /convert` - Convert PLY to USD
- `POST /render/{model_name}` - Render PLY to image

## 2. Open3D WebRTC Renderer (Port 8888)
Real-time high-quality WebRTC streaming with SuperSplat-quality output.

**Features:**
- Real-time Gaussian Splat rendering
- WebRTC streaming (1920x1080 @ 30 FPS)
- High-quality anti-aliased rendering
- GPU-accelerated visualization

**Endpoints:**
- `GET /` - Status page
- `GET /health` - Health check
- `POST /offer` - WebRTC offer endpoint

## Quick Start

### Start Both Containers
```bash
cd /home/dwatkins3/fvdb-docker
docker compose -f docker-compose.advanced-rendering.yml up --build -d
```

### Check Status
```bash
docker compose -f docker-compose.advanced-rendering.yml ps
docker logs usd_converter
docker logs webrtc_visualizer
```

### Access Services
- **USD Pipeline**: http://localhost:8002
- **WebRTC Renderer**: http://localhost:8888
- **Existing Streaming**: http://localhost:8080/test

### Convert PLY to USD
```bash
curl -X POST http://localhost:8002/convert \
  -H "Content-Type: application/json" \
  -d '{"input_file": "counter_registry_test.ply", "output_name": "counter_usd"}'
```

### Render PLY to Image
```bash
curl -X POST http://localhost:8002/render/counter_registry_test.ply \
  --output rendered_image.png
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Advanced Rendering Stack                   │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐    ┌────────────────┐   ┌──────────────┐ │
│  │ USD Pipeline │    │ Open3D WebRTC  │   │   Original   │ │
│  │  (Port 8002) │    │  (Port 8888)   │   │   Streaming  │ │
│  │              │    │                │   │  (Port 8080) │ │
│  │ • PLY→USD    │    │ • WebRTC       │   │              │ │
│  │ • HQ Render  │    │ • Real-time    │   │ • Fast       │ │
│  │ • Export     │    │ • SuperSplat   │   │ • Proven     │ │
│  └──────────────┘    └────────────────┘   └──────────────┘ │
│                                                               │
│  ┌───────────────────────────────────────────────────────┐  │
│  │         Shared Models Directory (./models)            │  │
│  │  counter_registry_test.ply, counter_latest.ply, etc. │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                               │
│  ┌───────────────────────────────────────────────────────┐  │
│  │       GPU Acceleration (NVIDIA Runtime)               │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## GPU Requirements

Both containers require NVIDIA GPU with Docker runtime:
- CUDA 12.1+ compatible GPU
- nvidia-docker2 installed
- NVIDIA Container Toolkit configured

## Stopping Services

```bash
docker compose -f docker-compose.advanced-rendering.yml down
```

## Integration with Existing Stack

These containers complement your existing setup:
- **Port 8001**: FVDB Rendering (existing)
- **Port 8080**: WebRTC Streaming (existing)
- **Port 8002**: USD Pipeline (new)
- **Port 8888**: High-Quality WebRTC (new)

All containers share the same `./models` directory for seamless integration.
