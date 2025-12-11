# NVIDIA Omniverse Kit USD Viewer Setup

## Overview
Running the official NVIDIA Omniverse Kit USD Viewer streaming application to work with the Omniverse Web Viewer.

## Container Information
- **Image**: nvcr.io/nvidia/omniverse/usd-viewer:107.3.2
- **Source**: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/omniverse/containers/usd-viewer
- **Status**: ✅ Downloaded successfully

## Configuration

### Ports
- **Signaling Port**: 8080 (WebRTC signaling)
- **Media Port**: 8081 (WebRTC media streaming)

### Requirements
1. NVIDIA GPU with drivers installed
2. Docker with NVIDIA Container Toolkit (`nvidia-docker2`)
3. Display server or virtual framebuffer for rendering

## Running the Container

### Basic Command
```bash
docker run --gpus all \
  --network host \
  -e DISPLAY=:0 \
  nvcr.io/nvidia/omniverse/usd-viewer:107.3.2 \
  --no-window \
  --/app/livestream/enabled=true \
  --/app/livestream/port=8080
```

### Using Docker Compose
```bash
cd /home/dwatkins3/fvdb-docker
docker compose -f docker-compose.omniverse-kit.yml up -d
```

### Checking Status
```bash
docker logs omniverse-kit
docker ps | grep omniverse-kit
```

## Connecting from Web Viewer

1. Ensure Omniverse Kit is running
2. Open Web Viewer: http://localhost:5173
3. Select "UI for any streaming app"
4. Click "Next"
5. Web Viewer connects to localhost:8080

## Configuration Files Updated

### stream.config.json
- Server: localhost
- Signaling Port: 8080
- Media Port: 8081

### AppStream.tsx
- Authentication: enabled (for official NVIDIA Kit)
- Removed custom server workarounds

## Troubleshooting

### GPU Not Available
If you don't have an NVIDIA GPU, the Kit application may not start or may need CPU rendering mode.

### Display Issues
If running headless, you may need to:
1. Install Xvfb (virtual framebuffer)
2. Start Xvfb: `Xvfb :0 -screen 0 1920x1080x24 &`
3. Set DISPLAY=:0

### Port Conflicts
Stop custom streaming server before running Kit:
```bash
docker stop streaming-server
```

## Next Steps

1. Verify GPU availability
2. Start Omniverse Kit container
3. Test connection from Web Viewer
4. Load USD models into the viewer

## Custom vs Official Streaming

### Custom Streaming Server (Your Implementation)
- ✅ Gaussian Splat rendering
- ✅ Custom model loading
- ✅ Complete WebRTC stack
- ✅ Test viewer at https://localhost:8080/test

### NVIDIA Omniverse Kit (Official)
- ✅ Full USD support
- ✅ Official NVIDIA streaming
- ✅ Complete Omniverse ecosystem
- ✅ Enterprise-grade features

Both are valuable:
- Use custom server for Gaussian Splat demos
- Use Omniverse Kit for full USD/Omniverse integration
