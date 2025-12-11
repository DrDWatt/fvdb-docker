# How to View Your 3D Models - Step by Step

## Current Setup Status

✅ **Web Viewer Running**: http://localhost:5173
✅ **Rendering Service Running**: http://localhost:8001
✅ **6 Test Models Available**
⚠️ **WebRTC Streaming**: Not available (minimal service)

## Why the Web Viewer Shows Options

The Omniverse Web Viewer is designed to connect to streaming servers like:
- Omniverse Kit USD Viewer
- Custom streaming applications
- Cloud streaming services

**However**, our minimal rendering service doesn't provide WebRTC streaming due to ARM64 build limitations.

## ✅ WORKING SOLUTION: View Models with SuperSplat

### Step 1: Open Rendering Service
```
http://localhost:8001
```

You'll see a list of 6 test models:
- bunny.ply
- elephant.ply
- sofa.ply
- icosahedron.ply
- sofa_ascii.ply
- icosahedron_ascii.ply

### Step 2: Download a Model

Click the **"📥 Download"** button next to any model (I recommend `bunny.ply` to start).

### Step 3: Open SuperSplat

Go to: https://playcanvas.com/supersplat

### Step 4: Drag and Drop

Drag the downloaded PLY file onto the SuperSplat page.

### Step 5: View!

✅ You'll instantly see your 3D model with:
- Interactive rotation (click and drag)
- Zoom (scroll wheel)
- Pan (right-click and drag)
- High-quality rendering

## Alternative: If You Want WebRTC Streaming

To use the web viewer's streaming functionality, you would need:

### Option A: Run Omniverse Kit (x86_64 only)

```bash
# Omniverse Kit doesn't work on ARM64
# You'd need an x86_64 machine with:
# - NVIDIA GPU
# - Omniverse Kit installed
# - Streaming extensions enabled
# - Port 49100 exposed
```

### Option B: Cloud Streaming

Use NVIDIA's GeForce NOW or Omniverse Cloud streaming services.

## Quick Commands

### Download Models via Command Line

```bash
# Download bunny
curl -O http://localhost:8001/download/bunny.ply

# Download elephant
curl -O http://localhost:8001/download/elephant.ply

# Download all models
cd ~/Downloads
for model in bunny elephant sofa icosahedron sofa_ascii icosahedron_ascii; do
    curl -O http://localhost:8001/download/${model}.ply
done
```

### List Available Models

```bash
curl http://localhost:8001/models
```

### Upload Your Own Model

```bash
curl -X POST "http://localhost:8001/upload" \
  -F "file=@your_model.ply"
```

## What Each Service Does

### 1. PLY Rendering Service (Port 8001)
- **Purpose**: Serve PLY files
- **Features**: Browse, download, upload models
- **Access**: http://localhost:8001
- **Status**: ✅ Working

### 2. Omniverse Web Viewer (Port 5173)
- **Purpose**: WebRTC streaming client
- **Features**: Connect to streaming servers
- **Access**: http://localhost:5173
- **Status**: ✅ Running (no streaming source)

### 3. fVDB Training (Port 8000)
- **Purpose**: Train 3D Gaussian Splats
- **Features**: Upload photos, train models
- **Access**: http://localhost:8000
- **Status**: ✅ Working

## Complete Workflow: Photo → 3D Model → View

### 1. Train a Model

```bash
# Open training service
open http://localhost:8000

# Upload 20-50 photos of an object
# Click "Train"
# Wait 2-5 minutes
```

### 2. Get the Model

```bash
# Download from training outputs
docker exec fvdb-training ls /app/outputs/

# Copy to rendering service
docker cp fvdb-training:/app/outputs/my_model.ply \
  /home/dwatkins3/fvdb-docker/test-models/

# Restart rendering service
docker restart fvdb-rendering
```

### 3. View the Model

```bash
# Download from rendering service
curl -O http://localhost:8001/download/my_model.ply

# View in SuperSplat
open https://playcanvas.com/supersplat
# Drag the file
```

## Why SuperSplat?

**SuperSplat is perfect for 3D Gaussian Splats because:**
- ✅ Web-based (no installation)
- ✅ High-quality rendering
- ✅ Interactive controls
- ✅ Free to use
- ✅ Supports PLY files natively
- ✅ Optimized for Gaussian Splatting
- ✅ Fast loading

## Other Viewing Options

### Polycam (iPhone/Mac)
```
1. Download Polycam from App Store
2. Import PLY file
3. View in AR!
```

### MeshLab (Desktop)
```bash
# Install MeshLab
sudo apt install meshlab  # Linux
# or download from https://www.meshlab.net/

# Open PLY file
meshlab bunny.ply
```

### Blender (Professional)
```bash
# Install Blender
sudo snap install blender  # Linux
# or download from https://www.blender.org/

# Import PLY: File > Import > PLY
```

## About the Web Viewer Radio Buttons

**"UI for default streaming USD Viewer app"**
- For connecting to Omniverse Kit USD Viewer
- Expects Kit streaming on port 49100
- Not available in our setup

**"UI for any streaming app"**
- For custom streaming applications
- Also requires a streaming server
- Not available in our setup

## Summary

### ✅ What Works
1. **Training models** at http://localhost:8000
2. **Serving PLY files** at http://localhost:8001
3. **Downloading models** for external viewing
4. **Viewing in SuperSplat** (recommended!)

### ⚠️ What Doesn't Work
1. **WebRTC streaming** (no streaming server)
2. **Web viewer streaming** (needs streaming source)
3. **Real-time rendering** (use SuperSplat instead)

### 🎯 Recommended Workflow

```
Train → Download → SuperSplat → Done!

1. http://localhost:8000 (train)
2. http://localhost:8001 (download)
3. https://playcanvas.com/supersplat (view)
```

---

**Next Step**: 
1. Go to http://localhost:8001
2. Click "Download" on bunny.ply
3. Open https://playcanvas.com/supersplat
4. Drag the file
5. ✅ See your 3D model!
