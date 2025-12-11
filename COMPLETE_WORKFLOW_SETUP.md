# 🚀 Complete 3D Reconstruction Workflow

## What's New

A **complete end-to-end pipeline** for creating 3D models from videos or photos!

### The Full Stack

```
Upload (MP4/MOV/ZIP) 
    ↓
COLMAP Processing (Structure-from-Motion)
    ↓
GPU Training (Gaussian Splat)
    ↓
3D Model (PLY + USD)
```

---

## 🎯 One-Page Workflow UI

**Access at**: http://localhost:8080/workflow

### Features

- **📤 File Upload**: Drag & drop MP4, MOV, or ZIP files
- **🔬 COLMAP Processing**: Automatic structure-from-motion
- **🎯 GPU Training**: Fast Gaussian Splat training
- **📦 Results**: Download PLY models, convert to USD
- **📊 Real-time Status**: Watch progress of each step
- **🎨 Beautiful UI**: Modern, responsive interface

---

## 🛠️ New COLMAP Service

**API**: http://localhost:8003  
**Docs**: http://localhost:8003/api

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/upload` | POST | Upload ZIP of photos |
| `/video/extract` | POST | Extract frames from video |
| `/process` | POST | Run COLMAP processing |
| `/jobs/{id}` | GET | Check processing status |
| `/download/{id}` | GET | Download processed dataset |

### Supported Video Formats

✅ **MP4** - Yes  
✅ **MOV** - Yes (tested and working!)  
✅ **AVI** - Yes  
✅ **MKV** - Yes  
✅ **WEBM** - Yes  

All formats supported by FFmpeg work!

---

## 📦 Quick Start

### 1. Build COLMAP Service

```bash
cd /home/dwatkins3/fvdb-docker

# Build COLMAP container (takes ~10-15 minutes first time)
docker build -t colmap-service:latest colmap-service/

# Or use the workflow compose file
docker compose -f docker-compose.workflow.yml build colmap-service
```

### 2. Start All Services

```bash
# Start complete workflow stack
docker compose -f docker-compose.workflow.yml up -d

# Check status
docker compose -f docker-compose.workflow.yml ps
```

### 3. Open Workflow Page

```bash
open http://localhost:8080/workflow
```

---

## 🎬 Using the Workflow

### Step 1: Upload Media

1. **For Videos** (MP4, MOV):
   - Drag & drop or click to browse
   - Set FPS (recommended: 2-4 for most scenes)
   - Optionally name your dataset
   - Click "Upload & Extract"

2. **For Photos** (ZIP):
   - Create ZIP with all photos in root or `images/` folder
   - Drag & drop ZIP file
   - Click "Upload & Extract"

### Step 2: COLMAP Processing

1. **Choose Quality**:
   - **Low**: Fast (~2-5 min), good for testing
   - **Medium**: Balanced (~5-15 min), recommended
   - **High**: Better quality (~15-30 min)
   - **Extreme**: Best quality (~30-60 min)

2. **Choose Matcher**:
   - **Exhaustive**: Best for <100 images, all-around
   - **Sequential**: For video sequences

3. Click "Run COLMAP"

4. Watch progress bar and status messages

### Step 3: GPU Training

1. **Choose Training Steps**:
   - **7,000**: Quick preview (~5 min)
   - **30,000**: Production quality (~20 min) ⭐ Recommended
   - **62,200**: Maximum quality (~45 min)

2. Name your model

3. Click "Start Training"

4. Watch training progress

### Step 4: Download Results

- Download PLY model
- Convert to USD for Blender/Omniverse
- View in 3D viewer

---

## 🏗️ Architecture

```
┌─────────────────────────────────────┐
│  Workflow UI (localhost:8080)       │
│  - File upload                      │
│  - Status tracking                  │
│  - Results display                  │
└──────────┬──────────────────────────┘
           │
           ├──────► COLMAP Service (8003)
           │        - Video extraction
           │        - Photo upload
           │        - SfM processing
           │        - GPU-accelerated
           │
           ├──────► Training Service (8000)
           │        - Gaussian Splat training
           │        - GPU-accelerated
           │        - fVDB Reality Capture
           │
           ├──────► USD Pipeline (8002)
           │        - PLY → USD conversion
           │        - High-quality rendering
           │
           └──────► Rendering Service (8001)
                    - Model management
                    - PLY downloads
```

---

## 📊 Service URLs

| Service | URL | Purpose |
|---------|-----|---------|
| **Workflow UI** | http://localhost:8080/workflow | Complete pipeline interface |
| **3D Viewer** | http://localhost:8080/test | View streaming models |
| **COLMAP API** | http://localhost:8003/api | COLMAP documentation |
| **Training API** | http://localhost:8000/api | Training documentation |
| **USD Pipeline** | http://localhost:8002 | USD conversion UI |
| **Rendering** | http://localhost:8001 | Model downloads |

---

## 🎯 Example Workflows

### From Video to 3D Model

```bash
# 1. Open workflow page
open http://localhost:8080/workflow

# 2. Upload video.mp4 (drag & drop)

# 3. Set FPS to 2, click "Upload & Extract"
#    → Wait for extraction (1-2 min)

# 4. Set Quality to "Medium", click "Run COLMAP"
#    → Wait for processing (5-15 min)

# 5. Set Steps to 30000, click "Start Training"
#    → Wait for training (~20 min)

# 6. Download your 3D model!
```

### From Photos to 3D Model

```bash
# 1. Create photos.zip with your images
zip -r photos.zip *.jpg

# 2. Open workflow page and upload photos.zip

# 3. Click "Upload & Extract"

# 4. Run COLMAP (Medium quality)

# 5. Train model (30K steps)

# 6. Download result
```

---

## ⚙️ COLMAP Processing Details

### What COLMAP Does

1. **Feature Extraction**
   - Detects SIFT features in each image
   - Uses GPU acceleration
   - Configurable quality settings

2. **Feature Matching**
   - Matches features between images
   - Exhaustive or sequential matching
   - GPU-accelerated

3. **Sparse Reconstruction**
   - Estimates camera poses
   - Triangulates 3D points
   - Bundle adjustment optimization

4. **Output**
   - `sparse/0/cameras.bin` - Camera parameters
   - `sparse/0/images.bin` - Image poses
   - `sparse/0/points3D.bin` - 3D point cloud

### Quality vs Speed

| Quality | Max Image Size | Features | Time (50 images) |
|---------|----------------|----------|------------------|
| Low | 1600px | 4,000 | ~2-5 min |
| Medium | 2400px | 8,000 | ~5-15 min |
| High | 3200px | 16,000 | ~15-30 min |
| Extreme | 4800px | 32,000 | ~30-60 min |

---

## 🐛 Troubleshooting

### COLMAP Build Takes Long

- First build compiles COLMAP from source (~10-15 min)
- Subsequent builds use cached layers
- ARM64 architecture requires source compile

### Video Upload Fails

- Check file size (large files may timeout)
- Verify format is supported by FFmpeg
- Try lower FPS setting

### COLMAP Processing Stuck

- Check logs: `docker logs colmap-processor`
- Verify GPU access: `docker exec colmap-processor nvidia-smi`
- Try lower quality setting

### Training Fails

- Ensure COLMAP completed successfully
- Check GPU is available: `curl http://localhost:8000/health`
- Verify dataset has images and sparse/ folder

---

## 📝 API Usage Examples

### Upload Video via API

```bash
curl -X POST http://localhost:8003/video/extract \
  -F "file=@video.mp4" \
  -F "fps=2.0" \
  -F "dataset_name=my_scene"
```

### Process with COLMAP

```bash
curl -X POST http://localhost:8003/process \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "my_scene",
    "quality": "medium",
    "matcher": "exhaustive"
  }'
```

### Check COLMAP Status

```bash
curl http://localhost:8003/jobs/colmap_my_scene_123456
```

### Download Processed Dataset

```bash
curl http://localhost:8003/download/colmap_my_scene_123456 -o dataset_colmap.zip
```

---

## 🎨 Workflow Page Features

### Real-time Updates

- Progress bars update every 2-3 seconds
- Status messages show current operation
- Color-coded status (processing/complete/error)

### Drag & Drop

- Drop files directly onto upload zone
- Visual feedback on hover and drop
- File info displayed before upload

### Smart Defaults

- FPS: 2.0 (good for most videos)
- Quality: Medium (balanced)
- Matcher: Exhaustive (best results)
- Steps: 30,000 (production quality)

### Cross-Service Integration

- Automatically transfers data between services
- Downloads COLMAP output for training
- Provides download links for results

---

## 🚀 Performance

### With GPU (NVIDIA GB10)

| Operation | Time |
|-----------|------|
| Video extraction (2 FPS, 2 min video) | ~1-2 min |
| COLMAP Medium (50 images) | ~5-15 min |
| Training 30K steps | ~15-30 min |
| **Total** | **~25-50 min** |

### Speedup vs CPU

- COLMAP: ~3-5x faster with GPU
- Training: ~10x faster with GPU

---

## 📦 Complete Service Stack

```yaml
Services:
  - colmap-service:8003     # New! COLMAP processing
  - training-service:8000   # GPU training
  - rendering-service:8001  # Model management
  - usd-pipeline:8002       # USD conversion
  - streaming-server:8080   # Workflow UI & viewer

All with:
  ✅ GPU acceleration
  ✅ Health checks
  ✅ Auto-restart
  ✅ Shared volumes
  ✅ Swagger APIs
```

---

## ✅ Success!

You now have a complete, production-ready 3D reconstruction pipeline!

**Start using it**: http://localhost:8080/workflow 🎉
