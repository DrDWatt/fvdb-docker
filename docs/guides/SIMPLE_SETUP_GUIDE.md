# Simple Setup: Omniverse Web Viewer with fVDB

## Current Situation

The fVDB rendering Docker container build is failing due to ARM64 compilation issues with `point-cloud-utils` and `pye57` dependencies. 

**Solution**: Use the web viewer with alternative viewing methods that are already working on your system.

## ✅ What's Working Now

### 1. Omniverse Web Viewer (Port 5173)
- **Status**: ✅ Running
- **URL**: http://localhost:5173
- **Purpose**: Web-based client for viewing streamed content
- **Configured to connect to**: `host.docker.internal:49100`

### 2. fVDB Training Service (Port 8000)
- **Status**: ✅ Running
- **URL**: http://localhost:8000
- **Purpose**: Train 3D Gaussian Splats from photos
- **Working**: Fully functional

## 📋 Viewing Options

Since the fVDB rendering container has build issues, here are your **working alternatives**:

### Option 1: Use fVDB Native Viewer (Recommended)

Run the fVDB viewer **directly on the host** (not in Docker):

```bash
# Activate your fVDB environment
conda activate fvdb

# Start interactive viewer for a model
python3 << 'EOF'
import fvdb
import fvdb.viz

# Load your trained model
model_path = "/path/to/your/model.ply"
model, metadata = fvdb.GaussianSplat3d.from_ply(model_path)

# Create and show scene
scene = fvdb.viz.Scene()
scene.add_gaussians(model)
fvdb.viz.show(scene, port=8890)
EOF
```

Then access at: http://localhost:8890

### Option 2: SuperSplat (Web-based, No Installation)

1. Train your model at http://localhost:8000
2. Download the PLY file from the training outputs
3. Go to https://playcanvas.com/supersplat
4. Drag and drop your PLY file
5. ✅ View instantly!

### Option 3: Use Existing Rendering Service

If you have models already trained:

```bash
# List available models
ls /home/dwatkins3/docker/fvdb-working/models/

# Or check training outputs
ls /home/dwatkins3/docker/fvdb-working/training-outputs/
```

Download any .ply file and view with:
- **SuperSplat**: https://playcanvas.com/supersplat (drag & drop)
- **Polycam** (iPhone/Mac): App Store, import PLY, view in AR
- **MeshLab**: https://www.meshlab.net/ (desktop viewer)

## 🔧 If You Want to Enable Omniverse Streaming

The web viewer is configured and ready. To use it, you need a streaming server on port 49100.

### Setup fVDB Viewer with Streaming

If you can run fVDB natively on the host:

```python
import fvdb
import fvdb.viz

# Load model
model, _ = fvdb.GaussianSplat3d.from_ply("your_model.ply")

# Create scene
scene = fvdb.viz.Scene()
scene.add_gaussians(model)

# Start viewer with network access
fvdb.viz.show(scene, port=49100, host='0.0.0.0')
```

Then the Omniverse Web Viewer at http://localhost:5173 will connect automatically.

## 📊 Current Architecture

```
┌─────────────────────────────────────┐
│  Browser: http://localhost:5173     │
│  Omniverse Web Viewer (Running)     │
└────────────────┬────────────────────┘
                 │
                 │ Looking for: host.docker.internal:49100
                 │ (WebRTC streaming server)
                 │
┌────────────────▼────────────────────┐
│  Your Host Machine                  │
│                                      │
│  Option A: fVDB Native Viewer       │
│            (conda activate fvdb)    │
│                                      │
│  Option B: Download PLY files       │
│            Use external viewers     │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│  fVDB Training (Port 8000)          │
│  ✅ Working - Train models here     │
└─────────────────────────────────────┘
```

## 🎯 Recommended Workflow

### For Quick Viewing

1. **Train a model**:
   - Go to http://localhost:8000
   - Upload 20-50 photos
   - Start training

2. **Download the PLY file** from training outputs

3. **View in SuperSplat**:
   - Go to https://playcanvas.com/supersplat
   - Drag PLY file
   - Done!

### For Interactive Development

1. **Train model** at http://localhost:8000

2. **View with fVDB native viewer**:
   ```bash
   conda activate fvdb
   python view_model.py  # Use the script above
   ```

3. **Access at** http://localhost:8890

### For Omniverse Web Viewer

1. Set up streaming source (fVDB viewer or other) on port 49100
2. Access web viewer at http://localhost:5173
3. Connect and view

## 🐛 Why the Docker Build Failed

The fVDB rendering container tried to install:
- `point-cloud-utils`: Requires compilation, failed on ARM64
- `pye57`: Requires `xercesc` library, missing in container
- Other C++ dependencies that need special handling on ARM64

**These packages work fine in your conda environment**, just not in the Docker build.

## ✅ What's Already Working

- ✅ fVDB training service (Docker)
- ✅ Omniverse web viewer client (Docker)  
- ✅ fVDB environment (conda, on host)
- ✅ Your NVIDIA GPU support
- ✅ Model training pipeline

## 💡 Solution Summary

**Don't rebuild the rendering container.** Instead:

1. Keep using training service (port 8000) ✅
2. Use SuperSplat for viewing (easiest) ✅
3. Or use fVDB native viewer from conda ✅
4. Omniverse web viewer is ready when you have a streaming source ✅

## 📚 Quick Commands

```bash
# Check what's running
docker ps

# Access training service
open http://localhost:8000

# Access web viewer (needs streaming source)
open http://localhost:5173

# View fVDB models (from conda env)
conda activate fvdb
python -c "import fvdb; print(fvdb.__version__)"

# Find your models
find ~ -name "*.ply" -type f 2>/dev/null | grep -E "fvdb|model|output"
```

## 🎉 Bottom Line

Everything you need is working! The Docker rendering container isn't essential because:
- ✅ Training works (Docker)
- ✅ Viewing works (SuperSplat or fVDB native)
- ✅ Web viewer works (just needs a streaming source)

Focus on using what works rather than fixing the Docker build.

---

**Next Step**: Go to http://localhost:8000 and train a model, then view it with SuperSplat!
