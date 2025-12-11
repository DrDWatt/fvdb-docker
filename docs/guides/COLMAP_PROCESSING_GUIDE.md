# 🎯 COLMAP Processing in Training Pipeline

## Current Implementation: Manual COLMAP Required ⚠️

The training service **does NOT automatically run COLMAP**. It expects datasets to already contain COLMAP structure-from-motion data.

---

## How It Works Now

### 1. Dataset Upload
When you upload images or extract video frames, the service stores them but **does not process them with COLMAP**.

### 2. COLMAP Detection
During training, the service searches for existing COLMAP data:

```python
def find_colmap_dir(dataset_path: Path):
    """Searches for COLMAP files in dataset"""
    # Looks for: cameras.bin, cameras.txt, images.bin, images.txt
    # Searches in: sparse/0, sparse, colmap, or root directory
```

### 3. Training Starts
If COLMAP data is found, training proceeds. If not, training fails with:
```
"Could not find COLMAP data in dataset"
```

---

## Required COLMAP Structure

Your uploaded dataset must have this structure:

```
dataset_name/
├── images/              # Your photos
│   ├── IMG_0001.jpg
│   ├── IMG_0002.jpg
│   └── ...
└── sparse/
    └── 0/               # COLMAP reconstruction
        ├── cameras.bin  # Camera parameters
        ├── images.bin   # Image poses
        └── points3D.bin # 3D points
```

**OR** text format:
```
dataset_name/
├── images/
└── sparse/
    └── 0/
        ├── cameras.txt
        ├── images.txt
        └── points3D.txt
```

---

## Current Workflow Options

### Option A: Pre-Process with COLMAP (Host System)

1. **Install COLMAP** on your DGX Spark:
```bash
# Ubuntu/Debian
sudo apt install colmap

# Or build from source for ARM64
git clone https://github.com/colmap/colmap.git
cd colmap
mkdir build && cd build
cmake .. -GNinja
ninja
sudo ninja install
```

2. **Process Your Images**:
```bash
# Create workspace
mkdir -p workspace/images
cp your_photos/*.jpg workspace/images/

# Run automatic reconstruction
colmap automatic_reconstructor \
  --workspace_path workspace \
  --image_path workspace/images \
  --sparse 1

# This creates workspace/sparse/0/ with COLMAP data
```

3. **Zip and Upload**:
```bash
cd workspace
zip -r my_dataset.zip images/ sparse/
curl -X POST http://localhost:8000/datasets/upload \
  -F "file=@my_dataset.zip" \
  -F "dataset_name=my_scene"
```

### Option B: Use Pre-Processed Datasets

Download datasets that already have COLMAP data (e.g., from Mip-NeRF 360, Tanks & Temples).

### Option C: Use Reality Capture Software

Tools like:
- **Reality Capture** (commercial)
- **Metashape** (commercial)
- **Meshroom** (free, open-source)

These can export in COLMAP format.

---

## What Should Be Added: Automatic COLMAP Processing 🎯

### Ideal Future Implementation

Add a COLMAP processing endpoint:

```python
@app.post("/datasets/{dataset_id}/process")
async def process_with_colmap(
    dataset_id: str,
    feature_extractor: str = "sift",  # or "superpoint"
    matcher: str = "exhaustive",      # or "sequential", "vocab_tree"
    quality: str = "high"             # "low", "medium", "high", "extreme"
):
    """
    Run COLMAP structure-from-motion on uploaded images
    """
    # 1. Feature extraction
    # 2. Feature matching
    # 3. Sparse reconstruction
    # 4. Bundle adjustment
    # Returns: status and reconstruction info
```

### Implementation Steps Needed

1. **Add COLMAP to Docker container**:
```dockerfile
RUN apt-get update && apt-get install -y \
    colmap \
    # Or build from source for ARM64
```

2. **Create COLMAP wrapper**:
```python
def run_colmap_reconstruction(image_dir: Path, output_dir: Path):
    """Run COLMAP automatic reconstruction"""
    import subprocess
    
    # Feature extraction
    subprocess.run([
        "colmap", "feature_extractor",
        "--database_path", f"{output_dir}/database.db",
        "--image_path", str(image_dir)
    ])
    
    # Feature matching
    subprocess.run([
        "colmap", "exhaustive_matcher",
        "--database_path", f"{output_dir}/database.db"
    ])
    
    # Sparse reconstruction
    subprocess.run([
        "colmap", "mapper",
        "--database_path", f"{output_dir}/database.db",
        "--image_path", str(image_dir),
        "--output_path", f"{output_dir}/sparse"
    ])
```

3. **Integrate into upload/extract workflows**:
```python
@app.post("/datasets/upload")
async def upload_dataset(
    file: UploadFile,
    auto_process: bool = True  # NEW parameter
):
    # ... extract files ...
    
    if auto_process and not has_colmap_data(dataset_path):
        # Automatically run COLMAP
        await run_colmap_reconstruction(images_dir, dataset_path)
```

---

## Temporary Workaround Script

Create `/home/dwatkins3/fvdb-docker/scripts/process_colmap.sh`:

```bash
#!/bin/bash
# Process images with COLMAP before uploading

DATASET_PATH=$1
IMAGE_DIR="${DATASET_PATH}/images"
SPARSE_DIR="${DATASET_PATH}/sparse/0"

mkdir -p "$SPARSE_DIR"

# Feature extraction
colmap feature_extractor \
  --database_path "${DATASET_PATH}/database.db" \
  --image_path "$IMAGE_DIR" \
  --ImageReader.camera_model SIMPLE_PINHOLE \
  --SiftExtraction.use_gpu 1

# Exhaustive matching (for < 100 images)
colmap exhaustive_matcher \
  --database_path "${DATASET_PATH}/database.db" \
  --SiftMatching.use_gpu 1

# Sparse reconstruction
colmap mapper \
  --database_path "${DATASET_PATH}/database.db" \
  --image_path "$IMAGE_DIR" \
  --output_path "${DATASET_PATH}/sparse"

echo "✅ COLMAP processing complete!"
echo "📦 Ready to zip and upload:"
echo "   cd ${DATASET_PATH} && zip -r dataset.zip images/ sparse/"
```

Usage:
```bash
chmod +x scripts/process_colmap.sh
./scripts/process_colmap.sh /path/to/my_images_folder
```

---

## Detection Logic in Training Service

Current code from `training_service.py`:

```python
def find_colmap_dir(dataset_path: Path) -> Optional[Path]:
    """Find COLMAP directory in dataset - searches recursively"""
    
    # Try common COLMAP directory patterns
    patterns = ["sparse/0", "sparse", "colmap", "."]
    for pattern in patterns:
        colmap_path = dataset_path / pattern
        if colmap_path.exists():
            # Check for COLMAP files
            if (colmap_path / "cameras.bin").exists() or \
               (colmap_path / "cameras.txt").exists():
                return colmap_path
    
    # If not found, recursively search (up to 3 levels deep)
    for root, dirs, files in os.walk(dataset_path):
        if ("cameras.bin" in files or "cameras.txt" in files) and \
           ("images.bin" in files or "images.txt" in files):
            return Path(root)
    
    return None
```

---

## Quick Reference

### Does Training Service Run COLMAP?
❌ **No** - It only detects existing COLMAP data

### What Happens with Video Frame Extraction?
✅ Extracts frames to `images/` folder  
❌ Does NOT run COLMAP on extracted frames  
⚠️ Training will fail unless you add COLMAP data manually

### What Happens with Photo Upload?
✅ Extracts uploaded images  
❌ Does NOT run COLMAP  
⚠️ Training will fail unless COLMAP data is included in the ZIP

### Recommended Workflow Right Now:
1. Extract frames OR prepare images locally
2. Run COLMAP locally on those images
3. Create ZIP with both `images/` and `sparse/` folders
4. Upload complete dataset
5. Train

---

## Summary

| Step | Automatic? | Manual Required? |
|------|-----------|------------------|
| **Video extraction** | ✅ Yes | - |
| **Photo upload** | ✅ Yes | - |
| **COLMAP processing** | ❌ **No** | ✅ **Yes - Run locally** |
| **Training** | ✅ Yes (if COLMAP exists) | - |

**Key Insight**: The training service is built to work with COLMAP output, but doesn't include COLMAP itself. This is a gap that should be filled for a complete end-to-end pipeline.

---

## Action Items for Complete Pipeline

1. ✅ Already done: Video frame extraction
2. ✅ Already done: Image upload
3. ⚠️ **Missing**: Automatic COLMAP processing
4. ✅ Already done: GPU training from COLMAP data
5. ✅ Already done: PLY export
6. ✅ Already done: USD conversion

**The missing piece is #3** - automatic COLMAP processing after upload/extraction.
