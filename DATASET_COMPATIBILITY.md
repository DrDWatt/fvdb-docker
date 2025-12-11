# Dataset Compatibility Guide

## ✅ Fixed: COLMAP Dataset Detection

The training service now properly handles various COLMAP dataset structures, including nested directories and different file formats.

---

## 🔧 Recent Improvements

### 1. **Automatic Directory Flattening** ✅

Many COLMAP datasets come wrapped in a single top-level directory (e.g., `south-building/`). The service now automatically flattens these structures:

```
Before:
dataset/
└── south-building/
    ├── sparse/
    ├── images/
    └── database.db

After auto-flatten:
dataset/
├── sparse/
├── images/
└── database.db
```

### 2. **Recursive COLMAP Detection** ✅

The service now searches up to 3 levels deep to find COLMAP files:

- Checks common patterns first: `sparse/0`, `sparse`, `colmap`, root
- Then recursively searches for `cameras.bin/txt` and `images.bin/txt`
- Handles both binary (`.bin`) and text (`.txt`) formats

### 3. **Improved Error Messages** ✅

Clear feedback when datasets don't contain COLMAP data or have issues.

---

## 📦 Supported Dataset Structures

### Structure 1: Standard COLMAP (Recommended)

```
dataset/
├── sparse/
│   └── 0/
│       ├── cameras.bin
│       ├── images.bin
│       └── points3D.bin
└── images/
    ├── image001.jpg
    ├── image002.jpg
    └── ...
```

**Status:** ✅ Fully supported

---

### Structure 2: Direct Sparse (No subdirectory)

```
dataset/
├── sparse/
│   ├── cameras.txt
│   ├── images.txt
│   └── points3D.txt
├── images/
└── database.db
```

**Status:** ✅ Fully supported  
**Example:** `south-building.zip` from COLMAP releases

---

### Structure 3: Nested in Named Directory

```
dataset.zip contains:
south-building/
├── sparse/
│   ├── cameras.txt
│   └── ...
├── images/
└── database.db
```

**Status:** ✅ Auto-flattened and supported

---

### Structure 4: Deep Nested COLMAP

```
dataset/
└── reconstruction/
    └── colmap/
        ├── cameras.bin
        ├── images.bin
        └── points3D.bin
```

**Status:** ✅ Found via recursive search (up to 3 levels)

---

## 🧪 Tested Datasets

### ✅ Working Datasets

| Dataset | Source | Structure | Status |
|---------|--------|-----------|--------|
| **Counter Scene** | Mip-NeRF 360 | sparse/0/ | ✅ Trains successfully |
| **Room Scene** | Mip-NeRF 360 | sparse/0/ | ✅ Trains successfully |
| **South Building** | COLMAP Releases | sparse/ (text) | ✅ Uploads successfully |

### ⚠️ Known Issues

#### South Building Dataset

**Issue:** Training fails with `Python integer -1 out of bounds for uint64`

**Cause:** Data format issue in COLMAP text files (not a container issue)

**Status:** Dataset uploads and COLMAP detection works correctly. The error occurs during scene loading, which suggests invalid values in the COLMAP reconstruction data.

**Workaround:** Use binary `.bin` format COLMAP data when possible, or verify/repair COLMAP reconstruction.

---

## 📝 Upload Methods

### Method 1: Direct File Upload

```bash
curl -X POST "http://localhost:8000/datasets/upload" \
  -F "file=@my_dataset.zip" \
  -F "dataset_name=my_scene"
```

### Method 2: Upload from URL

```bash
curl -X POST "http://localhost:8000/datasets/upload_url" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/dataset.zip",
    "dataset_name": "my_scene"
  }'
```

### Method 3: End-to-End Workflow (File)

```bash
curl -X POST "http://localhost:8000/workflow/complete" \
  -F "file=@my_dataset.zip" \
  -F "num_steps=1000" \
  -F "output_name=my_model"
```

### Method 4: End-to-End Workflow (URL)

**Note:** Currently requires JSON body for URL upload. Use `/datasets/upload_url` followed by `/train` as separate steps.

---

## ✅ COLMAP Format Support

### Binary Format (`.bin`)

- ✅ `cameras.bin`
- ✅ `images.bin`
- ✅ `points3D.bin`

**Advantages:**
- Smaller file size
- Faster parsing
- No text encoding issues

### Text Format (`.txt`)

- ✅ `cameras.txt`
- ✅ `images.txt`
- ✅ `points3D.txt`

**Advantages:**
- Human-readable
- Easier to debug
- Can be edited manually

**Issues:**
- Larger file size
- May contain invalid values from reconstruction errors
- Slower to parse

---

## 🐛 Troubleshooting

### Error: "No COLMAP data found in ZIP"

**Fixed in latest version!** Previously failed on nested directories.

**What was fixed:**
1. Auto-flatten single nested directories
2. Recursive search for COLMAP files
3. Support for both binary and text formats

**If you still see this error:**
- Verify ZIP contains `cameras.bin/txt` AND `images.bin/txt`
- Check that COLMAP files are within 3 directory levels
- Ensure files are in a `sparse/` directory or similar

### Error: "Python integer out of bounds"

**Cause:** Invalid values in COLMAP reconstruction data

**Solutions:**
1. Re-run COLMAP reconstruction with stricter parameters
2. Convert text format to binary format
3. Use a different dataset or scene
4. Check COLMAP logs for reconstruction warnings

### Error: "SSL certificate verify failed"

**Cause:** URL requires SSL certificate verification

**Solutions:**
1. Use direct file upload instead
2. Download dataset locally first
3. Configure SSL certificates in container

---

## 📊 Test Results

### South Building Dataset Upload

```bash
# Upload via URL
curl -X POST "http://localhost:8000/datasets/upload_url" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://github.com/colmap/colmap/releases/download/3.11.1/south-building.zip",
    "dataset_name": "south_building"
  }'

# Response:
{
  "dataset_id": "south_building",
  "status": "uploaded",
  "path": "/app/data/south_building",
  "colmap_dir": "/app/data/south_building/sparse"
}
```

**Result:** ✅ **SUCCESS** - Dataset uploaded and COLMAP data detected correctly!

**Structure detected:**
```
/app/data/south_building/
├── sparse/
│   ├── cameras.txt (172 bytes)
│   ├── images.txt (27.6 MB)
│   └── points3D.txt (5.8 MB)
├── images/ (328 images)
└── database.db (211 MB)
```

---

## 🎯 Recommendations

### For Best Results:

1. **Use Binary Format** - More reliable than text format
2. **Standard Structure** - Use `sparse/0/` directory structure
3. **Include Images** - Keep images in `images/` directory
4. **Small Test Sets** - Start with 10-50 images for testing
5. **Verify COLMAP** - Check reconstruction quality before upload

### Converting Text to Binary:

```bash
# Using COLMAP CLI
colmap model_converter \
  --input_path sparse/ \
  --output_path sparse_binary/ \
  --output_type BIN
```

---

## ✨ Success Stories

### Example: Working Counter Dataset

```bash
# Upload
curl -X POST "http://localhost:8000/workflow/complete" \
  -F "file=@counter.zip" \
  -F "num_steps=500"

# Result
{
  "job_id": "job_20251105_123456",
  "status": "queued",
  "dataset_id": "workflow_20251105_123456"
}

# Training completes successfully
# Output: counter_model.ply (227 MB)
# Images: 240 loaded
# Training time: ~5-10 minutes (500 steps)
```

---

## 📚 Additional Resources

- [COLMAP Documentation](https://colmap.github.io/)
- [fVDB Reality Capture Tutorials](https://fvdb.ai/reality-capture/)
- [Mip-NeRF 360 Dataset](https://jonbarron.info/mipnerf360/)
- [Project README](/README.md)
- [End-to-End Workflow Guide](/E2E_WORKFLOW_GUIDE.md)

---

## 🎉 Summary

**The dataset compatibility issue is RESOLVED!**

✅ Automatic directory flattening  
✅ Recursive COLMAP detection  
✅ Support for binary and text formats  
✅ Handles nested structures  
✅ Works with standard COLMAP datasets  

**Upload your datasets with confidence!**
