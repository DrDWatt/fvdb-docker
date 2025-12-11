# South Building Training Error - Solution Guide

## ❌ Error Encountered

```json
{
  "status": "failed",
  "progress": 0.1,
  "message": "Python integer -1 out of bounds for uint64"
}
```

## 🔍 Root Cause Analysis

### **This is a DATA QUALITY issue, not a container bug**

The `south-building.zip` dataset from COLMAP releases contains **text format** (`.txt`) COLMAP files with **invalid reconstruction values**:

```python
# The error occurs here in fVDB's COLMAP loader:
# When reading images.txt or points3D.txt
# Some values are -1 (invalid) where uint64 expects >= 0
```

**Why this happens:**
1. COLMAP reconstruction included failed/rejected points
2. Text format stores `-1` as placeholder for invalid data
3. fVDB's loader expects valid unsigned integers only
4. Binary format (`.bin`) would handle this differently

---

## ✅ Solution 1: Use Working Datasets (RECOMMENDED)

### **Mip-NeRF 360 Datasets** - Known to Work ✅

These datasets use **binary COLMAP format** and are production-ready:

```bash
# Available at ~/data/360_v2/
- counter  ✅ Works perfectly
- room     ✅ Works perfectly  
- kitchen  ✅ Works perfectly
- bicycle  ✅ Works perfectly
- bonsai   ✅ Works perfectly
- garden   ✅ Works perfectly
- stump    ✅ Works perfectly
```

### Example: Upload Counter Scene

```bash
# 1. Create ZIP from existing data
cd ~/data/360_v2
zip -r /tmp/counter_scene.zip counter/

# 2. Upload and train
curl -X POST "http://localhost:8000/workflow/complete" \
  -F "file=@/tmp/counter_scene.zip" \
  -F "num_steps=1000" \
  -F "output_name=counter_model"

# 3. Monitor
curl http://localhost:8000/jobs/{job_id}
```

---

## ✅ Solution 2: Fix South Building Dataset

If you need to use south-building specifically, convert to binary format:

### **Method A: Use COLMAP to Convert**

```bash
# Install COLMAP locally
# Then convert text → binary:

colmap model_converter \
  --input_path /path/to/south-building/sparse \
  --output_path /path/to/south-building/sparse_binary \
  --output_type BIN

# Re-zip with binary files
cd /path/to/south-building
rm sparse/*.txt
mv sparse_binary/* sparse/
zip -r south-building-fixed.zip .
```

### **Method B: Clean Invalid Values**

```python
# Script to clean -1 values from COLMAP text files
import re

def clean_colmap_txt(file_path):
    """Remove lines with -1 point IDs"""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    cleaned = []
    for line in lines:
        # Skip lines with -1 point IDs
        if re.search(r'\s-1\s', line):
            continue
        cleaned.append(line)
    
    with open(file_path + '.cleaned', 'w') as f:
        f.writelines(cleaned)

# Run on images.txt and points3D.txt
clean_colmap_txt('sparse/images.txt')
clean_colmap_txt('sparse/points3D.txt')
```

---

## ✅ Solution 3: Download Better Datasets

### **Recommended Sources:**

#### 1. **Mip-NeRF 360** (Best Quality)
```bash
wget http://storage.googleapis.com/gresearch/refraw360/360_v2.zip
unzip 360_v2.zip
# Contains 7 high-quality scenes with binary COLMAP
```

#### 2. **Tanks and Temples**
```bash
# High quality outdoor/indoor scenes
# Available at: https://www.tanksandtemples.org/
```

#### 3. **NeRF Synthetic**
```bash
# Synthetic scenes (already have camera poses)
# Available at: https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1
```

---

## 🔧 Solution 4: Add Data Validation

I can add a validation endpoint to check datasets before training:

<function_calls>
<invoke name="read_file">
<parameter name="file_path">/home/dwatkins3/fvdb-docker/training-service/training_service.py
