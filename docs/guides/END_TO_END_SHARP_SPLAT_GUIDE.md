# End-to-End Sharp 3D Splat Guide

## Current System Analysis

### Training Service Configuration

**File:** `/home/dwatkins3/fvdb-docker/training-service/training_service.py`

**Current Config (Lines 167-171):**
```python
config = frc.radiance_fields.GaussianSplatReconstructionConfig(
    max_steps=num_steps
)
```

**Default Endpoint (Line 506):**
```python
num_steps: int = 30000  # Default to 30000 steps
```

### Rendering Service Configuration

**File:** `/home/dwatkins3/fvdb-docker/rendering-service/rendering_service.py`

- Auto-loads models on upload (lines 363-403)
- Auto-copies to downloads directory (lines 387-392)
- Provides viewer interface with download + SuperSplat workflow (lines 565-690)

---

## Problem Analysis

### Why Models Are Still Blurry

With 240 images, batch_size=1, and 30000 steps:

```
Steps per epoch: 240
Total epochs: 125
Refinement starts: epoch 3 (step 720)
Refinement stops: epoch 100 (step 24000)
Refine interval: 0.65 epochs (156 steps)
Total refinement operations: ~149
```

**The problem:** Refinement stops at epoch 100, but we train for 125 epochs. The last 6000 steps (20% of training) have NO refinement happening!

### fVDB Refinement System

In fVDB, "refinement" = densification (adding/removing Gaussians):
- `refine_start_epoch`: When to start adding Gaussians
- `refine_stop_epoch`: When to STOP adding Gaussians
- `refine_every_epoch`: How often to refine

**Key insight:** After `refine_stop_epoch`, no more Gaussians are added. Training continues but only optimizes existing Gaussians.

---

## Solution: Optimize Refinement for Step Count

### Option 1: Match refine_stop_epoch to Training Duration

For 30000 steps with 240 images:
- Total epochs = 125
- Set `refine_stop_epoch` close to total epochs

```python
config = frc.radiance_fields.GaussianSplatReconstructionConfig(
    max_steps=30000,
    refine_stop_epoch=120  # Almost until the end
)
```

### Option 2: More Aggressive Refinement

Refine more frequently to add more Gaussians:

```python
config = frc.radiance_fields.GaussianSplatReconstructionConfig(
    max_steps=30000,
    refine_stop_epoch=120,
    refine_every_epoch=0.5  # More frequent (vs 0.65 default)
)
```

### Option 3: Use max_epochs Instead (Recommended)

Let fVDB calculate steps automatically:

```python
config = frc.radiance_fields.GaussianSplatReconstructionConfig(
    max_epochs=150,  # Train for more epochs
    refine_stop_epoch=140  # Refine until near the end
)
```

---

## Recommended Configuration

### For Sharp, Dense Results

```python
config = frc.radiance_fields.GaussianSplatReconstructionConfig(
    max_steps=num_steps,
    refine_stop_epoch=min(200, int(num_steps / 240 * 0.95)),  # Until 95% of training
    refine_every_epoch=0.5  # More frequent refinement
)
```

This ensures:
1. ✅ Refinement continues for 95% of training
2. ✅ More frequent refinement = more Gaussians
3. ✅ Only final 5% is pure optimization

### Implementation

Edit `/home/dwatkins3/fvdb-docker/training-service/training_service.py`:

```python
# Configure training for HIGH QUALITY sharp results
# Calculate epochs for this dataset
images_per_epoch = 240  # Adjust if different dataset
total_epochs = num_steps / images_per_epoch if num_steps else 200

config = frc.radiance_fields.GaussianSplatReconstructionConfig(
    max_steps=num_steps,
    refine_stop_epoch=int(total_epochs * 0.95),  # Refine until 95% done
    refine_every_epoch=0.5  # More aggressive refinement
)
```

---

## Complete End-to-End Workflow

### 1. Upload Dataset

```bash
curl -X POST "http://localhost:8000/workflow/complete" \
  -F "file=@counter.zip" \
  -F "output_name=counter_sharp_final"
```

### 2. Monitor Training

```bash
# Check status
curl http://localhost:8000/jobs | python3 -m json.tool

# Watch logs
docker logs -f fvdb-training
```

### 3. Model Auto-Loads

When training completes:
- Model exports to PLY
- Auto-loads into rendering service
- Auto-copies to downloads directory

### 4. View in SuperSplat

**Option A: Web Browser**
1. Go to: http://localhost:8001/
2. Click "View" on your model
3. Click "Download PLY File"
4. Click "Open SuperSplat"
5. Drag PLY onto SuperSplat

**Option B: Direct Download**
```bash
# Download directly
curl http://localhost:8001/static/downloads/counter_sharp_final.ply -o model.ply

# Open SuperSplat
open https://playcanvas.com/supersplat/editor

# Drag model.ply onto SuperSplat
```

---

## Expected Results

### With Optimized Configuration

| Metric | Before | After |
|--------|--------|-------|
| Refinement epochs | 97 / 125 (78%) | 119 / 125 (95%) |
| Refine interval | 0.65 epochs | 0.5 epochs |
| Refine operations | ~149 | ~238 |
| Expected Gaussians | 1.0M | 1.5-2M |
| Quality | ★★★☆☆ | ★★★★★ |

### Signs of Success

**Sharp Model:**
- ✅ Dense point cloud (not sparse)
- ✅ Smooth surfaces (not noisy)
- ✅ 1.5-2 million Gaussians
- ✅ Clear textures and details
- ✅ No blue noise artifacts

**Blurry Model:**
- ❌ Sparse, disconnected points
- ❌ Noisy, irregular geometry
- ❌ < 1 million Gaussians
- ❌ Fuzzy, unclear surfaces
- ❌ Blue noise everywhere

---

## Testing Plan

### Step 1: Update Configuration

```bash
cd /home/dwatkins3/fvdb-docker

# Edit training_service.py with new config
```

### Step 2: Rebuild and Restart

```bash
docker compose -f docker-compose.host.yml build training
docker compose -f docker-compose.host.yml up -d training
```

### Step 3: Train Test Model

```bash
cd ~/data/360_v2/counter
curl -X POST "http://localhost:8000/workflow/complete" \
  -F "file=@counter.zip" \
  -F "output_name=counter_optimized"
```

### Step 4: Wait and Verify

- Training time: ~1.5 hours
- Check Gaussian count in metadata
- Download and view in SuperSplat
- Compare with previous blurry models

---

## Debugging Checklist

If model is still blurry:

- [ ] Check Gaussian count (should be > 1.5M)
- [ ] Verify refine_stop_epoch in logs
- [ ] Confirm refinement happened throughout training
- [ ] Check if COLMAP reconstruction was good
- [ ] Verify all 240 images were used
- [ ] Check image quality (not too blurry/dark)
- [ ] Ensure sufficient camera coverage
- [ ] Look for error messages in logs

---

## System Diagram

```
┌─────────────────┐
│ iPhone Photos   │
└────────┬────────┘
         │
         ↓
┌─────────────────────────────────────────┐
│ Training Service (localhost:8000)       │
│                                         │
│ 1. Upload ZIP                           │
│ 2. Extract + validate COLMAP            │
│ 3. Configure training:                  │
│    • max_steps=30000                    │
│    • refine_stop_epoch=120              │
│    • refine_every_epoch=0.5             │
│ 4. Train for ~1.5 hours                 │
│ 5. Export PLY                           │
│ 6. Copy to shared volume                │
└────────┬────────────────────────────────┘
         │
         ↓
┌─────────────────────────────────────────┐
│ Rendering Service (localhost:8001)      │
│                                         │
│ 1. Auto-load PLY                        │
│ 2. Copy to /static/downloads/           │
│ 3. Provide viewer page with:            │
│    • Download button                    │
│    • SuperSplat link                    │
│    • Mobile viewing instructions        │
└────────┬────────────────────────────────┘
         │
         ↓
┌─────────────────────────────────────────┐
│ SuperSplat (Web Viewer)                 │
│                                         │
│ 1. User drags PLY file                  │
│ 2. Loads 3D model                       │
│ 3. Interactive viewing                  │
│ 4. Verify quality                       │
└─────────────────────────────────────────┘
```

---

## Key Files

1. **Training Service:** `/home/dwatkins3/fvdb-docker/training-service/training_service.py`
   - Lines 167-171: Configuration
   - Lines 500-570: Complete workflow endpoint

2. **Rendering Service:** `/home/dwatkins3/fvdb-docker/rendering-service/rendering_service.py`
   - Lines 363-403: Model upload with auto-download
   - Lines 565-690: Viewer page with SuperSplat integration

3. **Docker Compose:** `/home/dwatkins3/fvdb-docker/docker-compose.host.yml`
   - Training service with 8GB shared memory
   - Rendering service with shared volume

---

## Summary

**Current Issue:** 
- Refinement stops too early (epoch 100 / 125)
- Not enough Gaussians added
- Result: Sparse, blurry model

**Solution:**
- Extend `refine_stop_epoch` to 95% of training
- Increase refinement frequency (`refine_every_epoch=0.5`)
- This adds ~60% more refinement operations
- Result: Dense, sharp model

**Next Steps:**
1. Update configuration in training_service.py
2. Rebuild Docker container
3. Train new model
4. Verify >1.5M Gaussians
5. View in SuperSplat - should be SHARP!
