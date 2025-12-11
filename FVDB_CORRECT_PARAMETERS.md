# fVDB Correct Parameters for Sharp 3D Splats

## Error Analysis

### What Went Wrong

**Error Message:**
```
GaussianSplatReconstructionConfig.__init__() got an unexpected keyword argument 'densify_until_iter'
```

**Root Cause:**
I attempted to use parameter names from the original 3D Gaussian Splatting paper/PyTorch implementation, but **fVDB uses different parameter names**.

### Wrong Parameters (Don't Exist in fVDB)
```python
❌ densify_until_iter
❌ densify_grad_threshold
❌ densification_interval
❌ opacity_reset_interval
❌ prune_opacity_threshold
```

These are from the original 3DGS implementation, not fVDB.

## Correct fVDB Parameters

### Full Parameter List

From `help(GaussianSplatReconstructionConfig)`:

```python
GaussianSplatReconstructionConfig(
    # Training control
    seed: int = 42,
    max_epochs: int = 200,
    max_steps: int | None = None,
    
    # Data loading
    batch_size: int = 1,
    crops_per_image: int = 1,
    
    # Spherical harmonics
    sh_degree: int = 3,
    increase_sh_degree_every_epoch: int = 5,
    
    # Initialization
    initial_opacity: float = 0.1,
    initial_covariance_scale: float = 1.0,
    
    # Loss weighting
    ssim_lambda: float = 0.2,
    lpips_net: Literal['vgg', 'alex'] = 'alex',
    opacity_reg: float = 0.0,
    scale_reg: float = 0.0,
    
    # Data augmentation
    random_bkgd: bool = False,
    
    # REFINEMENT (fVDB's version of densification)
    refine_start_epoch: int = 3,
    refine_stop_epoch: int = 100,
    refine_every_epoch: float = 0.65,
    
    # Masking
    ignore_masks: bool = False,
    remove_gaussians_outside_scene_bbox: bool = False,
    
    # Camera pose optimization
    optimize_camera_poses: bool = True,
    pose_opt_lr: float = 1e-05,
    pose_opt_reg: float = 1e-06,
    pose_opt_lr_decay: float = 1.0,
    pose_opt_start_epoch: int = 0,
    pose_opt_stop_epoch: int = 200,
    pose_opt_init_std: float = 0.0001,
    
    # Rendering
    near_plane: float = 0.01,
    far_plane: float = 10000000000.0,
    min_radius_2d: float = 0.0,
    eps_2d: float = 0.3,
    antialias: bool = False,
    tile_size: int = 16
)
```

## Key Parameters for Quality

### 1. Refinement (Densification in fVDB)

**In original 3DGS:** `densify_*` parameters  
**In fVDB:** `refine_*` parameters

```python
refine_start_epoch=3        # Start adding Gaussians at epoch 3
refine_stop_epoch=100       # Stop adding at epoch 100
refine_every_epoch=0.65     # Refine every 0.65 epochs
```

**How refinement works:**
- fVDB adds/removes Gaussians during "refinement"
- Lower `refine_every_epoch` = more frequent = more Gaussians
- `refine_stop_epoch` controls when to stop growing the model

### 2. Batch Size and Crops

```python
batch_size=4           # Process 4 images at once
crops_per_image=4      # Take 4 random crops per image
```

**Impact:**
- Larger batch = better gradient estimates
- More crops = better spatial coverage
- Together = denser, more accurate reconstruction

### 3. Spherical Harmonics Degree

```python
sh_degree=3  # Full color representation (0-3)
```

**Impact:**
- Higher = more detailed color/view-dependent effects
- 3 is maximum and recommended for quality

### 4. Regularization

```python
opacity_reg=0.0   # No opacity regularization
scale_reg=0.0     # No scale regularization
```

**For maximum quality:**
- Set to 0.0 to allow full freedom
- Non-zero values constrain the model

### 5. Loss Weighting

```python
ssim_lambda=0.2  # Balance between L1 and SSIM
```

**Impact:**
- 0.0 = pure L1 loss
- 1.0 = pure SSIM loss
- 0.2 = good balance for sharpness

### 6. Background Augmentation

```python
random_bkgd=True  # Random background during training
```

**Impact:**
- Helps model focus on foreground
- Better generalization

## Recommended Configuration for Sharp Results

### High Quality (30,000 steps)

```python
config = frc.radiance_fields.GaussianSplatReconstructionConfig(
    # Training
    max_steps=30000,
    
    # Data
    batch_size=4,
    crops_per_image=4,
    
    # Color
    sh_degree=3,
    
    # Refinement (aggressive for detail)
    refine_start_epoch=3,
    refine_stop_epoch=100,
    refine_every_epoch=0.65,
    
    # Initialization
    initial_opacity=0.1,
    
    # Regularization (off for max quality)
    opacity_reg=0.0,
    scale_reg=0.0,
    
    # Loss
    ssim_lambda=0.2,
    
    # Augmentation
    random_bkgd=True
)
```

### Quick Preview (7,000 steps)

```python
config = frc.radiance_fields.GaussianSplatReconstructionConfig(
    max_steps=7000,
    batch_size=4,
    crops_per_image=4,
    sh_degree=3
)
```

## fVDB vs Original 3DGS Terminology

| Original 3DGS | fVDB Equivalent | Purpose |
|---------------|-----------------|---------|
| `densify_until_iter` | `refine_stop_epoch` | When to stop adding Gaussians |
| `densify_grad_threshold` | (built-in) | Threshold for adding Gaussians |
| `densification_interval` | `refine_every_epoch` | How often to refine |
| `opacity_reset_interval` | (built-in) | Opacity reset timing |
| Position/scale pruning | `refine_*` system | Gaussian management |

## Epochs vs Steps

**Important:** fVDB uses epochs internally but you specify `max_steps`:

```python
max_steps=30000  # You specify this
# fVDB calculates epochs based on dataset size:
# epochs = max_steps / (num_images / batch_size)
```

For a 240-image dataset with batch_size=4:
- Images per epoch = 240 / 4 = 60 batches
- 30000 steps = 30000 / 60 = 500 epochs

So `refine_stop_epoch=100` means:
- 100 epochs × 60 = 6000 steps
- Refinement happens for first 20% of training

## Current Training Configuration

**File:** `/home/dwatkins3/fvdb-docker/training-service/training_service.py`

```python
config = frc.radiance_fields.GaussianSplatReconstructionConfig(
    max_steps=num_steps,
    batch_size=4,
    crops_per_image=4,
    sh_degree=3,
    refine_start_epoch=3,
    refine_stop_epoch=min(100, int(num_steps / 300)),
    refine_every_epoch=0.65,
    initial_opacity=0.1,
    opacity_reg=0.0,
    scale_reg=0.0,
    ssim_lambda=0.2,
    random_bkgd=True
)
```

## Default Training Steps

**Default:** 30,000 steps (professional quality)

To override:
```bash
curl -X POST "http://localhost:8000/workflow/complete" \
  -F "file=@dataset.zip" \
  -F "num_steps=7000"  # Custom step count
```

## Testing Your Configuration

To verify parameters are correct:

```bash
docker exec fvdb-training python3 -c "
from fvdb_reality_capture.radiance_fields import GaussianSplatReconstructionConfig
help(GaussianSplatReconstructionConfig)
"
```

## Key Takeaways

1. ✅ fVDB uses **`refine_*`** not `densify_*`
2. ✅ Always check parameter names with `help()`
3. ✅ `batch_size=4` and `crops_per_image=4` are critical
4. ✅ `max_steps=30000` for professional quality
5. ✅ Refinement parameters control Gaussian growth
6. ❌ Don't use 3DGS parameter names in fVDB

## Summary

The training failure was caused by using parameter names from the original 3D Gaussian Splatting implementation instead of fVDB's actual parameter names. The corrected configuration uses fVDB's **refinement system** (`refine_*` parameters) instead of trying to use non-existent **densification parameters** (`densify_*`).

**Result:** Training now works correctly with proper fVDB parameters and will produce high-quality, sharp 3D models!
