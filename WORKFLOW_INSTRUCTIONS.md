# 3D Gaussian Splatting Workflow - User Guide

## System Status
All services running on localhost:
- **COLMAP Service**: http://localhost:8003 (video processing)
- **Training Service**: http://localhost:8000 (Gaussian Splat training)
- **Rendering Service**: http://localhost:8001 (model viewing)

## Step-by-Step Workflow

### 1. Upload Video
Navigate to: http://localhost:8003

**Option A - Direct Upload:**
```bash
# Via web interface - use the upload form
# Or via curl:
curl -X POST http://localhost:8003/upload \
  -F "file=@/path/to/your/video.mov" \
  -F "dataset_id=my_scene"
```

### 2. Process with COLMAP
**Recommended Settings:**
```bash
curl -X POST http://localhost:8003/video/process \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "my_scene",
    "video_filename": "your_video.mov",
    "fps": 1.0,
    "camera_model": "SIMPLE_RADIAL",
    "matcher": "exhaustive",
    "max_image_size": 2048,
    "max_num_features": 16384
  }'
```

**Monitor Progress:**
```bash
curl http://localhost:8003/jobs | jq
```

Expected time: 5-10 minutes for 2-minute video

### 3. Download COLMAP Results
```bash
# Get job ID from /jobs endpoint
curl -o colmap_output.zip http://localhost:8003/download/colmap_JOBID

# Extract to training location
unzip colmap_output.zip -d /home/dwatkins3/fvdb-docker/data/my_scene
```

### 4. Train Gaussian Splat Model
```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "my_scene",
    "num_training_steps": 30000,
    "output_name": "my_scene_model"
  }'
```

**Monitor Training:**
```bash
# Check status
curl http://localhost:8000/jobs | jq

# Watch logs
docker logs -f fvdb-training-gpu
```

Expected time: 20-30 minutes for 40-80 images

### 5. View Model
```bash
# Copy to rendering service
docker exec fvdb-training-gpu cp \
  /app/outputs/JOB_ID/my_scene_model.ply \
  /app/models/my_scene_model.ply

# View at: http://localhost:8001
```

## Troubleshooting

### Issue: "No registered images" or "Only 2-5 images"
**Cause**: Video has insufficient feature overlap
**Solution**: Re-record with slower movement, more overlap

### Issue: Training produces small file (<10MB)
**Cause**: Too few images registered by COLMAP
**Solution**: Check COLMAP results, may need to re-record

### Issue: "Dataset not found" error
**Cause**: COLMAP output not copied to training data directory
**Solution**: Ensure unzip step completed correctly

### Issue: Container memory errors
**Cause**: Shared memory limit
**Solution**: Already configured (8GB shm_size)

## Expected Results

**Good Reconstruction:**
- 40-80+ images registered by COLMAP
- 50,000-150,000 Gaussians generated
- 20-100MB PLY file size
- Detailed, viewable model in SuperSplat

**Poor Reconstruction:**
- <10 images registered
- <10,000 Gaussians
- <10MB file
- Sparse or empty model

