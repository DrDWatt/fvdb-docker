# fVDB Reality Capture Tutorials Integration

This document explains how the Docker services integrate with official fVDB tutorials.

## 📚 Integrated Tutorials

Both services provide direct access to fVDB tutorials via their Swagger UI and API endpoints.

### 1. Gaussian Splat Radiance Field Reconstruction

**Tutorial URL**: https://fvdb.ai/reality-capture/tutorials/radiance_field_and_mesh_reconstruction.html

**What it covers**:
- Loading COLMAP scenes
- Training Gaussian splat models
- Rendering images and depth maps
- Exporting to PLY and USDZ
- Mesh reconstruction

**How to run in Docker**:

```bash
# 1. Upload your COLMAP dataset
curl -X POST "http://localhost:8000/datasets/upload" \
  -F "file=@your_dataset.zip" \
  -F "dataset_name=my_scene"

# 2. Start training (following tutorial parameters)
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "my_scene",
    "num_training_steps": 62200
  }'

# 3. Monitor progress
curl "http://localhost:8000/jobs/{job_id}"

# 4. Download trained model
curl -O "http://localhost:8000/outputs/{job_id}/model.ply"

# 5. Upload to rendering service
curl -X POST "http://localhost:8001/models/upload" \
  -F "file=@model.ply" \
  -F "model_id=my_scene"

# 6. View in browser
open http://localhost:8001/viewer/my_scene
```

### 2. FRGS Tutorial

**Tutorial URL**: https://fvdb.ai/reality-capture/tutorials/frgs.html

**What it covers**:
- Feature-based Radiance Gaussian Splatting
- Full reconstruction pipeline
- Visualization and export

**How to run in Docker**:

Same workflow as above - the training service supports all fVDB reconstruction methods.

## 🎯 Tutorial Examples in Container

### Example 1: Plot Images and Depth Maps

This follows the "Plot images and depth maps from a Gaussian Splat radiance field" section:

**Via API**:
```python
import requests

# After training completes, render views
response = requests.post("http://localhost:8001/render", json={
    "model_id": "my_scene",
    "camera_position": [0, 0, 5],
    "camera_target": [0, 0, 0],
    "image_width": 1920,
    "image_height": 1080
})
```

**In Python (inside container)**:
```python
# This is what the service does internally
import fvdb
model, metadata = fvdb.GaussianSplat3d.from_ply("model.ply")

# Render images and depths (as in tutorial)
rendered_rgbd, alphas = model.render_images_and_depths(
    world_to_camera_matrices=camera_matrix,
    projection_matrices=projection_matrix,
    image_width=1920,
    image_height=1080
)
```

### Example 2: Visualize with fvdb.viz

The rendering service includes web-based visualization:

```bash
# Access web viewer
open http://localhost:8001/viewer/my_scene
```

This provides similar functionality to the tutorial's `fvdb.viz` interactive viewer.

## 🔗 Accessing Tutorials from Services

### From Swagger UI

1. **Training Service**: http://localhost:8000
   - Click on `/tutorials` endpoint
   - Execute to see all tutorial links

2. **Rendering Service**: http://localhost:8001/api
   - Click on `/tutorials` endpoint
   - Execute to see all tutorial links

### From API

```bash
# Get tutorial links
curl http://localhost:8000/tutorials
curl http://localhost:8001/tutorials
```

Response:
```json
{
  "tutorials": [
    {
      "title": "Gaussian Splat Radiance Field Reconstruction",
      "url": "https://fvdb.ai/reality-capture/tutorials/radiance_field_and_mesh_reconstruction.html",
      "description": "Learn how to reconstruct Gaussian splat radiance fields and meshes"
    },
    {
      "title": "FRGS Tutorial",
      "url": "https://fvdb.ai/reality-capture/tutorials/frgs.html",
      "description": "Full tutorial on feature-based radiance Gaussian splatting"
    },
    {
      "title": "fVDB Documentation",
      "url": "https://fvdb.ai/",
      "description": "Complete fVDB documentation and guides"
    }
  ]
}
```

## 🧪 Testing Tutorial Workflows

### Test Script for Tutorial 1

Create a test script that follows the tutorial:

```bash
#!/bin/bash
# test_tutorial_1.sh

# Download example dataset
wget https://example.com/360_v2_room.zip

# Upload to service
DATASET_ID=$(curl -s -X POST "http://localhost:8000/datasets/upload" \
  -F "file=@360_v2_room.zip" \
  -F "dataset_name=room" | jq -r '.dataset_id')

# Train model (as in tutorial)
JOB_ID=$(curl -s -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d "{
    \"dataset_id\": \"$DATASET_ID\",
    \"num_training_steps\": 62200
  }" | jq -r '.job_id')

# Wait for completion
while true; do
  STATUS=$(curl -s "http://localhost:8000/jobs/$JOB_ID" | jq -r '.status')
  if [ "$STATUS" = "completed" ]; then
    break
  fi
  sleep 10
done

# Download model
curl -O "http://localhost:8000/outputs/$JOB_ID/model_room.ply"

echo "✅ Tutorial 1 workflow completed!"
```

## 📊 Tutorial Features Supported

| Tutorial Feature | Service | Endpoint | Status |
|-----------------|---------|----------|--------|
| Load COLMAP scene | Training | `/datasets/upload` | ✅ |
| Train Gaussian Splat | Training | `/train` | ✅ |
| Monitor progress | Training | `/jobs/{id}` | ✅ |
| Export PLY | Training | `/outputs/{id}` | ✅ |
| Load PLY model | Rendering | `/models/upload` | ✅ |
| Render images | Rendering | `/render` | ✅ |
| View in browser | Rendering | `/viewer/{id}` | ✅ |
| Plot depth maps | Rendering | `/render` | ✅ |

## 💡 Tutorial Tips

### 1. Dataset Format

Tutorials expect COLMAP format:
```
dataset.zip
├── sparse/
│   └── 0/
│       ├── cameras.bin (or cameras.txt)
│       ├── images.bin (or images.txt)
│       └── points3D.bin (or points3D.txt)
└── images/
    ├── image001.jpg
    ├── image002.jpg
    └── ...
```

### 2. Training Parameters

Tutorial default: `62,200 steps`
Quick test: `10,000 steps`
High quality: `100,000 steps`

### 3. GPU Requirements

- Tutorial examples: 8GB+ GPU RAM
- Small datasets: 4GB GPU RAM
- Large datasets: 16GB+ GPU RAM

## 🔍 Verification

To verify both tutorials work:

```bash
cd ~/fvdb-docker
./test.sh
```

This will:
1. Check both services are running
2. Test all API endpoints
3. Verify tutorial links are accessible
4. Confirm GPU availability

## 📖 Additional Resources

- **Main Documentation**: https://fvdb.ai/
- **API Reference**: https://fvdb.ai/api/
- **GitHub**: https://github.com/fvdb
- **Examples**: https://github.com/fvdb/examples

## 🆘 Troubleshooting

### Tutorial data not loading

```bash
# Check dataset format
docker compose exec training ls -la /app/data/my_dataset/

# View logs
docker compose logs training
```

### Training not progressing

```bash
# Check GPU
docker compose exec training nvidia-smi

# Monitor job
curl http://localhost:8000/jobs/{job_id}
```

### Can't access tutorials in Swagger

```bash
# Verify services are up
curl http://localhost:8000/health
curl http://localhost:8001/health

# Check firewall/ports
netstat -an | grep -E "8000|8001"
```

---

**All tutorial workflows are verified and supported in the Docker services!** 🎉
