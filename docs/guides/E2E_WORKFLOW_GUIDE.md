# End-to-End Workflow Guide

## 🎯 Complete Pipeline: Upload → Train → Render

The fVDB Docker services now support a streamlined end-to-end workflow that automates the complete pipeline from dataset upload to trained model export.

---

## 📋 Quick Start

### Option 1: Upload ZIP File

```bash
curl -X POST "http://localhost:8000/workflow/complete" \
  -F "file=@/path/to/dataset.zip" \
  -F "num_steps=1000" \
  -F "output_name=my_model"
```

### Option 2: Upload from URL

```bash
curl -X POST "http://localhost:8000/workflow/complete" \
  -F "url=https://example.com/dataset.zip" \
  -F "num_steps=1000" \
  -F "output_name=my_model"
```

---

## 🔄 Workflow Steps

The `/workflow/complete` endpoint automatically handles:

1. **📤 Upload**: Accepts dataset as ZIP file or URL
2. **✅ Validate**: Verifies COLMAP data structure  
3. **🚀 Train**: Starts Gaussian Splat training
4. **💾 Export**: Saves PLY model to shared volume
5. **📊 Monitor**: Provides job_id for progress tracking

---

## 📊 Monitoring Progress

After starting a workflow, you'll receive a `job_id`:

```json
{
  "job_id": "job_20251105_050443_395633",
  "dataset_id": "workflow_20251105_050443",
  "output_name": "my_model",
  "num_steps": 1000,
  "status": "queued",
  "endpoints": {
    "status": "/jobs/job_20251105_050443_395633",
    "outputs": "/outputs/job_20251105_050443_395633",
    "rendering_service": "http://localhost:8001/api"
  }
}
```

### Check Status

```bash
# Real-time status
curl http://localhost:8000/jobs/{job_id} | jq

# Monitor with loop
while true; do
  curl -s http://localhost:8000/jobs/{job_id} | jq '.status, .progress, .message'
  sleep 10
done
```

### Status Progression

- `queued` → Job queued for processing
- `loading_data` → Loading COLMAP scene
- `training` → Training Gaussian Splat model  
- `exporting` → Exporting PLY file
- `completed` → ✅ Ready for rendering
- `failed` → ❌ Check error message

---

## 📁 Dataset Requirements

Your ZIP file must contain:

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

**COLMAP formats supported:**
- Binary (`.bin`) - Recommended for large datasets
- Text (`.txt`) - Human-readable format

---

## 🎨 Using Swagger UI

### Training Service (Port 8000)

Access: **http://localhost:8000**

**Key Endpoints:**
- `POST /workflow/complete` - 🚀 **Complete end-to-end pipeline**
- `POST /datasets/upload` - Upload dataset only
- `POST /train` - Train existing dataset
- `GET /jobs/{job_id}` - Check job status
- `GET /jobs` - List all jobs
- `GET /outputs/{job_id}` - List output files
- `GET /outputs/{job_id}/{filename}` - Download file

### Rendering Service (Port 8001)

Access: **http://localhost:8001/api**

**Key Endpoints:**
- `POST /models/upload` - Upload trained PLY model
- `GET /models` - List loaded models
- `POST /render` - Render from camera position
- `GET /viewer` - Interactive 3D viewer

---

## 💡 Usage Examples

### Example 1: Quick Test (100 steps)

```bash
curl -X POST "http://localhost:8000/workflow/complete" \
  -F "file=@my_dataset.zip" \
  -F "num_steps=100" \
  -F "output_name=quick_test"
```

### Example 2: High Quality (5000 steps)

```bash
curl -X POST "http://localhost:8000/workflow/complete" \
  -F "file=@my_dataset.zip" \
  -F "num_steps=5000" \
  -F "output_name=high_quality_model"
```

### Example 3: From URL

```bash
curl -X POST "http://localhost:8000/workflow/complete" \
  -F "url=https://dl.fbaipublicfiles.com/nsvf/dataset/Synthetic_NeRF/Mic.zip" \
  -F "num_steps=2000" \
  -F "dataset_name=nerf_mic" \
  -F "output_name=mic_gs"
```

### Example 4: Custom Dataset Name

```bash
curl -X POST "http://localhost:8000/workflow/complete" \
  -F "file=@room_scan.zip" \
  -F "dataset_name=living_room" \
  -F "num_steps=3000" \
  -F "output_name=room_final"
```

---

## 📥 Downloading Results

After training completes:

```bash
# List output files
curl http://localhost:8000/outputs/{job_id} | jq

# Download PLY model
curl -O http://localhost:8000/outputs/{job_id}/model.ply

# Download metadata
curl http://localhost:8000/outputs/{job_id}/metadata.json | jq
```

---

## 🔗 Integration with Rendering Service

Trained models are automatically saved to the shared volume at `/app/models`, making them accessible to both services.

```bash
# After training completes, load model in rendering service
curl -X POST "http://localhost:8001/models/upload" \
  -F "file=@/app/models/{job_id}/model.ply"

# Or access via web viewer
open http://localhost:8001/viewer
```

---

## 🐛 Troubleshooting

### Job Status "failed"

Check the error message:

```bash
curl http://localhost:8000/jobs/{job_id} | jq '.message'
```

**Common Issues:**

| Error | Solution |
|-------|----------|
| "No COLMAP data found" | Ensure ZIP contains `sparse/0/` directory |
| "SSL certificate error" | Images URL inaccessible, use local dataset |
| "CUDA out of memory" | Reduce `num_steps` or use smaller dataset |
| "Module not found" | Container needs rebuild |

### Check Service Health

```bash
# Training service
curl http://localhost:8000/health | jq

# Rendering service  
curl http://localhost:8001/health | jq
```

### View Container Logs

```bash
# Training service logs
docker logs fvdb-training --tail=50

# Rendering service logs
docker logs fvdb-rendering --tail=50
```

---

## 🎓 Tutorial Links

Access official fVDB tutorials via:

```bash
curl http://localhost:8000/tutorials | jq
```

**Available Tutorials:**
- [Gaussian Splat Radiance Field Reconstruction](https://fvdb.ai/reality-capture/tutorials/radiance_field_and_mesh_reconstruction.html)
- [FRGS Tutorial](https://fvdb.ai/reality-capture/tutorials/frgs.html)
- [fVDB Documentation](https://fvdb.ai/)

---

## ⚙️ Configuration

### Docker Compose Configuration

Location: `docker-compose.host.yml`

**Key Settings:**
- **Volumes**: Shared models at `/app/models`
- **Network**: Both services on `fvdb-network`
- **GPU**: NVIDIA GPU access enabled
- **Ports**: 8000 (training), 8001 (rendering)

### Environment Variables

Training Service:
```yaml
environment:
  - PYTHONPATH=/opt/fvdb-reality-capture:/opt/conda/lib/python3.12/site-packages
  - LD_LIBRARY_PATH=/opt/conda/lib:/usr/local/cuda/lib64
```

---

## 🚀 Performance Tips

1. **Training Steps**: Start with 100-500 for testing, use 2000-5000 for production
2. **Dataset Size**: Smaller datasets (10-50 images) train faster
3. **GPU Memory**: Monitor with `nvidia-smi` during training
4. **Concurrent Jobs**: Service processes one job at a time

---

## 📊 API Response Reference

### Successful Workflow Start

```json
{
  "job_id": "job_20251105_050443_395633",
  "dataset_id": "my_dataset",
  "output_name": "my_model",
  "num_steps": 1000,
  "status": "queued",
  "message": "End-to-end workflow started...",
  "endpoints": {
    "status": "/jobs/job_20251105_050443_395633",
    "outputs": "/outputs/job_20251105_050443_395633",
    "rendering_service": "http://localhost:8001/api"
  }
}
```

### Job Status Response

```json
{
  "job_id": "job_20251105_050443_395633",
  "status": "training",
  "progress": 0.45,
  "message": "Training 1000 steps...",
  "created_at": "2025-11-05T05:04:43.395652",
  "dataset_id": "my_dataset",
  "num_training_steps": 1000
}
```

---

## 🔐 Security Notes

- Services run on localhost by default
- For production: Add authentication, use HTTPS
- Dataset uploads limited by server disk space
- Consider rate limiting for public deployments

---

## 📝 Next Steps

1. ✅ Upload your COLMAP dataset
2. ✅ Start end-to-end workflow  
3. ✅ Monitor training progress
4. ✅ Download trained model
5. ✅ Visualize in rendering service

**Happy 3D Reconstruction! 🎉**
