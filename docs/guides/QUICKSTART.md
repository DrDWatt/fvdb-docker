# fVDB Docker Quick Start Guide

Get up and running with fVDB Reality Capture Docker services in minutes!

## ⚡ 3-Minute Setup

### Step 1: Build Images (~15-30 minutes first time)

```bash
cd ~/fvdb-docker
./build.sh
```

### Step 2: Start Services

```bash
docker compose up -d
```

### Step 3: Verify

```bash
./test.sh
```

### Step 4: Access

- **Training API:** http://localhost:8000 (Swagger UI with tutorials)
- **Rendering:** http://localhost:8001

---

## 🎯 First Workflow

### Train Your First Gaussian Splat

1. **Prepare COLMAP Dataset:**
   ```
   your_dataset.zip
   ├── sparse/0/
   │   ├── cameras.bin
   │   ├── images.bin
   │   └── points3D.bin
   └── images/
       ├── img001.jpg
       └── ...
   ```

2. **Upload Dataset:**
   ```bash
   curl -X POST "http://localhost:8000/datasets/upload" \
     -F "file=@your_dataset.zip" \
     -F "dataset_name=my_scene"
   ```

3. **Start Training:**
   ```bash
   curl -X POST "http://localhost:8000/train" \
     -H "Content-Type: application/json" \
     -d '{
       "dataset_id": "my_scene",
       "num_training_steps": 30000
     }'
   ```
   
   **Response:**
   ```json
   {
     "job_id": "job_20251104_123456",
     "status": "queued"
   }
   ```

4. **Monitor Progress:**
   ```bash
   curl http://localhost:8000/jobs/job_20251104_123456
   ```

5. **Download Model:**
   ```bash
   curl -O http://localhost:8000/outputs/job_20251104_123456/model.ply
   ```

6. **Visualize:**
   ```bash
   curl -X POST "http://localhost:8001/models/upload" \
     -F "file=@model.ply" \
     -F "model_id=my_scene"
   
   # Open in browser
   open http://localhost:8001/viewer/my_scene
   ```

---

## 📚 Use Tutorials

Both services link directly to fVDB tutorials!

### From Swagger UI:
1. Open http://localhost:8000
2. Click `/tutorials` endpoint
3. Click "Try it out" → "Execute"
4. See all tutorial links with descriptions

### From Command Line:
```bash
curl http://localhost:8000/tutorials | python3 -m json.tool
```

**Response:**
```json
{
  "tutorials": [
    {
      "title": "Gaussian Splat Radiance Field Reconstruction",
      "url": "https://fvdb.ai/reality-capture/tutorials/radiance_field_and_mesh_reconstruction.html",
      "description": "..."
    },
    ...
  ]
}
```

---

## 🎨 Example Scripts

### Automated Training:
```bash
./examples/upload_and_train.sh dataset.zip my_scene 30000
```

### Monitor Job:
```bash
./examples/monitor_training.sh job_20251104_123456
```

---

## 🔧 Common Commands

### Logs:
```bash
docker compose logs -f training
docker compose logs -f rendering
```

### Restart:
```bash
docker compose restart
```

### Stop:
```bash
docker compose down
```

### Rebuild:
```bash
docker compose down
./build.sh
docker compose up -d
```

---

## 🐛 Troubleshooting

### Services won't start?
```bash
docker compose logs
# Check for port conflicts, GPU access
```

### GPU not detected?
```bash
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi
```

### Can't access web UI?
```bash
# Check if services are running
docker compose ps

# Check ports
netstat -an | grep -E "8000|8001"
```

---

## 📊 What's Running?

```
┌─────────────────────┐
│   Port 8000         │  Training Service
│   - Upload datasets │  - FastAPI + Swagger
│   - Train models    │  - Background jobs
│   - Monitor jobs    │  - GPU training
│   - Download PLY    │
└─────────────────────┘

┌─────────────────────┐
│   Port 8001         │  Rendering Service
│   - Upload models   │  - FastAPI + Swagger
│   - Render images   │  - Web viewer
│   - View in browser │  - Depth maps
└─────────────────────┘

     Shared Volume: /app/models
```

---

## 🎓 Next Steps

1. **Read Full Docs:** `README.md`
2. **Follow Tutorials:** `TUTORIALS.md`
3. **Check Project Summary:** `PROJECT_SUMMARY.md`
4. **Try Examples:** `examples/*.sh`

---

## ✅ Success Indicators

When everything is working:

- ✅ `./test.sh` passes all tests
- ✅ http://localhost:8000 shows Swagger UI
- ✅ http://localhost:8001 shows service home
- ✅ `docker compose ps` shows both services "Up"
- ✅ GPU visible in containers: `docker compose exec training nvidia-smi`

---

**Ready to create amazing Gaussian Splats!** 🚀

For help: Check `README.md` or tutorial links in Swagger UI
