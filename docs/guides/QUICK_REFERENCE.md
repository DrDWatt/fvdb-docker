# Quick Reference - DGX 3D Splat System

## �� View Current Model

**Web Browser:**
```
http://192.168.1.75:8001/viewer/e2e_demo
```

---

## 📱 Upload New iPhone Photos

```bash
curl -X POST "http://192.168.1.75:8000/workflow/complete" \
  -F "file=@my_photos.zip" \
  -F "num_steps=1000" \
  -F "output_name=my_scene"
```

---

## 🔄 Transfer Model to Rendering

After training completes:

```bash
# Get job ID from training response
JOB_ID="job_xxx"
MODEL_NAME="my_scene"

# Copy to rendering container
docker exec fvdb-training cp \
  "/app/outputs/$JOB_ID/${MODEL_NAME}.ply" \
  /app/models/${MODEL_NAME}.ply

# Upload to rendering service
docker exec fvdb-rendering curl -X POST \
  "http://localhost:8001/models/upload" \
  -F "file=@/app/models/${MODEL_NAME}.ply" \
  -F "model_id=$MODEL_NAME"
```

---

## 🌐 Access Points

| Service | URL |
|---------|-----|
| **Training API** | http://192.168.1.75:8000 |
| **Training Swagger** | http://192.168.1.75:8000/docs |
| **Rendering API** | http://192.168.1.75:8001 |
| **Rendering Swagger** | http://192.168.1.75:8001/api |
| **Web Viewer** | http://192.168.1.75:8001/viewer/{model_id} |

---

## 🔍 Check Status

**Services:**
```bash
docker ps | grep fvdb
curl http://localhost:8000/health
curl http://localhost:8001/health
```

**Training Job:**
```bash
curl http://localhost:8000/jobs/{job_id}
```

**Loaded Models:**
```bash
curl http://localhost:8001/models
```

---

## 📊 Current System

- Training: ✅ Running on port 8000
- Rendering: ✅ Running on port 8001  
- Model loaded: `e2e_demo` (173,350 Gaussians)
- GPU: NVIDIA GB100 (cuda:0)
- Network: 192.168.1.75

---

**Last Updated:** November 5, 2025
