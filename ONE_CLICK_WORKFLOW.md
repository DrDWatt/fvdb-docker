# 🚀 ONE-CLICK WORKFLOW: Video/Photos → 3D Model

## New Combined Endpoint!

### **POST /workflow/video-to-model**
Upload video or photo ZIP and let it run completely automatically through COLMAP → Training → Model

---

## How to Use

### Option 1: Swagger UI (Easiest!)

1. Go to: **http://localhost:8003/api**

2. Find endpoint: **POST /workflow/video-to-model**

3. Click "Try it out"

4. Fill in parameters:
   - **file**: Choose your Tesla video (415MB .mov file)
   - **dataset_id**: `tesla_model_s`
   - **fps**: `1.0` (extract 1 frame per second)
   - **camera_model**: `SIMPLE_RADIAL`
   - **matcher**: `exhaustive`
   - **num_training_steps**: `30000`

5. Click **"Execute"**

6. You'll get a response with `workflow_id`

7. Monitor progress:
   - **GET /workflow/status/{workflow_id}**
   - Keep checking until `status: "completed"`

---

### Option 2: Command Line

```bash
# Upload and start complete workflow
curl -X POST http://localhost:8003/workflow/video-to-model \
  -F "file=@/path/to/tesla_video.mov" \
  -F "dataset_id=tesla_model_s" \
  -F "fps=1.0" \
  -F "camera_model=SIMPLE_RADIAL" \
  -F "matcher=exhaustive" \
  -F "num_training_steps=30000"

# Response will give you workflow_id

# Monitor progress
curl http://localhost:8003/workflow/status/WORKFLOW_ID | jq

# List all workflows
curl http://localhost:8003/workflow/list | jq
```

---

## What It Does Automatically

The endpoint handles the complete pipeline:

1. ✅ **Upload video** (1-2 min)
2. ✅ **Extract frames** at specified FPS (1 min)
3. ✅ **COLMAP feature extraction** (2-3 min)
4. ✅ **COLMAP feature matching** (3-5 min)
5. ✅ **COLMAP sparse reconstruction** (2-3 min)
6. ✅ **Start Gaussian Splat training** (25-30 min)
7. ✅ **Export trained model** (automatic)

**Total Time: ~35-45 minutes** (fire and forget!)

---

## Monitoring Progress

### Check Status:
```bash
curl http://localhost:8003/workflow/status/WORKFLOW_ID | jq
```

### Response shows:
```json
{
  "workflow_id": "workflow_20251201_151234_567890",
  "status": "training",
  "progress": 0.85,
  "current_step": "Training: 25500/30000 steps complete",
  "dataset_id": "tesla_model_s",
  "colmap_job_id": "colmap_...",
  "training_job_id": "job_...",
  "started_at": "2025-12-01T15:12:34",
  "estimated_time_minutes": "35-45 minutes"
}
```

### Status values:
- `uploading` - Uploading video
- `processing` - Running COLMAP
- `training` - Training Gaussian Splat
- `completed` - Done! Model ready
- `failed` - Error occurred

---

## After Completion

### Get the trained model:
```bash
# Workflow status will show output_files
curl http://localhost:8003/workflow/status/WORKFLOW_ID | jq '.output_files'

# Copy to rendering service for viewing
TRAIN_JOB_ID="job_XXXXXXXX"  # From workflow status

docker exec fvdb-training-gpu cp \
  /app/outputs/$TRAIN_JOB_ID/tesla_model_s_model.ply \
  /app/models/tesla_model_s.ply
```

### View model:
**http://localhost:8001**

---

## For Photo ZIP Files

Same process! Just upload a ZIP containing JPG images:

```bash
curl -X POST http://localhost:8003/workflow/video-to-model \
  -F "file=@photos.zip" \
  -F "dataset_id=my_photos" \
  -F "camera_model=SIMPLE_RADIAL" \
  -F "matcher=exhaustive" \
  -F "num_training_steps=30000"
```

(No need for fps parameter with photo ZIP)

---

## Troubleshooting

### Check if it's stuck:
```bash
# View COLMAP logs
docker logs colmap-processor --tail 50

# View training logs  
docker logs fvdb-training-gpu --tail 50
```

### Workflow failed?
Check the `error` field in status:
```bash
curl http://localhost:8003/workflow/status/WORKFLOW_ID | jq '.error'
```

---

## Summary

**ONE ENDPOINT DOES IT ALL!**

✅ Upload → Extract → COLMAP → Train → Model  
✅ Fire and forget (35-45 minutes)  
✅ Monitor with simple status endpoint  
✅ Works for video or photo ZIP  

**Refresh Swagger UI at http://localhost:8003/api to see the new endpoint!**

