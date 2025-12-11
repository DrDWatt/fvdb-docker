# Tesla Model S - Upload & Processing Guide

## Option 1: Swagger UI (What You're Looking At)

### Step 1: Upload Video
**Endpoint:** `POST /upload`
- Click "Try it out"
- dataset_id: `tesla_model_s`
- file: Choose your 415MB video file
- Click "Execute"

This uploads the video to the server.

### Step 2: Process with COLMAP
**Endpoint:** `POST /video/process`
- Click "Try it out"
- Request body:
```json
{
  "dataset_id": "tesla_model_s",
  "video_filename": "your_video_filename.mov",
  "fps": 1.0,
  "camera_model": "SIMPLE_RADIAL",
  "matcher": "exhaustive",
  "max_image_size": 2048,
  "max_num_features": 16384
}
```
- Click "Execute"

### Step 3: Monitor Progress
**Endpoint:** `GET /jobs`
- Keep checking until status = "completed"
- Takes 5-10 minutes

### Step 4: Download COLMAP Results
**Endpoint:** `GET /download/{job_id}`
- Use the job_id from Step 3
- Save the ZIP file

---

## Option 2: Command Line (Faster)

```bash
cd /home/dwatkins3/fvdb-docker

# Upload video
curl -X POST http://localhost:8003/upload \
  -F "file=@/path/to/tesla_video.mov" \
  -F "dataset_id=tesla_model_s"

# Process with COLMAP (replace filename)
curl -X POST http://localhost:8003/video/process \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "tesla_model_s",
    "video_filename": "tesla_video.mov",
    "fps": 1.0,
    "camera_model": "SIMPLE_RADIAL",
    "matcher": "exhaustive",
    "max_image_size": 2048,
    "max_num_features": 16384
  }'

# Check status
curl http://localhost:8003/jobs | jq

# When complete, download
curl -o tesla_colmap.zip http://localhost:8003/download/JOBID

# Extract to training directory
sudo unzip tesla_colmap.zip -d ./data/tesla_model_s
sudo chown -R $(whoami):$(whoami) ./data/tesla_model_s
```

---

## After COLMAP Completes

### Train Gaussian Splat Model
Go to: http://localhost:8000/api

**Endpoint:** `POST /train`
```json
{
  "dataset_id": "tesla_model_s",
  "num_training_steps": 30000,
  "output_name": "tesla_model"
}
```

Or via curl:
```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "tesla_model_s",
    "num_training_steps": 30000,
    "output_name": "tesla_model"
  }'
```

Training takes 20-30 minutes.

---

## View Final Model

### Copy to rendering service:
```bash
# Get job ID from training
TRAIN_JOB_ID="job_XXXXXXXX"

docker exec fvdb-training-gpu cp \
  /app/outputs/$TRAIN_JOB_ID/tesla_model.ply \
  /app/models/tesla_model.ply
```

### View at:
http://localhost:8001

---

## Expected Timeline

| Step | Duration |
|------|----------|
| Upload | 1-2 minutes |
| COLMAP Processing | 8-12 minutes |
| Training | 25-30 minutes |
| **Total** | **~35-45 minutes** |

