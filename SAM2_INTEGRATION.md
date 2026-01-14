# SAM-2 Segmentation Service Integration

GPU-accelerated object segmentation using **Segment Anything Model 2** (SAM-2) from Meta AI.

## Overview

The SAM-2 service provides:
- **Zero-shot object segmentation** - No training required
- **Video consistency tracking** - SAM-2 maintains temporal consistency across frames
- **Multiple segmentation modes** - Auto, point prompts, box prompts
- **Splat render segmentation** - Segment objects in rendered Gaussian Splat views
- **Label management** - Save and export object labels

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SAM-2 Integration Architecture                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐        │
│  │  Training       │────▶│  SAM-2 Service  │◀────│  fVDB Viewer    │        │
│  │  :8000          │     │  :8004          │     │  :8085          │        │
│  └─────────────────┘     └────────┬────────┘     └─────────────────┘        │
│                                   │                                          │
│                                   ▼                                          │
│                          ┌─────────────────┐                                 │
│                          │  Rendering      │                                 │
│                          │  :8001          │                                 │
│                          └─────────────────┘                                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Start SAM-2 Service (with existing workflow)

```bash
# Start all services including SAM-2
docker compose -f docker-compose.workflow.yml up -d

# Or start SAM-2 standalone
docker compose -f docker-compose.sam2.yml up -d
```

### Access the Service

| Endpoint | URL | Description |
|----------|-----|-------------|
| Web UI | http://localhost:8004 | Interactive segmentation interface |
| API Docs | http://localhost:8004/api | Swagger API documentation |
| Health | http://localhost:8004/health | Service health check |

## API Endpoints

### Image Segmentation

```bash
# Auto segmentation (all objects)
curl -X POST http://localhost:8004/api/segment/image \
  -F "file=@image.jpg" \
  -F "mode=auto"

# Point-based segmentation
curl -X POST http://localhost:8004/api/segment/image \
  -F "file=@image.jpg" \
  -F "mode=point" \
  -F 'points_json=[{"x": 100, "y": 200, "label": 1}]'

# Box-based segmentation
curl -X POST http://localhost:8004/api/segment/image \
  -F "file=@image.jpg" \
  -F "mode=box" \
  -F 'boxes_json=[{"x1": 50, "y1": 50, "x2": 200, "y2": 200}]'
```

### Video Segmentation (with tracking)

```bash
curl -X POST http://localhost:8004/api/segment/video \
  -F "file=@video.mp4" \
  -F 'prompts_json={"0": [{"x": 100, "y": 200, "label": 1}]}' \
  -F "track_objects=true"
```

### Splat Render Segmentation

```bash
# Segment objects in a rendered splat view
curl -X POST http://localhost:8004/api/segment/splat-render \
  -F "model_name=my_model" \
  -F "mode=auto"
```

### Get Results

```bash
# Get job status
curl http://localhost:8004/api/jobs/{job_id}

# Get mask image
curl http://localhost:8004/api/mask/{job_id}/{mask_idx} -o mask.png

# Get overlay image
curl http://localhost:8004/api/overlay/{job_id} -o overlay.png
```

### Save Labels

```bash
curl -X POST http://localhost:8004/api/labels/save \
  -F "job_id={job_id}" \
  -F 'labels_json={"0": "car", "1": "person", "2": "tree"}'
```

## Segmentation Modes

### Auto Mode
Automatically detects and segments all objects in the image. Best for:
- Initial exploration of scene contents
- Getting all possible segments
- No prior knowledge of objects

### Point Mode
Click on objects to segment them. Supports:
- **Foreground points** (label=1): Include this region
- **Background points** (label=0): Exclude this region
- Multiple points for complex selections

### Box Mode
Draw bounding boxes around objects. Best for:
- Known object locations
- Multiple specific objects
- Integration with object detection

## Model Sizes

| Model | Speed | Quality | VRAM | Use Case |
|-------|-------|---------|------|----------|
| Tiny | Fastest | Good | ~2GB | Real-time, preview |
| Small | Fast | Better | ~4GB | Balanced |
| Base+ | Medium | High | ~6GB | **Recommended** |
| Large | Slow | Best | ~8GB | Maximum quality |

## Integration with fVDB Viewer

The SAM-2 service integrates with the fVDB viewer to enable:

1. **Segment rendered views** - Automatically segment objects in splat renders
2. **Object labeling** - Assign labels to segments for scene understanding
3. **Export masks** - Download segmentation masks for further processing

### Workflow Example

```python
# 1. Train a Gaussian Splat model
# ... using existing workflow at :8000

# 2. Segment a rendered view
response = requests.post("http://localhost:8004/api/segment/splat-render", data={
    "model_name": "my_trained_model",
    "mode": "auto"
})

# 3. Get segmentation results
job_id = response.json()["job_id"]
masks = requests.get(f"http://localhost:8004/api/jobs/{job_id}").json()["masks"]

# 4. Save labels
requests.post("http://localhost:8004/api/labels/save", data={
    "job_id": job_id,
    "labels_json": json.dumps({"0": "table", "1": "chair", "2": "lamp"})
})
```

## Video Consistency (SAM-2 Feature)

SAM-2's key advantage over SAM-1 is **video consistency**. When segmenting video:

1. Provide prompts on key frames
2. SAM-2 propagates masks through entire video
3. Objects maintain consistent IDs across frames
4. Handles occlusions and reappearances

This is particularly useful for:
- Segmenting objects in training videos before splat creation
- Tracking objects through rendered view sequences
- Creating consistent object labels across viewpoints

## Directory Structure

```
sam2-service/
├── Dockerfile
├── requirements.txt
├── sam2_service.py          # Main FastAPI service
└── static/
    └── index.html           # Web UI

sam2-data/
├── uploads/                 # Uploaded images/videos
├── outputs/                 # Segmentation results
├── models/                  # SAM-2 checkpoints
└── cache/                   # Torch cache
```

## Troubleshooting

### Model Download Issues

SAM-2 checkpoints are downloaded automatically on first use. If download fails:

```bash
# Manual download
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt \
  -O sam2-data/models/sam2_hiera_base_plus.pt
```

### GPU Memory Issues

If running out of VRAM:
1. Use a smaller model (tiny or small)
2. Reduce image resolution before upload
3. Process fewer objects at once

### Service Not Starting

```bash
# Check logs
docker logs sam2-segmentation

# Verify GPU access
docker exec sam2-segmentation nvidia-smi
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SAM2_CHECKPOINT_DIR` | `/app/models/sam2` | Checkpoint storage |
| `TORCH_HOME` | `/app/cache` | PyTorch cache |
| `VIEWER_SERVICE_URL` | `http://fvdb-viewer:8085` | Viewer service |
| `RENDERING_SERVICE_URL` | `http://fvdb-rendering:8001` | Rendering service |

## References

- [SAM-2 Paper](https://ai.meta.com/research/publications/sam-2-segment-anything-in-images-and-videos/)
- [SAM-2 GitHub](https://github.com/facebookresearch/segment-anything-2)
- [Meta AI Blog](https://ai.meta.com/blog/segment-anything-2/)
