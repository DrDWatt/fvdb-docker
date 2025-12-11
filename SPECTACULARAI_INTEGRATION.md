# SpectacularAI Gaussian Splatting Integration

This document describes the SpectacularAI integration for SLAM-based 3D Gaussian Splatting.

## Overview

The SpectacularAI service provides an alternative pipeline for 3D Gaussian Splatting that uses:
- **SpectacularAI SDK** for SLAM-based camera pose estimation
- **Nerfstudio** for Gaussian Splatting training
- **PLY export** for integration with existing rendering/viewing services

### Key Differences from COLMAP Pipeline

| Feature | COLMAP Pipeline | SpectacularAI Pipeline |
|---------|-----------------|------------------------|
| Input | Images/Video | SpectacularAI recordings |
| Pose Estimation | Structure-from-Motion | Visual-Inertial SLAM |
| Speed | Minutes to hours | Real-time capable |
| Best For | Static scenes, high quality | Moving captures, speed |

## Supported Devices

- **iOS**: iPhone/iPad with [Spectacular Rec](https://apps.apple.com/us/app/spectacular-rec/id6473188128) app
- **Android**: Devices with [Spectacular Rec](https://play.google.com/store/apps/details?id=com.spectacularai.rec) app
- **OAK-D**: Luxonis depth cameras
- **Intel RealSense**: D4xx series
- **Azure Kinect**: Microsoft depth camera
- **Orbbec**: Astra/Femto series

## Quick Start

### Option 1: Integrated with Existing Workflow

Uses shared models directory with existing rendering/viewing services:

```bash
# Start SpectacularAI service (connects to existing workflow network)
docker compose -f docker-compose.spectacularai.yml up -d

# Ensure existing services are running
docker compose -f docker-compose.workflow.yml up -d
```

**Ports:**
- SpectacularAI API: http://localhost:8004
- Rendering (shared): http://localhost:8001
- Viewer (shared): http://localhost:8085

### Option 2: Standalone Workflow

Completely separate, non-interfering workflow:

```bash
docker compose -f docker-compose.spectacularai-standalone.yml up -d
```

**Ports:**
- SpectacularAI API: http://localhost:8004
- Rendering: http://localhost:8011
- Viewer: http://localhost:8095

## API Endpoints

### Swagger Documentation

- **Swagger UI**: http://localhost:8004/docs
- **ReDoc**: http://localhost:8004/redoc
- **OpenAPI JSON**: http://localhost:8004/openapi.json

### Key Endpoints

#### Upload Recording
```bash
curl -X POST "http://localhost:8004/upload?scene_size=medium&max_iterations=30000" \
  -F "file=@recording.zip"
```

#### Check Job Status
```bash
curl "http://localhost:8004/jobs/{job_id}"
```

#### List Models
```bash
curl "http://localhost:8004/models"
```

#### Download PLY
```bash
curl -O "http://localhost:8004/models/{model_name}/download"
```

## Workflow

### 1. Record Scene

Use Spectacular Rec app on iOS/Android:
1. Open app and select recording mode
2. Slowly scan the scene from multiple angles
3. Export recording as zip file
4. Transfer to computer

### 2. Upload and Process

```bash
# Upload recording
curl -X POST "http://localhost:8004/upload" \
  -F "file=@my_scene.zip" \
  -F "scene_size=medium"

# Response includes job_id
# {"job_id": "abc-123", "status": "pending"}
```

### 3. Monitor Progress

```bash
# Check status
curl "http://localhost:8004/jobs/abc-123"

# Poll until completed
watch -n 5 'curl -s http://localhost:8004/jobs/abc-123 | jq .status'
```

### 4. View Results

Once completed, the model is automatically available in:
- Rendering service: `/models/{model_name}.ply`
- Viewer service: Select from model list

## Configuration

### Scene Size Options

| Size | Key Frame Distance | Use Case |
|------|-------------------|----------|
| `small` | 0.05m (5cm) | Table-sized objects |
| `medium` | 0.10m (10cm) | Room corners, furniture |
| `large` | 0.15m (15cm) | Full rooms, outdoor |

### Training Options

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `max_iterations` | 30000 | 1000-100000 | Training iterations |
| `fast_mode` | false | - | Trade quality for speed |

## Integration with Rendering Service

The SpectacularAI service exports models to the shared `./models` directory. The rendering service automatically picks up new models.

To manually notify the rendering service:

```bash
curl -X POST "http://localhost:8004/models/{model_name}/notify-rendering"
```

## File Structure

```
spectacularai-service/
├── Dockerfile
├── requirements.txt
├── spectacularai_service.py    # Main API service
└── static/
    └── workflow.html           # Web UI

spectacularai-data/              # Created at runtime
├── uploads/                    # Uploaded recordings
├── processing/                 # Processing workspace
└── outputs/                    # Exported PLY files

spectacularai-models/           # Standalone mode models
models/                         # Shared with main workflow
```

## Troubleshooting

### Recording Not Processing

1. Check recording format (must be SpectacularAI format)
2. Verify file is not corrupted
3. Check logs: `docker logs spectacularai-processor`

### Training Fails

1. Check GPU memory: `nvidia-smi`
2. Reduce `max_iterations` for faster test
3. Enable `fast_mode` for initial testing

### Model Not Appearing in Viewer

1. Check model exists: `ls -la models/`
2. Restart rendering service: `docker restart fvdb-rendering`
3. Use notify endpoint to refresh

## API Reference

See full Swagger documentation at: http://localhost:8004/docs
