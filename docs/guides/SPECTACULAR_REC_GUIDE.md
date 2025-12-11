# Using Spectacular Rec with fVDB Workflow

This guide shows how to use recordings from the **Spectacular Rec** iPhone/Android app with your existing COLMAP → fVDB training pipeline.

## Quick Start (Video Export)

The simplest approach - no special tools needed:

1. **Record** your scene in Spectacular Rec app
2. **Export as video** (MP4 or MOV format)
3. **Transfer** to your computer (AirDrop, iCloud, etc.)
4. **Upload** at http://localhost:8000/workflow
5. **Wait** for COLMAP processing → Training → Done!

The workflow automatically:
- Extracts frames from video (default: 2 FPS)
- Runs COLMAP structure-from-motion
- Trains Gaussian Splatting model
- Makes it available for viewing

## Alternative: Image Export

For higher quality results, export images directly:

### From Spectacular Rec App

1. Record your scene
2. Go to export options
3. Select "Raw data" or "Images" export
4. Transfer the ZIP file to your computer

### Convert for COLMAP

```bash
# Use the converter script
./scripts/spectacular-to-colmap.sh recording.zip ./extracted_images

# Create zip for upload
cd extracted_images
zip -r ../images.zip .

# Upload via web UI or API
curl -X POST http://localhost:8003/upload -F 'file=@images.zip'
```

## Recording Tips

For best 3D reconstruction results:

1. **Move slowly** - Reduces motion blur
2. **Overlap** - Ensure 60-80% overlap between views
3. **Multiple angles** - Walk around the subject
4. **Good lighting** - Avoid harsh shadows
5. **Stable shots** - Use both hands to hold phone

## Workflow URLs

| Service | URL | Purpose |
|---------|-----|---------|
| Main Workflow | http://localhost:8000/workflow | Upload & process |
| COLMAP API | http://localhost:8003/api | Processing status |
| Viewer | http://localhost:8085 | View trained models |
| Rendering | http://localhost:8001 | Render views |

## Why Not Use SpectacularAI SDK?

The SpectacularAI SDK provides faster SLAM-based pose estimation, but:
- Only available for x86_64 (not ARM64)
- Requires commercial license for ARM binaries

Your existing COLMAP workflow:
- Works on ARM64 ✅
- High quality results ✅
- Already running ✅

COLMAP may be slower than SLAM, but produces excellent results for static scenes.
