# Video to 3D Gaussian Splat - Complete Guide

Convert your videos into 3D Gaussian Splat models using fVDB Reality Capture.

---

## Quick Start

### 1. Extract Frames from Video

```bash
# Upload video and extract frames
curl -X POST "http://localhost:8000/video/extract" \
  -F "file=@your_video.mp4" \
  -F "fps=2.0" \
  -F "output_name=my_object" \
  | python3 -m json.tool
```

### 2. Process with COLMAP

The extracted frames need to be processed with COLMAP to generate camera poses and sparse reconstruction.

```bash
# You'll need to run COLMAP on the extracted frames
# Or upload frames that already have COLMAP data
```

### 3. Train 3D Gaussian Splat

```bash
# Use the dataset_id from step 1
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "my_object_frames",
    "num_training_steps": 30000,
    "output_name": "my_object_splat"
  }' | python3 -m json.tool
```

---

## Video Capture Best Practices

### Recording Tips

**Camera Movement:**
- Move in a **circular path** around the object
- Keep the object centered in frame
- Maintain consistent distance
- Smooth, steady movement (use gimbal if possible)
- 360° coverage recommended

**Lighting:**
- Even, diffused lighting
- Avoid harsh shadows
- Consistent lighting throughout video
- No moving lights or changing conditions

**Camera Settings:**
- **1080p or 4K resolution**
- **30 or 60 fps** (higher is better for smooth extraction)
- Fixed focus (disable autofocus if possible)
- Fixed exposure
- Avoid motion blur (use faster shutter speed)

**Subject:**
- Non-reflective surfaces work best
- Rich texture details
- Avoid transparent or shiny objects
- Static object (no movement)

---

## Frame Extraction Parameters

### FPS (Frames Per Second)

Controls how many frames to extract per second of video.

**Recommendations:**
- **Small objects / short videos (< 30 sec):** `fps=2-3`
  - Example: 20 second video at 2 fps = 40 frames ✓
- **Medium videos (30-60 sec):** `fps=1-2`
  - Example: 45 second video at 1.5 fps = 68 frames ✓
- **Long videos (> 60 sec):** `fps=0.5-1`
  - Example: 90 second video at 0.7 fps = 63 frames ✓

**Target:** 40-60 frames for best results

### Max Frames

Limit total number of frames extracted.

```bash
# Extract at most 60 frames regardless of fps
curl -X POST "http://localhost:8000/video/extract" \
  -F "file=@video.mp4" \
  -F "fps=2.0" \
  -F "max_frames=60"
```

### Quality

JPEG quality for extracted frames (set in code, default: 2 = very high quality)

---

## Complete Workflow Examples

### Example 1: iPhone Video → 3D Splat

**1. Record video on iPhone:**
- Walk in circle around object
- 30-60 seconds
- 4K 60fps
- Keep object in center

**2. Extract frames:**
```bash
curl -X POST "http://localhost:8000/video/extract" \
  -F "file=@iphone_object.mov" \
  -F "fps=1.5" \
  -F "output_name=iphone_object" \
  | python3 -m json.tool
```

Response:
```json
{
  "output_name": "iphone_object",
  "video_info": {
    "duration": 45.2,
    "width": 3840,
    "height": 2160
  },
  "extraction": {
    "num_frames": 68
  },
  "dataset_id": "iphone_object_frames"
}
```

**3. Process with COLMAP:**
*(Currently requires manual COLMAP processing)*

**4. Train:**
```bash
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "iphone_object_frames",
    "num_training_steps": 30000,
    "output_name": "iphone_object_3d"
  }'
```

---

## Video Recommendations API

Get automatic recommendations for frame extraction:

```bash
# The /video/extract endpoint automatically provides recommendations
# based on video duration
```

Example response:
```json
{
  "recommendations": {
    "recommended_fps": 1.8,
    "estimated_frames": 54,
    "video_duration": 30.0,
    "note": "Extracting at 1.80 fps will yield ~54 frames"
  }
}
```

---

## Supported Video Formats

- **MP4** (.mp4) - Recommended
- **MOV** (.mov) - iPhone/QuickTime
- **AVI** (.avi)
- **MKV** (.mkv)
- **WebM** (.webm)
- **FLV** (.flv)

---

## Troubleshooting

### "Too many frames"

Reduce fps or set max_frames:
```bash
-F "fps=1.0" \
-F "max_frames=50"
```

### "Too few frames"

Increase fps or use longer video:
```bash
-F "fps=3.0"
```

### "Video too large"

Compress video before upload:
```bash
ffmpeg -i input.mov -vcodec h264 -acodec aac output.mp4
```

### "Blurry frames"

- Use higher quality video (4K)
- Reduce motion blur (faster shutter)
- Slower camera movement

---

## Advanced: Manual Frame Extraction

Using the standalone script:

```bash
# Inside container
docker exec fvdb-training python3 /app/extract_frames.py \
  /app/uploads/video.mp4 \
  /app/data/my_frames \
  2.0 \
  60
```

Parameters:
1. Video file path
2. Output directory
3. FPS (optional, default: 2.0)
4. Max frames (optional, default: unlimited)

---

## Full Pipeline: Video → SuperSplat

```bash
# 1. Extract frames from video
RESULT=$(curl -X POST "http://localhost:8000/video/extract" \
  -F "file=@my_video.mp4" \
  -F "fps=2.0" \
  -F "output_name=my_model")

DATASET_ID=$(echo $RESULT | python3 -c "import sys, json; print(json.load(sys.stdin)['dataset_id'])")

echo "Frames extracted! Dataset: $DATASET_ID"

# 2. TODO: Process with COLMAP
#    (Manual step for now)

# 3. Train 3D Gaussian Splat
JOB=$(curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d "{
    \"dataset_id\": \"$DATASET_ID\",
    \"num_training_steps\": 30000,
    \"output_name\": \"my_model_3d\"
  }")

JOB_ID=$(echo $JOB | python3 -c "import sys, json; print(json.load(sys.stdin)['job_id'])")

echo "Training started! Job: $JOB_ID"

# 4. Monitor training
watch -n 10 "curl -s http://localhost:8000/jobs/$JOB_ID | python3 -m json.tool"

# 5. When complete, download PLY
curl -s "http://localhost:8001/static/downloads/my_model_3d.ply" -o my_model.ply

# 6. Open in SuperSplat
# https://playcanvas.com/supersplat/editor
```

---

## Performance Notes

**Frame Extraction Speed:**
- 1080p video: ~2-5 seconds per 30 seconds of video
- 4K video: ~5-10 seconds per 30 seconds of video

**Storage:**
- Frames: ~1-5 MB each
- 50 frames @ 4K: ~250 MB
- Final PLY: ~200-300 MB

**Training Time:**
- 40-60 frames: 30-45 minutes @ 30k steps
- More frames = longer training but better quality

---

## Tips for Best Results

1. **Record more than you need**
   - 30-90 seconds is ideal
   - Extract best portion using fps/max_frames

2. **Multiple passes**
   - First pass: circular around object
   - Second pass: higher/lower angles
   - Extract from both videos

3. **Consistent motion**
   - Smooth, constant speed
   - No sudden movements
   - No zooming

4. **Good lighting**
   - Overcast day (outdoors)
   - Soft boxes (indoors)
   - Avoid direct sunlight

5. **Frame overlap**
   - Adjacent frames should share 80%+ content
   - Helps COLMAP find feature matches

---

## Next Steps

After frame extraction, you need to:

1. **Run COLMAP** on extracted frames to generate:
   - Camera poses
   - Sparse point cloud
   - Intrinsic/extrinsic parameters

2. **Package as dataset** with structure:
   ```
   dataset/
   ├── images/          # Your extracted frames
   └── sparse/
       └── 0/
           ├── cameras.bin
           ├── images.bin
           └── points3D.bin
   ```

3. **Upload & train** using the `/workflow/complete` endpoint

---

## Future Enhancements

- [ ] Automatic COLMAP processing
- [ ] Real-time frame preview
- [ ] Video quality analysis
- [ ] Automatic fps recommendation
- [ ] Batch video processing
- [ ] Frame deduplication
- [ ] Motion blur detection

---

**Ready to create 3D models from your videos!**
