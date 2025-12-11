#\!/bin/bash
# Automated workflow script for new video

set -e

DATASET_ID="$1"
VIDEO_FILE="$2"

if [ -z "$DATASET_ID" ] || [ -z "$VIDEO_FILE" ]; then
    echo "Usage: ./process_new_video.sh <dataset_id> <video_file>"
    echo "Example: ./process_new_video.sh kitchen_v2 /path/to/video.mov"
    exit 1
fi

echo "=================================="
echo "3D Gaussian Splatting Workflow"
echo "=================================="
echo "Dataset: $DATASET_ID"
echo "Video: $VIDEO_FILE"
echo ""

# Step 1: Upload video
echo "📤 [1/4] Uploading video..."
UPLOAD_RESPONSE=$(curl -s -X POST http://localhost:8003/upload \
  -F "file=@$VIDEO_FILE" \
  -F "dataset_id=$DATASET_ID")
echo "Response: $UPLOAD_RESPONSE"
echo ""

# Step 2: Process with COLMAP
echo "🔄 [2/4] Processing with COLMAP (this takes 5-10 minutes)..."
PROCESS_RESPONSE=$(curl -s -X POST http://localhost:8003/video/process \
  -H "Content-Type: application/json" \
  -d "{
    \"dataset_id\": \"$DATASET_ID\",
    \"video_filename\": \"$(basename $VIDEO_FILE)\",
    \"fps\": 1.0,
    \"camera_model\": \"SIMPLE_RADIAL\",
    \"matcher\": \"exhaustive\",
    \"max_image_size\": 2048,
    \"max_num_features\": 16384
  }")

JOB_ID=$(echo $PROCESS_RESPONSE | jq -r '.job_id // empty')

if [ -z "$JOB_ID" ]; then
    echo "❌ Failed to start COLMAP processing"
    echo "Response: $PROCESS_RESPONSE"
    exit 1
fi

echo "Job ID: $JOB_ID"

# Wait for COLMAP to complete
while true; do
    STATUS=$(curl -s http://localhost:8003/jobs | jq -r ".jobs[] | select(.job_id==\"$JOB_ID\") | .status")
    PROGRESS=$(curl -s http://localhost:8003/jobs | jq -r ".jobs[] | select(.job_id==\"$JOB_ID\") | .progress")
    
    if [ "$STATUS" = "completed" ]; then
        echo "✅ COLMAP processing complete\!"
        break
    elif [ "$STATUS" = "failed" ]; then
        echo "❌ COLMAP processing failed"
        curl -s http://localhost:8003/jobs | jq ".jobs[] | select(.job_id==\"$JOB_ID\")"
        exit 1
    fi
    
    echo "Progress: $(echo "$PROGRESS * 100" | bc)%"
    sleep 10
done

NUM_IMAGES=$(curl -s http://localhost:8003/jobs | jq -r ".jobs[] | select(.job_id==\"$JOB_ID\") | .num_images")
echo "Registered images: $NUM_IMAGES"

if [ "$NUM_IMAGES" -lt 20 ]; then
    echo "⚠️  WARNING: Only $NUM_IMAGES images registered. Recommend 40+ for good results."
    echo "Consider re-recording with slower movement and more overlap."
fi
echo ""

# Step 3: Download and extract COLMAP data
echo "📥 [3/4] Downloading COLMAP results..."
mkdir -p ./data/$DATASET_ID
curl -s -o /tmp/${DATASET_ID}_colmap.zip http://localhost:8003/download/$JOB_ID
sudo rm -rf ./data/$DATASET_ID/*
sudo unzip -q /tmp/${DATASET_ID}_colmap.zip -d ./data/$DATASET_ID
sudo chown -R $(whoami):$(whoami) ./data/$DATASET_ID
echo "✅ COLMAP data extracted to ./data/$DATASET_ID"
echo ""

# Step 4: Train Gaussian Splat
echo "🎓 [4/4] Training Gaussian Splat (this takes 20-30 minutes)..."
TRAIN_RESPONSE=$(curl -s -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d "{
    \"dataset_id\": \"$DATASET_ID\",
    \"num_training_steps\": 30000,
    \"output_name\": \"${DATASET_ID}_model\"
  }")

TRAIN_JOB_ID=$(echo $TRAIN_RESPONSE | jq -r '.job_id // empty')

if [ -z "$TRAIN_JOB_ID" ]; then
    echo "❌ Failed to start training"
    echo "Response: $TRAIN_RESPONSE"
    exit 1
fi

echo "Training Job ID: $TRAIN_JOB_ID"
echo "Monitor with: docker logs -f fvdb-training-gpu"
echo ""
echo "Once complete, model will be at: ./outputs/$TRAIN_JOB_ID/${DATASET_ID}_model.ply"
echo ""
echo "To view model:"
echo "  docker exec fvdb-training-gpu cp /app/outputs/$TRAIN_JOB_ID/${DATASET_ID}_model.ply /app/models/"
echo "  Open: http://localhost:8001"
echo ""
echo "✅ Workflow initiated successfully\!"

