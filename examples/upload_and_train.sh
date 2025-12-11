#!/bin/bash
# Example: Upload dataset and start training

set -e

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <dataset.zip> [dataset_name] [training_steps]"
    echo ""
    echo "Example:"
    echo "  $0 my_dataset.zip my_scene 30000"
    exit 1
fi

DATASET_FILE=$1
DATASET_NAME=${2:-"dataset_$(date +%s)"}
TRAINING_STEPS=${3:-62200}

echo "======================================================================"
echo "Upload and Train Gaussian Splat"
echo "======================================================================"
echo ""
echo "Dataset file: $DATASET_FILE"
echo "Dataset name: $DATASET_NAME"
echo "Training steps: $TRAINING_STEPS"
echo ""

# Check if file exists
if [ ! -f "$DATASET_FILE" ]; then
    echo "❌ File not found: $DATASET_FILE"
    exit 1
fi

# Upload dataset
echo "📤 Uploading dataset..."
UPLOAD_RESPONSE=$(curl -s -X POST "http://localhost:8000/datasets/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@$DATASET_FILE" \
  -F "dataset_name=$DATASET_NAME")

echo "$UPLOAD_RESPONSE" | python3 -m json.tool
DATASET_ID=$(echo "$UPLOAD_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['dataset_id'])")
echo "✅ Dataset uploaded: $DATASET_ID"
echo ""

# Start training
echo "🚀 Starting training..."
TRAIN_RESPONSE=$(curl -s -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d "{
    \"dataset_id\": \"$DATASET_ID\",
    \"num_training_steps\": $TRAINING_STEPS,
    \"output_name\": \"model_$DATASET_NAME\"
  }")

echo "$TRAIN_RESPONSE" | python3 -m json.tool
JOB_ID=$(echo "$TRAIN_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['job_id'])")
echo "✅ Training started: $JOB_ID"
echo ""

echo "======================================================================"
echo "Training in progress!"
echo "======================================================================"
echo ""
echo "📊 Monitor progress:"
echo "   curl http://localhost:8000/jobs/$JOB_ID"
echo ""
echo "📥 Download when complete:"
echo "   curl -O http://localhost:8000/outputs/$JOB_ID/model_$DATASET_NAME.ply"
echo ""
echo "🌐 Or use Swagger UI:"
echo "   http://localhost:8000"
echo ""
