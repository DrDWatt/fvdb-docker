#!/bin/bash
# Track workflow progress

echo "🔍 Finding active workflow..."
WORKFLOW_ID=$(curl -s http://localhost:8003/workflow/list | jq -r '.workflows[0].workflow_id' 2>/dev/null)

if [ -z "$WORKFLOW_ID" ] || [ "$WORKFLOW_ID" = "null" ]; then
    echo "❌ No active workflow found"
    echo "Upload may still be in progress"
    exit 1
fi

echo "✅ Tracking workflow: $WORKFLOW_ID"
echo ""

while true; do
    RESPONSE=$(curl -s http://localhost:8003/workflow/status/$WORKFLOW_ID)
    
    STATUS=$(echo $RESPONSE | jq -r '.status')
    PROGRESS=$(echo $RESPONSE | jq -r '.progress')
    STEP=$(echo $RESPONSE | jq -r '.current_step')
    
    PERCENT=$(echo "$PROGRESS * 100" | bc | cut -d. -f1)
    
    clear
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║          TESLA VIDEO → 3D MODEL WORKFLOW                  ║"
    echo "╚═══════════════════════════════════════════════════════════╝"
    echo ""
    echo "Workflow ID: $WORKFLOW_ID"
    echo "Dataset: omg"
    echo ""
    echo "Status: $STATUS"
    echo "Progress: $PERCENT%"
    echo "Current Step: $STEP"
    echo ""
    
    # Progress bar
    BAR_LENGTH=50
    FILLED=$((PERCENT * BAR_LENGTH / 100))
    printf "["
    for ((i=0; i<BAR_LENGTH; i++)); do
        if [ $i -lt $FILLED ]; then
            printf "="
        else
            printf " "
        fi
    done
    printf "] $PERCENT%%\n"
    echo ""
    
    if [ "$STATUS" = "completed" ]; then
        echo "✅ Workflow complete!"
        echo ""
        echo "Output files:"
        echo $RESPONSE | jq -r '.output_files[]' 2>/dev/null
        echo ""
        echo "To view model:"
        echo "1. Copy to rendering service:"
        TRAIN_JOB=$(echo $RESPONSE | jq -r '.training_job_id')
        echo "   docker exec fvdb-training-gpu cp /app/outputs/$TRAIN_JOB/omg_model.ply /app/models/tesla.ply"
        echo "2. View at: http://localhost:8001"
        break
    elif [ "$STATUS" = "failed" ]; then
        echo "❌ Workflow failed!"
        ERROR=$(echo $RESPONSE | jq -r '.error')
        echo "Error: $ERROR"
        break
    fi
    
    echo "Refreshing in 10 seconds... (Ctrl+C to stop)"
    sleep 10
done

