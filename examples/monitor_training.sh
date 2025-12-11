#!/bin/bash
# Monitor training job progress

if [ $# -lt 1 ]; then
    echo "Usage: $0 <job_id>"
    exit 1
fi

JOB_ID=$1

echo "Monitoring job: $JOB_ID"
echo "Press Ctrl+C to stop monitoring"
echo ""

while true; do
    RESPONSE=$(curl -s "http://localhost:8000/jobs/$JOB_ID")
    
    STATUS=$(echo "$RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('status', 'unknown'))" 2>/dev/null || echo "error")
    PROGRESS=$(echo "$RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('progress', 0))" 2>/dev/null || echo "0")
    MESSAGE=$(echo "$RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('message', ''))" 2>/dev/null || echo "")
    
    # Clear line and print status
    echo -ne "\r\033[K"
    echo -n "Status: $STATUS | Progress: $(echo "$PROGRESS * 100" | bc)% | $MESSAGE"
    
    # Check if completed or failed
    if [ "$STATUS" = "completed" ]; then
        echo ""
        echo ""
        echo "✅ Training completed!"
        echo ""
        echo "$RESPONSE" | python3 -m json.tool
        break
    elif [ "$STATUS" = "failed" ]; then
        echo ""
        echo ""
        echo "❌ Training failed!"
        echo ""
        echo "$RESPONSE" | python3 -m json.tool
        exit 1
    fi
    
    sleep 5
done
