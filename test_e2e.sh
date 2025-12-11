#!/bin/bash
# End-to-end test script for fVDB workflow
set -e

echo "========================================="
echo "  fVDB Workflow End-to-End Test"
echo "========================================="

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

pass() { echo -e "${GREEN}✓ $1${NC}"; }
fail() { echo -e "${RED}✗ $1${NC}"; exit 1; }

# 1. Check all services are running
echo -e "\n[1/6] Checking services..."
docker ps --format "{{.Names}}" | grep -q "colmap-processor" && pass "COLMAP service" || fail "COLMAP service not running"
docker ps --format "{{.Names}}" | grep -q "fvdb-training-gpu" && pass "Training service" || fail "Training service not running"
docker ps --format "{{.Names}}" | grep -q "fvdb-rendering" && pass "Rendering service" || fail "Rendering service not running"
docker ps --format "{{.Names}}" | grep -q "fvdb-viewer" && pass "Viewer service" || fail "Viewer service not running"

# 2. Check health endpoints
echo -e "\n[2/6] Checking health endpoints..."
curl -sf http://localhost:8003/health > /dev/null && pass "COLMAP health (8003)" || fail "COLMAP health check failed"
curl -sf http://localhost:8000/health > /dev/null && pass "Training health (8000)" || fail "Training health check failed"
curl -sf http://localhost:8001/health > /dev/null && pass "Rendering health (8001)" || fail "Rendering health check failed"
curl -sf http://localhost:8085/health > /dev/null && pass "Viewer health (8085)" || fail "Viewer health check failed"

# 3. Check GPU availability
echo -e "\n[3/6] Checking GPU in training container..."
GPU_CHECK=$(docker exec fvdb-training-gpu python3 -c "import torch; print('yes' if torch.cuda.is_available() else 'no')" 2>/dev/null)
[ "$GPU_CHECK" = "yes" ] && pass "GPU available in training container" || fail "GPU not available"

# 4. Check fVDB availability
echo -e "\n[4/6] Checking fVDB..."
FVDB_CHECK=$(docker exec fvdb-training-gpu python3 -c "import fvdb; print(fvdb.__version__)" 2>/dev/null)
[ -n "$FVDB_CHECK" ] && pass "fVDB version: $FVDB_CHECK" || fail "fVDB not available"

# 5. Check models are visible
echo -e "\n[5/6] Checking model availability..."
MODEL_COUNT=$(curl -sf http://localhost:8001/models | python3 -c "import sys,json; print(json.load(sys.stdin)['count'])" 2>/dev/null)
[ "$MODEL_COUNT" -gt 0 ] && pass "Models in rendering service: $MODEL_COUNT" || fail "No models found"

TRAINING_MODELS=$(docker exec fvdb-training-gpu find /app/models -name '*.ply' 2>/dev/null | wc -l)
RENDERING_MODELS=$(docker exec fvdb-rendering find /app/models -name '*.ply' 2>/dev/null | wc -l)
[ "$TRAINING_MODELS" = "$RENDERING_MODELS" ] && \
    pass "Training and rendering see same models ($TRAINING_MODELS)" || fail "Model visibility mismatch"

# 6. Check viewer can see models
echo -e "\n[6/6] Checking viewer..."
VIEWER_MODELS=$(curl -sf http://localhost:8085/info | python3 -c "import sys,json; print(len(json.load(sys.stdin)['models_available']))" 2>/dev/null)
[ "$VIEWER_MODELS" -gt 0 ] && pass "Viewer sees $VIEWER_MODELS models" || fail "Viewer has no models"

echo -e "\n========================================="
echo -e "${GREEN}  All tests passed!${NC}"
echo "========================================="
echo ""
echo "Services:"
echo "  - COLMAP:    http://localhost:8003 (upload video)"
echo "  - Training:  http://localhost:8000 (train models)"
echo "  - Rendering: http://localhost:8001 (download PLY)"
echo "  - Viewer:    http://localhost:8085 (view models)"
echo ""
echo "End-to-end workflow:"
echo "  1. Upload video to COLMAP (8003)"
echo "  2. Training auto-starts after COLMAP completes"
echo "  3. View result in Viewer (8085)"
