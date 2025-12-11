#!/bin/bash
# Start Complete 3D Reconstruction Workflow

echo "🚀 Starting Complete 3D Reconstruction Workflow"
echo "=============================================="
echo ""

# Create directories
echo "📁 Creating data directories..."
mkdir -p colmap-data/{uploads,processing,outputs,temp}
mkdir -p data uploads outputs models usd-outputs

# Build COLMAP service if not exists
if [[ "$(docker images -q colmap-service:latest 2> /dev/null)" == "" ]]; then
  echo "🔨 Building COLMAP service (this will take 10-15 minutes first time)..."
  docker build -t colmap-service:latest colmap-service/
else
  echo "✅ COLMAP service image exists"
fi

# Start all services
echo "🎬 Starting all services..."
docker compose -f docker-compose.workflow.yml up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to start..."
sleep 15

# Check service health
echo ""
echo "🏥 Health Check:"
echo "=============================================="

services=("colmap-processor:8003" "fvdb-training-gpu:8000" "fvdb-rendering:8001" "usd-pipeline:8002" "streaming-server:8080")

for service in "${services[@]}"; do
    IFS=':' read -ra ADDR <<< "$service"
    name="${ADDR[0]}"
    port="${ADDR[1]}"
    
    if curl -sf "http://localhost:$port/health" > /dev/null 2>&1; then
        echo "✅ $name (port $port)"
    else
        echo "⚠️  $name (port $port) - starting..."
    fi
done

# Display URLs
echo ""
echo "🎉 Workflow Stack Ready!"
echo "=============================================="
echo ""
echo "📍 Access Points:"
echo "   🎬 Workflow UI:    http://localhost:8080/workflow"
echo "   👁️  3D Viewer:      http://localhost:8080/test"
echo "   🔬 COLMAP API:     http://localhost:8003/api"
echo "   🎯 Training API:   http://localhost:8000/api"
echo "   🎨 USD Pipeline:   http://localhost:8002"
echo "   📦 Rendering:      http://localhost:8001/docs"
echo ""
echo "📊 Service Status:"
docker compose -f docker-compose.workflow.yml ps
echo ""
echo "💡 Quick Start:"
echo "   1. Open http://localhost:8080/workflow"
echo "   2. Upload MP4/MOV video or ZIP of photos"
echo "   3. Run COLMAP processing"
echo "   4. Train Gaussian Splat model"
echo "   5. Download your 3D model!"
echo ""
echo "📚 Full documentation: COMPLETE_WORKFLOW_SETUP.md"
echo ""
