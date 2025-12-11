#!/bin/bash
# Start Advanced Rendering Containers

echo "🚀 Starting Advanced Rendering Services..."
echo ""

# Build and start containers
docker compose -f docker-compose.advanced-rendering.yml up --build -d

echo ""
echo "⏳ Waiting for services to start..."
sleep 10

echo ""
echo "📊 Container Status:"
docker compose -f docker-compose.advanced-rendering.yml ps

echo ""
echo "✅ Services Available:"
echo "  🎬 USD Pipeline:      http://localhost:8002"
echo "  📺 WebRTC Renderer:   http://localhost:8888"
echo "  🎨 Existing Streaming: http://localhost:8080/test"
echo "  📁 PLY Files Service:  http://localhost:8001"
echo ""
echo "📖 Documentation: README_ADVANCED_RENDERING.md"
echo ""
echo "🔍 Check logs:"
echo "  docker logs usd_converter"
echo "  docker logs webrtc_visualizer"
