#!/bin/bash
# Build multi-architecture fVDB Docker images

set -e

echo "======================================================================"
echo "Building fVDB Docker Images"
echo "======================================================================"

# Check if buildx is available
if ! docker buildx version &> /dev/null; then
    echo "❌ Docker buildx not found. Please install Docker Buildx."
    exit 1
fi

# Create builder if it doesn't exist
if ! docker buildx inspect fvdb-builder &> /dev/null; then
    echo "📦 Creating buildx builder..."
    docker buildx create --name fvdb-builder --use
    docker buildx inspect --bootstrap
else
    echo "✅ Using existing buildx builder"
    docker buildx use fvdb-builder
fi

# Determine architecture
ARCH=$(uname -m)
case "$ARCH" in
    x86_64|amd64)
        PLATFORM="linux/amd64"
        ;;
    aarch64|arm64)
        PLATFORM="linux/arm64"
        ;;
    *)
        echo "❌ Unsupported architecture: $ARCH"
        exit 1
        ;;
esac

echo ""
echo "🏗️  Building for platform: $PLATFORM"
echo ""

# Build training service
echo "📦 Building Training Service..."
docker buildx build \
    --platform $PLATFORM \
    --tag fvdb-training:latest \
    --load \
    ./training-service

echo "✅ Training service built"
echo ""

# Build rendering service  
echo "📦 Building Rendering Service..."
docker buildx build \
    --platform $PLATFORM \
    --tag fvdb-rendering:latest \
    --load \
    ./rendering-service

echo "✅ Rendering service built"
echo ""

echo "======================================================================"
echo "✅ Build Complete!"
echo "======================================================================"
echo ""
echo "📋 Images created:"
docker images | grep fvdb
echo ""
echo "🚀 To start services:"
echo "   docker compose up -d"
echo ""
echo "🔍 To view logs:"
echo "   docker compose logs -f"
echo ""
echo "🌐 Access services:"
echo "   Training API:  http://localhost:8000"
echo "   Rendering API: http://localhost:8001"
echo ""
