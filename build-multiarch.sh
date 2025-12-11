#!/bin/bash
# Build and push multi-architecture images to registry

set -e

# Configuration
REGISTRY="${REGISTRY:-localhost:7000}"
IMAGE_NAME="fvdb"
VERSION="${VERSION:-latest}"

echo "======================================================================"
echo "Building Multi-Architecture fVDB Docker Images"
echo "======================================================================"
echo ""
echo "Registry: $REGISTRY"
echo "Image: $IMAGE_NAME"
echo "Version: $VERSION"
echo "Platforms: linux/amd64, linux/arm64"
echo ""

# Check if buildx is available
if ! docker buildx version &> /dev/null; then
    echo "❌ Docker buildx not found"
    exit 1
fi

# Create/use builder
if ! docker buildx inspect multiarch-builder &> /dev/null; then
    echo "📦 Creating multi-arch builder..."
    docker buildx create --name multiarch-builder --use
    docker buildx inspect --bootstrap
else
    echo "✅ Using existing multi-arch builder"
    docker buildx use multiarch-builder
fi

# Build and push training service
echo ""
echo "🏗️  Building Training Service (multi-arch)..."
docker buildx build \
    --platform linux/amd64,linux/arm64 \
    --tag $REGISTRY/$IMAGE_NAME-training:$VERSION \
    --tag $REGISTRY/$IMAGE_NAME-training:latest \
    --push \
    ./training-service

echo "✅ Training service pushed to registry"

# Build and push rendering service
echo ""
echo "🏗️  Building Rendering Service (multi-arch)..."
docker buildx build \
    --platform linux/amd64,linux/arm64 \
    --tag $REGISTRY/$IMAGE_NAME-rendering:$VERSION \
    --tag $REGISTRY/$IMAGE_NAME-rendering:latest \
    --push \
    ./rendering-service

echo "✅ Rendering service pushed to registry"

echo ""
echo "======================================================================"
echo "✅ Multi-Architecture Build Complete!"
echo "======================================================================"
echo ""
echo "📦 Images pushed to:"
echo "   $REGISTRY/$IMAGE_NAME-training:$VERSION"
echo "   $REGISTRY/$IMAGE_NAME-rendering:$VERSION"
echo ""
echo "🚀 To pull on any architecture:"
echo "   docker pull $REGISTRY/$IMAGE_NAME-training:$VERSION"
echo "   docker pull $REGISTRY/$IMAGE_NAME-rendering:$VERSION"
echo ""
