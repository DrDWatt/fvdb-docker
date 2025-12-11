#!/bin/bash
# Build and push to container registry
# This builds x86_64 first (works immediately), then ARM64 (if possible)

set -e

# Configuration
REGISTRY="${REGISTRY:-localhost:7000}"
VERSION="${VERSION:-latest}"
BUILD_AMD64="${BUILD_AMD64:-true}"
BUILD_ARM64="${BUILD_ARM64:-false}"  # Disabled by default due to compilation issues

echo "======================================================================"
echo "Building fVDB Docker Images for Container Registry"
echo "======================================================================"
echo ""
echo "Registry: $REGISTRY"
echo "Version: $VERSION"
echo "Build AMD64: $BUILD_AMD64"
echo "Build ARM64: $BUILD_ARM64"
echo ""

# Check buildx
if ! docker buildx version &> /dev/null; then
    echo "❌ Docker buildx not found"
    exit 1
fi

# Create/use builder
if ! docker buildx inspect fvdb-builder &> /dev/null; then
    echo "📦 Creating buildx builder..."
    docker buildx create --name fvdb-builder --use --driver docker-container
    docker buildx inspect --bootstrap
else
    echo "✅ Using existing buildx builder"
    docker buildx use fvdb-builder
fi

# Determine platforms
PLATFORMS=""
if [ "$BUILD_AMD64" = "true" ]; then
    PLATFORMS="linux/amd64"
fi
if [ "$BUILD_ARM64" = "true" ]; then
    if [ -n "$PLATFORMS" ]; then
        PLATFORMS="$PLATFORMS,linux/arm64"
    else
        PLATFORMS="linux/arm64"
    fi
fi

if [ -z "$PLATFORMS" ]; then
    echo "❌ No platforms enabled. Set BUILD_AMD64=true or BUILD_ARM64=true"
    exit 1
fi

echo "🏗️  Building for platforms: $PLATFORMS"
echo ""

# Build Training Service
echo "📦 Building Training Service..."
if [ "$BUILD_AMD64" = "true" ] && [ "$BUILD_ARM64" = "false" ]; then
    # Use AMD64-specific Dockerfile (faster, uses pre-built wheels)
    docker buildx build \
        --platform linux/amd64 \
        --file training-service/Dockerfile.amd64 \
        --tag $REGISTRY/fvdb-training:$VERSION \
        --tag $REGISTRY/fvdb-training:latest \
        --push \
        ./training-service
else
    # Use multi-arch Dockerfile
    docker buildx build \
        --platform $PLATFORMS \
        --file training-service/Dockerfile \
        --tag $REGISTRY/fvdb-training:$VERSION \
        --tag $REGISTRY/fvdb-training:latest \
        --push \
        ./training-service
fi

echo "✅ Training service built and pushed"
echo ""

# Build Rendering Service
echo "📦 Building Rendering Service..."
if [ "$BUILD_AMD64" = "true" ] && [ "$BUILD_ARM64" = "false" ]; then
    # Use AMD64-specific Dockerfile
    docker buildx build \
        --platform linux/amd64 \
        --file rendering-service/Dockerfile.amd64 \
        --tag $REGISTRY/fvdb-rendering:$VERSION \
        --tag $REGISTRY/fvdb-rendering:latest \
        --push \
        ./rendering-service
else
    # Use multi-arch Dockerfile
    docker buildx build \
        --platform $PLATFORMS \
        --file rendering-service/Dockerfile \
        --tag $REGISTRY/fvdb-rendering:$VERSION \
        --tag $REGISTRY/fvdb-rendering:latest \
        --push \
        ./rendering-service
fi

echo "✅ Rendering service built and pushed"
echo ""

echo "======================================================================"
echo "✅ Build Complete!"
echo "======================================================================"
echo ""
echo "📦 Images pushed to registry:"
echo "   $REGISTRY/fvdb-training:$VERSION"
echo "   $REGISTRY/fvdb-rendering:$VERSION"
echo ""
echo "🚀 To pull and use:"
echo "   docker pull $REGISTRY/fvdb-training:$VERSION"
echo "   docker pull $REGISTRY/fvdb-rendering:$VERSION"
echo ""
echo "📝 Update docker-compose.yml to use registry images:"
echo "   image: $REGISTRY/fvdb-training:$VERSION"
echo "   image: $REGISTRY/fvdb-rendering:$VERSION"
echo ""
