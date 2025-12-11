#!/bin/bash
# Build Docker images using native installation wheels
# Run this on the DGX Spark where fVDB is already installed

set -e

echo "======================================================================"
echo "Building fVDB Docker Images from Native Installation"
echo "======================================================================"
echo ""

# Configuration
REGISTRY="${REGISTRY:-localhost:7000}"
VERSION="${VERSION:-latest}"
PLATFORM="linux/arm64"  # Native ARM64 build

# Get current architecture
ARCH=$(uname -m)
echo "Building on: $ARCH"
echo "Platform: $PLATFORM"
echo "Registry: $REGISTRY"
echo ""

# Step 1: Create wheels directory
echo "📦 Step 1: Creating wheels from native installation..."
mkdir -p training-service/wheels
mkdir -p rendering-service/wheels

# Step 2: Download/create wheels for fVDB packages
echo "📥 Step 2: Collecting fVDB wheels..."

# Option A: If you have pip cache, copy from there
if [ -d ~/.cache/pip/wheels ]; then
    echo "  Found pip cache, searching for fVDB wheels..."
    find ~/.cache/pip/wheels -name "fvdb*.whl" -exec cp {} training-service/wheels/ \; 2>/dev/null || true
    find ~/.cache/pip/wheels -name "torch*.whl" -exec cp {} training-service/wheels/ \; 2>/dev/null || true
fi

# Option B: Download wheels using pip download
if [ ! "$(ls -A training-service/wheels/*.whl 2>/dev/null)" ]; then
    echo "  Downloading wheels..."
    python3 -m pip download \
        --only-binary :all: \
        --dest training-service/wheels \
        --platform manylinux2014_aarch64 \
        --python-version 312 \
        torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu126 2>/dev/null || echo "  Note: Some packages may need compilation"
fi

# Option C: Create wheels from installed packages
echo "  Creating wheels from installed packages..."
python3 -m pip wheel \
    --wheel-dir=training-service/wheels \
    --no-deps \
    torch torchvision torchaudio fvdb fvdb-reality-capture 2>/dev/null || echo "  Note: Will install from PyPI if wheels unavailable"

# If no wheels found, that's okay - we'll install from PyPI in container
WHEEL_COUNT=$(ls -1 training-service/wheels/*.whl 2>/dev/null | wc -l)
echo "  Collected $WHEEL_COUNT wheel files"

if [ "$WHEEL_COUNT" -eq 0 ]; then
    echo "  ⚠️  No wheels found - will install directly from PyPI in container"
    echo "  This is okay but will take longer during build"
fi

echo ""

# Step 3: Build Training Service
echo "📦 Step 3: Building Training Service (native ARM64)..."
docker build \
    --platform $PLATFORM \
    --file training-service/Dockerfile.native \
    --tag $REGISTRY/fvdb-training:$VERSION \
    --tag $REGISTRY/fvdb-training:latest \
    ./training-service

echo "✅ Training service built"
echo ""

# Step 4: Build Rendering Service
echo "📦 Step 4: Building Rendering Service (native ARM64)..."
cp -r training-service/wheels rendering-service/wheels 2>/dev/null || true

docker build \
    --platform $PLATFORM \
    --file training-service/Dockerfile.native \
    --tag $REGISTRY/fvdb-rendering:$VERSION \
    --tag $REGISTRY/fvdb-rendering:latest \
    ./rendering-service

echo "✅ Rendering service built"
echo ""

# Cleanup
rm -rf training-service/wheels rendering-service/wheels

echo "======================================================================"
echo "✅ Build Complete!"
echo "======================================================================"
echo ""
echo "📦 Images created:"
docker images | grep fvdb
echo ""
echo "🚀 To run locally:"
echo "   docker compose up -d"
echo ""
echo "📤 To push to registry:"
echo "   docker push $REGISTRY/fvdb-training:$VERSION"
echo "   docker push $REGISTRY/fvdb-rendering:$VERSION"
echo ""
echo "🌐 Access services:"
echo "   Training:  http://localhost:8000"
echo "   Rendering: http://localhost:8001"
echo ""
