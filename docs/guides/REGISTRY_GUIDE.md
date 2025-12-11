# Container Registry Guide

## 🎯 Overview

This guide explains how to build and share fVDB Docker images via a container registry.

## 📦 Architecture Strategy

Due to ARM64 compilation complexity with fVDB dependencies, we use this approach:

- **AMD64/x86_64**: ✅ **Full support** - Uses pre-built wheels
- **ARM64/aarch64**: ⏳ **Partial support** - Requires additional build steps

## 🚀 Quick Start (AMD64)

### Step 1: Set Registry

```bash
export REGISTRY=localhost:7000  # e.g., docker.io/username, gcr.io/project
export VERSION=1.0.0
```

### Step 2: Build and Push

```bash
cd ~/fvdb-docker
./build-registry.sh
```

This builds **AMD64 images** and pushes to your registry.

### Step 3: Use from Registry

```bash
# On any machine with Docker + NVIDIA GPU
export REGISTRY=your-registry.com
export VERSION=1.0.0

docker compose -f docker-compose.registry.yml up -d
```

---

## 🏗️ Building for Different Registries

### Docker Hub

```bash
export REGISTRY=docker.io/yourusername
export VERSION=1.0.0
./build-registry.sh
```

**Result:**
- `docker.io/yourusername/fvdb-training:1.0.0`
- `docker.io/yourusername/fvdb-rendering:1.0.0`

### GitHub Container Registry

```bash
export REGISTRY=ghcr.io/yourusername
export VERSION=1.0.0
./build-registry.sh
```

### Google Container Registry

```bash
export REGISTRY=gcr.io/your-project
export VERSION=1.0.0
./build-registry.sh
```

### NVIDIA NGC

```bash
export REGISTRY=nvcr.io/your-org/your-team
export VERSION=1.0.0
./build-registry.sh
```

### AWS ECR

```bash
export REGISTRY=123456789.dkr.ecr.us-east-1.amazonaws.com
export VERSION=1.0.0
./build-registry.sh
```

---

## 🔧 Build Options

### AMD64 Only (Default - Recommended)

```bash
export BUILD_AMD64=true
export BUILD_ARM64=false
./build-registry.sh
```

**Advantages:**
- ✅ Fast build (~10-15 minutes)
- ✅ Uses pre-built wheels
- ✅ Works immediately
- ✅ Most users run AMD64

### Multi-Architecture (Experimental)

```bash
export BUILD_AMD64=true
export BUILD_ARM64=true
./build-registry.sh
```

**Note:** ARM64 build requires:
- Additional compilation time (~30-60 min)
- May fail on some dependencies
- Best to pre-build wheels separately

---

## 📊 Image Sizes

| Image | AMD64 Size | Description |
|-------|------------|-------------|
| **fvdb-training** | ~8-10 GB | Full CUDA + PyTorch + fVDB |
| **fvdb-rendering** | ~6-8 GB | Runtime only (smaller) |

---

## 🔐 Authentication

### Docker Hub

```bash
docker login
# Enter username and password
./build-registry.sh
```

### GitHub Container Registry

```bash
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin
./build-registry.sh
```

### Google Cloud

```bash
gcloud auth configure-docker
./build-registry.sh
```

### AWS ECR

```bash
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin 123456789.dkr.ecr.us-east-1.amazonaws.com
./build-registry.sh
```

---

## 🎯 Using Images from Registry

### Pull Images

```bash
# If using local registry on port 7000
export REGISTRY=localhost:7000
docker pull $REGISTRY/fvdb-training:$VERSION
docker pull $REGISTRY/fvdb-rendering:$VERSION
```

### Run with Docker Compose

```bash
export REGISTRY=your-registry.com
export VERSION=1.0.0
docker compose -f docker-compose.registry.yml up -d
```

### Run Standalone

```bash
docker run -d \
  --gpus all \
  -p 8000:8000 \
  --name fvdb-training \
  $REGISTRY/fvdb-training:$VERSION
```

---

## 🔄 CI/CD Integration

### GitHub Actions Example

```yaml
name: Build and Push fVDB Images

on:
  push:
    tags:
      - 'v*'

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2
      
      - name: Login to GitHub Container Registry
        uses: docker/login-action@v2
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Build and push
        env:
          REGISTRY: ghcr.io/${{ github.repository_owner }}
          VERSION: ${{ github.ref_name }}
        run: |
          cd fvdb-docker
          ./build-registry.sh
```

### GitLab CI Example

```yaml
build:
  image: docker:latest
  services:
    - docker:dind
  script:
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
    - export REGISTRY=$CI_REGISTRY_IMAGE
    - export VERSION=$CI_COMMIT_TAG
    - cd fvdb-docker
    - ./build-registry.sh
  only:
    - tags
```

---

## 🐛 Troubleshooting

### Build Fails

```bash
# Check logs
docker buildx ls
docker buildx inspect fvdb-builder

# Clean and rebuild
docker buildx rm fvdb-builder
./build-registry.sh
```

### Push Fails (Authentication)

```bash
# Check login
docker login $REGISTRY

# Test push
docker push $REGISTRY/test:latest
```

### ARM64 Build Fails

```bash
# Disable ARM64, build AMD64 only
export BUILD_ARM64=false
./build-registry.sh
```

---

## 📈 Best Practices

### 1. Use Semantic Versioning

```bash
export VERSION=1.0.0  # Not just "latest"
```

### 2. Tag Latest AND Version

```bash
# Script automatically tags both:
# - $REGISTRY/fvdb-training:1.0.0
# - $REGISTRY/fvdb-training:latest
```

### 3. Test Before Sharing

```bash
# Build
./build-registry.sh

# Pull and test
docker pull $REGISTRY/fvdb-training:$VERSION
docker run --rm --gpus all $REGISTRY/fvdb-training:$VERSION python -c "import fvdb; print('OK')"
```

### 4. Document Your Registry

Add to your README:
```markdown
## Using Pre-Built Images

```bash
docker pull your-registry.com/fvdb-training:1.0.0
docker compose -f docker-compose.registry.yml up
```
```

---

## 🎓 Advanced: Multi-Stage Builds

For ARM64 support, use multi-stage builds:

```dockerfile
# Stage 1: Build wheels on AMD64
FROM --platform=linux/amd64 nvidia/cuda:12.6.0-devel-ubuntu24.04 AS wheel-builder
# ... build fVDB wheels ...

# Stage 2: Copy to ARM64
FROM nvidia/cuda:12.6.0-runtime-ubuntu24.04
COPY --from=wheel-builder /wheels/*.whl /tmp/
RUN pip install /tmp/*.whl
```

---

## ✅ Verification

After pushing to registry:

```bash
# On different machine
export REGISTRY=your-registry.com
export VERSION=1.0.0

# Pull
docker pull $REGISTRY/fvdb-training:$VERSION

# Verify
docker run --rm $REGISTRY/fvdb-training:$VERSION python -c "
import torch
import fvdb
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'fVDB: {fvdb.__version__}')
"
```

Expected output:
```
PyTorch: 2.9.0+cu126
CUDA: True
fVDB: 0.3.x
```

---

## 📊 Registry Comparison

| Registry | Free Tier | Private | Public | GPU Support |
|----------|-----------|---------|--------|-------------|
| **Docker Hub** | 1 private | Yes | Yes | Yes |
| **GitHub Container Registry** | Unlimited | Yes | Yes | Yes |
| **Google GCR** | 5GB free | Yes | Yes | Yes |
| **AWS ECR** | 500MB free | Yes | Limited | Yes |
| **NVIDIA NGC** | Free | Yes | Yes | Optimized ✅ |

---

## 🆘 Support

For registry-specific issues:
- Docker Hub: https://hub.docker.com/support
- GitHub: https://docs.github.com/packages
- Google: https://cloud.google.com/container-registry/docs
- AWS: https://docs.aws.amazon.com/ecr/
- NGC: https://docs.nvidia.com/ngc/

For fVDB issues:
- https://fvdb.ai/
- https://github.com/fvdb

---

**Status:** ✅ **AMD64 images work and are production-ready for registry sharing!**
