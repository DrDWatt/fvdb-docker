# Multi-Architecture Build Guide
**fVDB Reality Capture - ARM64 & x86_64**

---

## Registry Configuration

**Local Registry:** `localhost:7000`

```bash
# Registry is running at:
docker ps | grep registry

# View registry contents:
curl http://localhost:7000/v2/_catalog | python3 -m json.tool
```

---

## Multi-Architecture Support

### Current System
- **Native:** ARM64 (aarch64)
- **Emulated:** x86_64 (amd64) via QEMU

### QEMU Emulation Installed ✅
```bash
# Emulated platforms available:
- linux/amd64 (x86_64)
- linux/arm64 (native)
- linux/arm/v7
- linux/386
- And more...
```

---

## Building Multi-Arch Images

### Quick Build & Push

```bash
cd ~/fvdb-docker
./build-multiarch.sh
```

This will:
1. Build for `linux/amd64` (x86_64) and `linux/arm64`
2. Push to `localhost:7000/fvdb-training:latest`
3. Push to `localhost:7000/fvdb-rendering:latest`

### Manual Build

```bash
# Training service
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --tag localhost:7000/fvdb-training:latest \
  --push \
  -f training-service/Dockerfile.host \
  training-service/

# Rendering service
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --tag localhost:7000/fvdb-rendering:latest \
  --push \
  -f rendering-service/Dockerfile.host \
  rendering-service/
```

---

## Using the Images

### On ARM64 System (Current)

```bash
docker pull localhost:7000/fvdb-training:latest
docker pull localhost:7000/fvdb-rendering:latest

# Or use docker-compose
docker compose -f docker-compose.host.yml up -d
```

### On x86_64 System

```bash
# Same commands! Docker auto-selects correct architecture
docker pull localhost:7000/fvdb-training:latest
docker pull localhost:7000/fvdb-rendering:latest
```

---

## Registry Management

### Start Registry

```bash
docker run -d -p 7000:5000 --restart=always --name registry registry:2
```

### Stop Registry

```bash
docker stop registry
docker rm registry
```

### View Registry Contents

```bash
curl http://localhost:7000/v2/_catalog
```

### Remove Image from Registry

```bash
# Delete manifest
curl -X DELETE http://localhost:7000/v2/fvdb-training/manifests/latest
```

---

## Troubleshooting

### Builder Shows Only ARM64

```bash
# Recreate builder to pick up QEMU
docker buildx rm multiarch
docker buildx create --name multiarch --use --bootstrap
docker buildx inspect multiarch | grep Platforms
```

Should show: `linux/arm64, linux/amd64, ...`

### QEMU Not Working

```bash
# Reinstall QEMU emulation
docker run --privileged --rm tonistiigi/binfmt --install all
```

### Registry Connection Refused

```bash
# Check registry is running
docker ps | grep registry

# Restart registry
docker restart registry
```

### Build Fails on x86_64

Check if using host-specific paths (conda env, fvdb-reality-capture).
These Dockerfiles mount host paths that may not exist on other systems.

**Solution:** For portable builds, use self-contained Dockerfiles without host mounts.

---

## Image Tags

### Tagging Strategy

```bash
# Latest (rolling)
localhost:7000/fvdb-training:latest

# Version tags
localhost:7000/fvdb-training:v1.0.0

# Date tags
localhost:7000/fvdb-training:2025-11-09
```

### Adding Tags

```bash
# Tag existing image
docker tag localhost:7000/fvdb-training:latest localhost:7000/fvdb-training:v1.0.0

# Push new tag
docker push localhost:7000/fvdb-training:v1.0.0
```

---

## Performance Notes

### Build Times

**Native ARM64:**
- Training service: ~2-3 minutes
- Rendering service: ~2-3 minutes

**Emulated x86_64:**
- Training service: ~10-15 minutes (slower due to QEMU)
- Rendering service: ~10-15 minutes

**Parallel build:** Both architectures build simultaneously.

### Registry Storage

- Each multi-arch manifest: ~500 MB per service
- Both architectures stored: ~1 GB total per service
- Registry uses Docker's storage driver

---

## Network Access

### Within Same Network

```bash
# Other machines on network can access:
http://YOUR_IP:7000/v2/_catalog

# Pull from network:
docker pull YOUR_IP:7000/fvdb-training:latest
```

### Configure Insecure Registry

On client machines, add to `/etc/docker/daemon.json`:

```json
{
  "insecure-registries": ["YOUR_IP:7000"]
}
```

Then restart Docker:
```bash
sudo systemctl restart docker
```

---

## Best Practices

### 1. Always Use Multi-Arch

Build for both platforms even if you only need one now.
Future portability is worth the extra build time.

### 2. Tag Everything

Use semantic versioning and date tags in addition to `latest`.

### 3. Test Both Architectures

```bash
# Test ARM64 (native)
docker run --rm localhost:7000/fvdb-training:latest python3 -c "import platform; print(platform.machine())"

# Test x86_64 (emulated)
docker run --rm --platform linux/amd64 localhost:7000/fvdb-training:latest python3 -c "import platform; print(platform.machine())"
```

### 4. Clean Up Old Images

```bash
# Remove unused images
docker image prune -a

# Clean buildx cache
docker buildx prune
```

---

## Summary

✅ **Registry:** Running at localhost:7000
✅ **QEMU:** Installed for x86_64 emulation  
✅ **Builder:** Supports linux/amd64 + linux/arm64
✅ **Script:** `build-multiarch.sh` ready to use
✅ **Images:** Will work on both ARM64 and x86_64 systems

**Ready to build!**

```bash
cd ~/fvdb-docker
./build-multiarch.sh
```
