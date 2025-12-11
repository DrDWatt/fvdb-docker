# COLMAP Build Fix

## Problem
Build failed with:
```
CMake Error: CMake was unable to find a build program corresponding to "Ninja"
CMAKE_C_COMPILER not set
CMAKE_CXX_COMPILER not set
```

## Root Cause
1. `ninja-build` package was missing from apt-get install
2. CMake syntax was incorrect: `-GNinja` should be `-G Ninja` (with space)

## Fix Applied

### 1. Added ninja-build to dependencies
```dockerfile
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    ninja-build \    # ← ADDED THIS
    git \
    ...
```

### 2. Fixed CMake generator syntax
```dockerfile
# Before:
cmake .. -GNinja \

# After:
cmake .. -G Ninja \    # ← Space between -G and Ninja
```

## Build Progress

The build is now running successfully and will take 10-15 minutes to:
1. Download COLMAP source (3.9.1)
2. Configure with CMake
3. Compile with Ninja (fast parallel build)
4. Install to /usr/local
5. Install Python dependencies
6. Create service

Monitor with:
```bash
tail -f /tmp/colmap-build.log
```

Look for:
```
[1/XXX] Building...
[XXX/XXX] Linking...
Installing...
```

## Expected Timeline
- Install system deps: ~1-2 min (done)
- Clone COLMAP: ~5 sec (done)
- Configure with CMake: ~30 sec
- Compile with Ninja: ~8-12 min (in progress)
- Install: ~30 sec
- Python deps: ~1-2 min
- Finalize: ~30 sec

**Total: 10-15 minutes**

## Success Indicator
```
Successfully tagged colmap-service:latest
```

Then you can start with:
```bash
docker compose -f docker-compose.workflow.yml up -d
```

And access:
```
http://localhost:8080/workflow
```
