# COLMAP Build - Final Fix

## Issue #3 - Source Code Bug ❌

**Error**: Missing `#include <memory>` header in COLMAP source
```cpp
/tmp/colmap/src/colmap/image/line.cc:63:8: error: 'unique_ptr' is not a member of 'std'
note: 'std::unique_ptr' is defined in header '<memory>'; did you forget to '#include <memory>'?
```

**Root Cause**: COLMAP 3.9.1 has a C++ compilation bug on certain compilers where the `<memory>` header is not included in `src/colmap/image/line.cc`.

**Fix Applied**: Patch the source code after cloning
```dockerfile
sed -i '35a#include <memory>' src/colmap/image/line.cc
```

This adds `#include <memory>` after line 35 (after the LSD header include).

---

## Build Attempt History

| Attempt | Issue | Fix |
|---------|-------|-----|
| #1 | Ninja not found | Added `ninja-build` package |
| #2 | FLANN library missing | Added `libflann-dev` package |
| #3 | Missing header in source | Patched source code with `sed` |
| **#4** | **Building...** | **Should work now** ✅ |

---

## What Changed

### Before
```dockerfile
RUN git clone --depth 1 --branch 3.9.1 https://github.com/colmap/colmap.git && \
    cd colmap && \
    mkdir build && cd build && \
    cmake .. -G Ninja \
```

### After
```dockerfile
RUN git clone --depth 1 --branch 3.9.1 https://github.com/colmap/colmap.git && \
    cd colmap && \
    sed -i '35a#include <memory>' src/colmap/image/line.cc && \  ← ADDED THIS
    mkdir build && cd build && \
    cmake .. -G Ninja \
```

---

## Monitoring

```bash
# Watch build (will take 10-15 min)
tail -f /tmp/colmap-build-v3.log

# Look for progress
tail -f /tmp/colmap-build-v3.log | grep "\[.*\]"

# Check for completion
tail -f /tmp/colmap-build-v3.log | grep "Successfully"
```

---

## Expected Timeline

- Clone COLMAP: 2 sec (done)
- Patch source: <1 sec
- Configure CMake: ~1 min (done)
- **Compile (154 files)**: **~10-12 min** ⏳
- Install: ~30 sec
- Python deps: ~1-2 min
- Finalize: ~30 sec

**Total**: ~12-15 minutes

---

## Progress Indicators

### Compilation Phase
```
[1/154] Building CXX object ...
[50/154] Building CXX object ...  ← Failed here before
[100/154] Building CXX object ...
[154/154] Linking CXX executable bin/colmap
```

### Installation Phase
```
Install the project...
-- Install configuration: "Release"
-- Installing: /usr/local/bin/colmap
-- Installing: /usr/local/lib/libcolmap.a
```

### Success
```
Successfully tagged colmap-service:latest
```

---

## Why This Happened

COLMAP 3.9.1 was compiled on compilers that implicitly include `<memory>` through other headers. On newer GCC versions (like 13.3.0 in Ubuntu 24.04), the implicit inclusion doesn't happen, causing the compilation error.

This is a known issue and was likely fixed in later COLMAP versions, but 3.9.1 is a stable release that works well once patched.

---

## After Build Completes

```bash
# Verify COLMAP binary exists
docker run --rm colmap-service:latest colmap -h

# Start the workflow
docker compose -f docker-compose.workflow.yml up -d

# Open workflow page
open http://localhost:8080/workflow

# Test COLMAP service
curl http://localhost:8003/health
```

---

## Alternative Approach (if this fails)

Instead of building from source, use apt package:
```dockerfile
RUN apt-get update && apt-get install -y colmap
```

But this may be an older version. Building from source gives us the latest stable release with all features.

---

**Status**: Build restarted with patch applied  
**ETA**: 12-15 minutes  
**Confidence**: 🟢 High - This should be the final issue
