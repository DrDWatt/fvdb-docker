# Known Issues and Limitations

## ARM64 Build Issues

### Issue: fVDB Reality Capture Dependencies Fail to Compile

**Symptoms:**
- `point-cloud-utils` fails to build wheel
- `pye57` fails to compile
- Build error: "Could not build wheels for point-cloud-utils, pye57"

**Root Cause:**
- These packages have C++ code that needs compilation
- No pre-built wheels available for ARM64
- Requires Eigen3, Boost, and other C++ libraries

**Status:** IN PROGRESS - Adding build dependencies

**Workaround:**
- Use x86_64 containers (have pre-built wheels)
- Or build on x86 and push to registry, then pull on ARM64

## Temporary Solution

For now, the Docker images work best on **x86_64/AMD64** architecture.

ARM64 support coming soon once dependency build issues are resolved.

## Alternative Approaches

### Option 1: Multi-Stage Build
Build dependencies on x86, copy to ARM64

### Option 2: Pre-Built Wheels
Build wheels separately and include in image

### Option 3: Simplified Container
Create training-only container without full fVDB Reality Capture

---

**Update:** Adding Eigen3, Boost, and TBB to resolve compilation issues...
