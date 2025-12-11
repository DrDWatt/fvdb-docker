# iPhone 16 Recording Guide for 3D Gaussian Splatting

## Camera Settings

### In Camera App:
1. **Format**: Settings → Camera → Formats → "High Efficiency" (HEVC)
2. **Resolution**: 4K at 30fps (best quality/performance balance)
   - Avoid 60fps - too many similar frames
   - 24fps also works well
3. **Stabilization**: Keep ON (helps with smooth motion)
4. **HDR**: Auto HDR ON (better dynamic range)
5. **Grid**: Enable 3x3 grid (helps with framing)

### Alternative - Filmic Pro App (Better Control):
- Manual focus lock
- Manual exposure lock
- 4K 24fps or 30fps
- Flat color profile for better feature detection

## Recording Technique

### Camera Movement:
1. **SLOW deliberate movements** - move 10-20cm, pause 1 second, repeat
2. **Overlap**: Each frame should overlap 60-80% with previous frame
3. **Distance**: Stay 1-2 meters from subject
4. **Duration**: 60-90 seconds for small object, 2-3 minutes for room
5. **Orbit pattern**: Move in a smooth circle/arc around subject

### Lighting:
- **Bright, diffuse lighting** (avoid harsh shadows)
- **Consistent lighting** (don't record at different times of day)
- **Avoid windows** in frame (causes exposure issues)

### Subject Requirements:
- **Textured surfaces** (avoid plain white walls, glass, mirrors)
- **Static scene** (no moving objects, people)
- **Well-defined features** (good for COLMAP feature matching)

### What to Avoid:
❌ Fast pans or tilts
❌ Quick movements
❌ Zooming in/out
❌ Low light or backlit scenes
❌ Reflective/shiny surfaces
❌ Repetitive patterns (tiles, wallpaper)

## Example Shot Plan for Kitchen:

1. Start at entrance, pan slowly right
2. Move forward 30cm, pause, continue pan
3. Circle around island/counter (small steps)
4. Get different heights (crouch, stand, slight angles)
5. Capture 120-180 seconds total
6. Result: ~150-250 distinct frames at 1 FPS extraction

## Processing Settings for Your Workflow:

- **Extraction FPS**: 1.0 (one frame per second)
- **Matcher**: Exhaustive (for <100 frames)
- **Camera Model**: SIMPLE_RADIAL (works for iPhone)
- **Expected**: 40-80 registered images for good reconstruction

