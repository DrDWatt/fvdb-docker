#!/bin/bash
# Extract images from Spectacular Rec recording for COLMAP processing
#
# Usage: ./spectacular-to-colmap.sh <recording.zip> [output_dir]
#
# This extracts images from a Spectacular Rec recording and prepares
# them for the existing COLMAP workflow.

set -e

RECORDING_ZIP="$1"
OUTPUT_DIR="${2:-./extracted_images}"

if [ -z "$RECORDING_ZIP" ]; then
    echo "Usage: $0 <recording.zip> [output_dir]"
    echo ""
    echo "Extracts images from Spectacular Rec app recording for COLMAP processing."
    echo ""
    echo "After extraction, upload the images to:"
    echo "  - http://localhost:8000/workflow (zip the output folder)"
    echo "  - Or use: curl -X POST http://localhost:8003/upload -F 'file=@images.zip'"
    exit 1
fi

if [ ! -f "$RECORDING_ZIP" ]; then
    echo "Error: File not found: $RECORDING_ZIP"
    exit 1
fi

echo "📱 Spectacular Rec → COLMAP Converter"
echo "======================================"
echo ""

# Create temp extraction dir
TEMP_DIR=$(mktemp -d)
echo "📦 Extracting recording..."
unzip -q "$RECORDING_ZIP" -d "$TEMP_DIR"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Find and copy images
echo "🔍 Looking for images..."
IMAGE_COUNT=0

# Spectacular Rec stores images in various locations
for pattern in "*.jpg" "*.jpeg" "*.png" "frames/*.jpg" "frames/*.png" "images/*.jpg" "images/*.png"; do
    for img in $(find "$TEMP_DIR" -name "$(basename "$pattern")" -type f 2>/dev/null); do
        cp "$img" "$OUTPUT_DIR/"
        ((IMAGE_COUNT++))
    done
done

# Also check for video files to extract frames
VIDEO_FILE=$(find "$TEMP_DIR" -name "*.mp4" -o -name "*.mov" | head -1)
if [ -n "$VIDEO_FILE" ] && [ "$IMAGE_COUNT" -eq 0 ]; then
    echo "📹 Found video, extracting frames at 2 FPS..."
    ffmpeg -i "$VIDEO_FILE" -vf "fps=2" -q:v 2 "$OUTPUT_DIR/frame_%04d.jpg" 2>/dev/null
    IMAGE_COUNT=$(ls -1 "$OUTPUT_DIR"/*.jpg 2>/dev/null | wc -l)
fi

# Cleanup
rm -rf "$TEMP_DIR"

echo ""
echo "✅ Extracted $IMAGE_COUNT images to: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. Create zip: cd $OUTPUT_DIR && zip -r ../images.zip ."
echo "  2. Upload to workflow: http://localhost:8000/workflow"
echo "     Or API: curl -X POST http://localhost:8003/upload -F 'file=@images.zip'"
echo ""
