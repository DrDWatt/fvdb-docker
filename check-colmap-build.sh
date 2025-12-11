#!/bin/bash
# Monitor COLMAP build progress

echo "╔══════════════════════════════════════════════════════════╗"
echo "║           COLMAP SOURCE BUILD MONITOR                    ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

if [ ! -f /tmp/colmap-build.log ]; then
    echo "❌ Build log not found"
    exit 1
fi

# Check if build is still running
if docker ps -a | grep -q "colmap-service"; then
    BUILD_RUNNING=$(ps aux | grep "docker build.*colmap-service" | grep -v grep)
    if [ -n "$BUILD_RUNNING" ]; then
        echo "🔨 Build Status: IN PROGRESS"
    else
        echo "✅ Build Status: COMPLETED or STOPPED"
    fi
else
    echo "🔨 Build Status: RUNNING"
fi

echo ""
echo "📊 Recent Progress:"
echo "─────────────────────────────────────────────────────────"
tail -20 /tmp/colmap-build.log | grep -E "(Step|#|Cloning|cmake|ninja|Building|COLMAP|Error|error)" || tail -20 /tmp/colmap-build.log
echo ""
echo "─────────────────────────────────────────────────────────"
echo ""
echo "⏱️  Estimated time remaining:"
LINES=$(wc -l < /tmp/colmap-build.log)
if [ $LINES -lt 100 ]; then
    echo "   ~60-80 minutes (just started)"
elif [ $LINES -lt 500 ]; then
    echo "   ~50-70 minutes (downloading/installing deps)"
elif [ $LINES -lt 2000 ]; then
    echo "   ~30-50 minutes (compiling COLMAP)"
elif [ $LINES -lt 5000 ]; then
    echo "   ~10-20 minutes (finishing up)"
else
    echo "   Should be done soon or completed!"
fi

echo ""
echo "📝 Full log: /tmp/colmap-build.log"
echo "💡 Run this script again to check progress"
echo ""
echo "Commands:"
echo "  • Watch build: tail -f /tmp/colmap-build.log"
echo "  • Check status: bash check-colmap-build.sh"
