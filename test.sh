#!/bin/bash
# Test fVDB Docker services

set -e

echo "======================================================================"
echo "Testing fVDB Docker Services"
echo "======================================================================"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# Check if services are running
if ! docker compose ps | grep -q "Up"; then
    echo "❌ Services not running. Starting..."
    docker compose up -d
    echo "⏳ Waiting for services to start..."
    sleep 10
fi

echo ""
echo "🔍 Testing Training Service..."
echo ""

# Test health endpoint
echo -n "  Health check... "
if curl -sf http://localhost:8000/health > /dev/null; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    exit 1
fi

# Test tutorials endpoint
echo -n "  Tutorials endpoint... "
if curl -sf http://localhost:8000/tutorials > /dev/null; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    exit 1
fi

# Test datasets list
echo -n "  Datasets list... "
if curl -sf http://localhost:8000/datasets > /dev/null; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    exit 1
fi

# Test jobs list
echo -n "  Jobs list... "
if curl -sf http://localhost:8000/jobs > /dev/null; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    exit 1
fi

echo ""
echo "🔍 Testing Rendering Service..."
echo ""

# Test health endpoint
echo -n "  Health check... "
if curl -sf http://localhost:8001/health > /dev/null; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    exit 1
fi

# Test tutorials endpoint
echo -n "  Tutorials endpoint... "
if curl -sf http://localhost:8001/tutorials > /dev/null; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    exit 1
fi

# Test models list
echo -n "  Models list... "
if curl -sf http://localhost:8001/models > /dev/null; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    exit 1
fi

# Test Swagger UI
echo -n "  Swagger UI (training)... "
if curl -sf http://localhost:8000/ > /dev/null; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
fi

echo -n "  Swagger UI (rendering)... "
if curl -sf http://localhost:8001/api > /dev/null; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
fi

echo ""
echo "🖥️  GPU Status:"
echo ""
docker compose exec -T training nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader 2>/dev/null || echo "  GPU not available or not accessible"

echo ""
echo "======================================================================"
echo -e "${GREEN}✅ All Tests Passed!${NC}"
echo "======================================================================"
echo ""
echo "🌐 Access services:"
echo "   Training:  http://localhost:8000 (Swagger UI)"
echo "   Rendering: http://localhost:8001"
echo ""
echo "📚 Tutorial links available at both:"
echo "   http://localhost:8000/tutorials"
echo "   http://localhost:8001/tutorials"
echo ""
