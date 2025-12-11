FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \
    numpy==1.26.0 \
    opencv-python-headless==4.9.0.80 \
    pillow==10.4.0 \
    plyfile==1.0.3 \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    pydantic==2.5.0 \
    aiofiles==23.2.1

# Create workspace
WORKDIR /workspace
RUN mkdir -p /workspace/app /workspace/data/models /workspace/data/outputs

# Copy application
COPY app/ /workspace/app/

# Expose port for web API
EXPOSE 8002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

CMD ["python3", "/workspace/app/convert_and_render.py"]
