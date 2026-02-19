# ISAAC Viewer Service - Refactored with SVO/SAM2/GARField support
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    curl \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY isaac_viewer_service.py .

RUN mkdir -p /app/svo /app/rosbags /app/outputs /app/frames

ENV SVO_DIR=/app/svo
ENV ROSBAG_DIR=/app/rosbags
ENV OUTPUT_DIR=/app/outputs
ENV FRAME_DIR=/app/frames
ENV PORT=8012

EXPOSE 8012

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8012/health || exit 1

CMD ["python", "-m", "uvicorn", "isaac_viewer_service:app", "--host", "0.0.0.0", "--port", "8012"]
