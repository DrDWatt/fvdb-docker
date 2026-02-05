# SVO to ROSBAG Converter Service
# Uses ZED SDK for SVO file processing and ROS for ROSBAG creation

FROM nvcr.io/nvidia/l4t-jetpack:r36.2.0 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    curl \
    wget \
    git \
    libusb-1.0-0 \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY svo_converter_service.py .

RUN mkdir -p /app/uploads /app/outputs /app/processing

ENV UPLOAD_DIR=/app/uploads
ENV OUTPUT_DIR=/app/outputs
ENV PROCESSING_DIR=/app/processing
ENV PORT=8009

EXPOSE 8009

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8009/health || exit 1

CMD ["python3", "-m", "uvicorn", "svo_converter_service:app", "--host", "0.0.0.0", "--port", "8009"]
