# Depth Estimation Service using Depth Anything V2
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY depth_service.py .

ENV PORT=8013
EXPOSE 8013

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8013/health || exit 1

CMD ["python", "-m", "uvicorn", "depth_service:app", "--host", "0.0.0.0", "--port", "8013"]
