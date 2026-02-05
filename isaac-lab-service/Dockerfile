# ISAAC Lab Service
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY isaac_lab_service.py .

RUN mkdir -p /app/checkpoints /app/logs /app/outputs

ENV CHECKPOINT_DIR=/app/checkpoints
ENV LOG_DIR=/app/logs
ENV OUTPUT_DIR=/app/outputs
ENV PORT=8011

EXPOSE 8011

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8011/health || exit 1

CMD ["python", "-m", "uvicorn", "isaac_lab_service:app", "--host", "0.0.0.0", "--port", "8011"]
