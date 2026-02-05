# ISAAC Sim Service
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY isaac_sim_service.py .

RUN mkdir -p /app/rosbags /app/scenes /app/outputs

ENV ROSBAG_DIR=/app/rosbags
ENV SCENE_DIR=/app/scenes
ENV OUTPUT_DIR=/app/outputs
ENV PORT=8010

EXPOSE 8010

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8010/health || exit 1

CMD ["python", "-m", "uvicorn", "isaac_sim_service:app", "--host", "0.0.0.0", "--port", "8010"]
