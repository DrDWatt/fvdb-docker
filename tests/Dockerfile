# Test runner for the splat viewers (:8085 fvdb-viewer, :8086 supersplat-viewer).
# Python (pytest) for service/API tests, Node for the frontend unit tests.
# The repository is bind-mounted read-only at /repo by docker-compose.test.yml.
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends nodejs curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

WORKDIR /repo
CMD ["sh", "-c", "pytest -m 'not integration' && node --test tests/js/"]
