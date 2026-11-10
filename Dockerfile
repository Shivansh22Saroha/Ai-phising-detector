# Lean serving image: slim base + serve-only requirements (no MLflow, pandas, pytest)
FROM python:3.11-slim

# No .pyc files, unbuffered logs (shows up immediately in `docker logs`)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# Install dependencies first: this layer is cached until requirements change
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# Only what the server needs at runtime
COPY app.py features.py ./
COPY model/ ./model/
COPY static/ ./static/

# Run as a non-root user; it owns the logs folder for prediction logging
RUN useradd --create-home appuser && mkdir -p /app/logs && chown -R appuser /app/logs
USER appuser

EXPOSE 8000

# Docker restarts/flags the container if /health stops answering
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health', timeout=4)"

# Shell form so $PORT expands (Render/Railway inject their own PORT)
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT}
