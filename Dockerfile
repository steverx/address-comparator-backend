FROM python:3.9-slim

WORKDIR /app

# Install system dependencies more efficiently
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        python3-dev \
        libpq-dev \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN useradd -m appuser && \
    chown -R appuser:appuser /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install gunicorn==21.2.0 && \
    pip cache purge

# Copy application code
COPY --chown=appuser:appuser . .

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    FLASK_ENV=production \
    PORT=5000 \
    GUNICORN_CMD_ARGS="--workers=4 --threads=8 --timeout=60 --bind=0.0.0.0:5000 --worker-class=gthread"

# Create start script with enhanced logging
RUN echo '#!/bin/bash\n\
echo "Starting Gunicorn server..."\n\
echo "Environment: $FLASK_ENV"\n\
echo "Port: $PORT"\n\
echo "Python version: $(python --version)"\n\
echo "Memory limit: $(cat /sys/fs/cgroup/memory/memory.limit_in_bytes)"\n\
echo "CPU quota: $(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us)"\n\
exec gunicorn \
--bind 0.0.0.0:$PORT \
--workers 4 \
--threads 8 \
--timeout 60 \
--log-level info \
--access-logfile - \
--error-logfile - \
--preload \
--worker-class gthread \
--max-requests 1000 \
--max-requests-jitter 50 \
--graceful-timeout 30 \
--keep-alive 5 \
wsgi:app' > /app/start.sh && \
chmod +x /app/start.sh

# Switch to non-root user
USER appuser

# Health check with improved logging
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || (echo "Health check failed" && exit 1)

# Start command
CMD ["/app/start.sh"]