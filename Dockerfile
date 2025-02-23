FROM python:3.9-slim

WORKDIR /app

# Add the debian backports repository
RUN echo "deb http://deb.debian.org/debian bookworm-backports main" > /etc/apt/sources.list.d/bookworm-backports.list

# Update apt and install system dependencies, including libpostal-dev from backports
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        python3-dev \
        libpq-dev \
        curl \
    && apt-get install -y -t bookworm-backports libpostal-dev \
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

# Set environment variables (consolidated)
ENV PYTHONUNBUFFERED=1 \
    FLASK_ENV=production \
    PORT=5000 \
    GUNICORN_CMD_ARGS="--workers=4 --threads=8 --timeout=120 --bind=0.0.0.0:5000 --worker-class=gthread --max-requests=1000 --max-requests-jitter=50 --graceful-timeout=30 --keep-alive=5"

# Create start script with enhanced logging
RUN echo '#!/bin/bash\n\
echo "Starting Gunicorn server..."\n\
echo "Environment: $FLASK_ENV"\n\
echo "Port: $PORT"\n\
echo "Python version: $(python --version)"\n\
echo "Memory limit: $(cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo N/A)"\n\
echo "CPU quota: $(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null || echo N/A)"\n\
exec gunicorn \
--bind 0.0.0.0:$PORT \
--workers 4 \
--threads 8 \
--timeout 120 \
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

# Health check with improved logging and error handling
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || (echo "Health check failed at $(date)" >> /tmp/health.log && exit 1)

# Single CMD instruction
CMD ["/app/start.sh"]