FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    gcc \
    python3-dev \
    libpq-dev \
    python3-pip \
    python3-setuptools \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install gunicorn==21.2.0

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production
ENV PORT=5000
ENV GUNICORN_CMD_ARGS="--workers=4 --threads=8 --timeout=30 --bind=0.0.0.0:5000"

# Create start script with health check logging
RUN echo '#!/bin/bash\n\
echo "Starting Gunicorn server..."\n\
echo "Environment: $FLASK_ENV"\n\
echo "Port: $PORT"\n\
echo "Python version: $(python --version)"\n\
exec gunicorn \
--bind 0.0.0.0:$PORT \
--workers 4 \
--threads 8 \
--timeout 30 \
--log-level info \
--access-logfile - \
--error-logfile - \
--preload \
--worker-class gthread \
wsgi:app' > /app/start.sh && \
chmod +x /app/start.sh

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

# Start command
CMD ["/app/start.sh"]