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
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies and explicitly install gunicorn
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install gunicorn==21.2.0

# Copy application code
COPY . .

# Set environment variables
ENV PORT=8080
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Create start script with enhanced logging
RUN echo '#!/bin/bash\n\
echo "Starting Gunicorn server..."\n\
echo "Current directory: $(pwd)"\n\
echo "Python path: $(which python)"\n\
echo "Files in current directory: $(ls -la)"\n\
exec gunicorn --bind 0.0.0.0:$PORT \
--workers 4 \
--threads 8 \
--timeout 0 \
--log-level debug \
--access-logfile - \
--error-logfile - \
--preload \
wsgi:application' > start.sh && \
chmod +x start.sh

# Expose the port
EXPOSE 8080

# Run the start script
CMD ["./start.sh"]