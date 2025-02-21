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

# Upgrade pip and install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV RAILWAY_ENVIRONMENT=production

# Make the port available
EXPOSE 8080

# Start command
CMD exec gunicorn --bind 0.0.0.0:$PORT \
    --workers 4 \
    --threads 8 \
    --timeout 0 \
    --preload \
    --worker-class=gthread \
    "app:create_app()"