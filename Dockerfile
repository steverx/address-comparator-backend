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
    pip install --no-cache-dir -r requirements.txt && \
    pip install flask-cors==4.0.0  # Explicitly install flask-cors

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV RAILWAY_ENVIRONMENT=production
ENV GUNICORN_TIMEOUT=120
ENV GUNICORN_WORKERS=4

# Make the port available
EXPOSE 8000

# Start command with explicit bind
CMD gunicorn \
    --bind "0.0.0.0:$PORT" \
    --workers $GUNICORN_WORKERS \
    --timeout $GUNICORN_TIMEOUT \
    --log-level debug \
    --access-logfile - \
    --error-logfile - \
    --capture-output \
    "app:create_app()"