FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y gcc python3-dev libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV RAILWAY_ENVIRONMENT=production
ENV FLASK_DEBUG=0

# Make the port available
EXPOSE 8000

# Start command with explicit bind
CMD exec gunicorn --bind "0.0.0.0:$PORT" \
    --workers 4 \
    --timeout 120 \
    --log-level debug \
    --access-logfile - \
    --error-logfile - \
    --capture-output \
    "app:create_app()"