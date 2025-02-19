FROM python:3.9-slim

WORKDIR /app

# Install minimal required packages
RUN pip install flask==3.1.0 gunicorn==23.0.0 psutil==7.0.0

# Copy application code
COPY app.py .

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