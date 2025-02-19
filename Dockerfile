FROM python:3.9-slim

WORKDIR /app

# Install minimal required packages
RUN pip install flask gunicorn psutil

# Copy application code
COPY app.py .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV FLASK_ENV=production

# Start with explicit logging
CMD gunicorn --bind 0.0.0.0:$PORT \
    --log-level debug \
    --access-logfile - \
    --error-logfile - \
    --capture-output \
    "app:create_app()"