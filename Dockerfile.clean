FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir waitress psutil

# Copy application code
COPY . .

# Simple, direct command - explicitly avoid gunicorn
ENV PORT=5000
CMD ["python", "wsgi.py"]