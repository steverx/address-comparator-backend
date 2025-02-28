FROM python:3.9-slim

WORKDIR /app

# Install essential build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends grep && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies (filtering out postal)
COPY requirements.txt .
RUN grep -v "postal" requirements.txt > requirements_filtered.txt && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements_filtered.txt && \
    pip install --no-cache-dir waitress psutil

# Copy application code
COPY . .

# Simple, direct command using waitress
ENV PORT=5000
CMD ["python", "wsgi.py"]