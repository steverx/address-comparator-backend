FROM python:3.9-slim

WORKDIR /app

# Install essential build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends grep && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies (filtering out BOTH postal AND gunicorn)
COPY requirements.txt .
RUN grep -v -E "postal|gunicorn" requirements.txt > requirements_filtered.txt && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements_filtered.txt && \
    pip install --no-cache-dir waitress psutil

# Copy application code
COPY . .

# Explicitly set CMD to use python and wsgi.py
ENV PORT=5000
CMD ["python", "wsgi.py"]