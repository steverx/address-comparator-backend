# Use a consistent name for the builder stage
FROM steverx/libpostal-builder:latest as libpostal-builder

# --- Final Backend Image ---
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        nginx \
        ca-certificates \
        gosu \
        libpq-dev \
        curl \
        grep \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m -d /home/appuser -s /bin/bash appuser

# Copy libpostal files from the builder stage
COPY --from=libpostal-builder --chown=appuser:appuser /usr/local/lib/libpostal.so* /usr/local/lib/
COPY --from=libpostal-builder --chown=appuser:appuser /usr/local/include/libpostal/ /usr/local/include/libpostal/
COPY --from=libpostal-builder --chown=appuser:appuser /usr/local/data/ /usr/local/data/

# Set environment variables
ENV LIBPOSTAL_INCLUDE_DIR=/usr/local/include
ENV LIBPOSTAL_LIB_DIR=/usr/local/lib
ENV LIBPOSTAL_DATA_DIR=/usr/local/data
ENV LD_LIBRARY_PATH="${LIBPOSTAL_LIB_DIR}:${LD_LIBRARY_PATH}"
ENV PORT=5000

# Run ldconfig to update shared libraries
RUN ldconfig

# Copy requirements and install Python dependencies
COPY requirements.txt .
# Create a filtered requirements file without postal
RUN grep -v "postal" requirements.txt > requirements_filtered.txt && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir setuptools wheel && \
    pip install --no-cache-dir -r requirements_filtered.txt

# Install waitress instead of gunicorn
RUN pip install --no-cache-dir waitress requests flask

# Copy application code (with correct ownership)
COPY --chown=appuser:appuser . .

# Ensure Python module structure is correctly set up
RUN mkdir -p utils api tasks && \
    touch utils/__init__.py api/__init__.py tasks/__init__.py

# Explicitly create module directories if they don't exist
RUN mkdir -p /app/api /app/config /app/utils /app/tasks /app/tests && \
    touch /app/api/__init__.py /app/config/__init__.py /app/utils/__init__.py /app/tasks/__init__.py /app/tests/__init__.py

# Copy nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Use waitress instead of gunicorn
CMD sh -c "nginx -g 'daemon on;' && cd /app && python -m waitress --port=${PORT} --call app:create_app"