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
    pip install --no-cache-dir -r requirements_filtered.txt && \
    pip install --no-cache-dir gunicorn requests flask

# Insert in your Dockerfile where you install Python packages
RUN pip install --no-cache-dir gunicorn

# Copy application code (with correct ownership)
COPY --chown=appuser:appuser . .

# Ensure Python module structure is correctly set up
RUN mkdir -p utils api tasks && \
    touch utils/__init__.py api/__init__.py tasks/__init__.py

# Copy nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Configure nginx
COPY --chown=www-data:www-data nginx.conf /etc/nginx/nginx.conf
RUN mkdir -p /usr/share/nginx/html && \
    chown -R www-data:www-data /usr/share/nginx/html

# Copy the entrypoint script first
COPY entrypoint.sh /entrypoint.sh

# Set permissions explicitly in Docker
RUN chmod +x /entrypoint.sh

# Expose the port Nginx listens on
EXPOSE 80

ENTRYPOINT ["/entrypoint.sh"]

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1