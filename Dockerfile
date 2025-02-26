# Use a consistent name for the builder stage
FROM steverx/libpostal-builder:latest as libpostal-builder

# --- Final Backend Image ---
FROM python:3.9-slim

WORKDIR /app  # Set the working directory

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        nginx \
        ca-certificates \
        gosu \
        libpq-dev \  # Keep libpq-dev ONLY if needed
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m -d /home/appuser -s /bin/bash appuser

# Copy libpostal files from the builder stage *with correct ownership*
COPY --from=libpostal-builder --chown=appuser:appuser /usr/local/lib/libpostal.so* /usr/local/lib/
COPY --from=libpostal-builder --chown=appuser:appuser /usr/local/include/libpostal/ /usr/local/include/libpostal/
COPY --from=libpostal-builder --chown=appuser:appuser /usr/local/share/libpostal/ /usr/local/share/libpostal/
COPY --from=libpostal-builder --chown=appuser:appuser /usr/local/data/ /usr/local/data/

# Set environment variables
ENV LIBPOSTAL_INCLUDE_DIR=/usr/local/include
ENV LIBPOSTAL_LIB_DIR=/usr/local/lib
ENV LIBPOSTAL_DATA_DIR=/usr/local/data
ENV LD_LIBRARY_PATH="${LIBPOSTAL_LIB_DIR}:${LD_LIBRARY_PATH}"
ENV APP_PORT=5000

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gunicorn

# Copy application code (with correct ownership)
COPY --chown=appuser:appuser . .

# Configure nginx
COPY --chown=www-data:www-data nginx.conf /etc/nginx/nginx.conf
RUN mkdir -p /usr/share/nginx/html && \
    chown -R www-data:www-data /usr/share/nginx/html

# Switch to root user for entrypoint
USER root

# Set up entrypoint
COPY entrypoint.sh /
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

# Expose port
EXPOSE 80

# Health check (use environment variable for PORT)
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Start Nginx (your app will be started by gunicorn, proxied by Nginx)
CMD ["nginx", "-g", "daemon off;"]