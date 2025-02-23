# Builder stage (Pulls from your registry - replace with your actual image)
FROM steverx/libpostal-builder:latest as libpostal-builder

# --- Final Backend Image ---
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies (nginx and ca-certificates ONLY)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gosu \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m -d /home/appuser -s /bin/bash appuser && \
    mkdir -p /usr/share/nginx/html && \
    chown -R www-data:www-data /usr/share/nginx/html

# Create log directories
RUN mkdir -p /var/log/nginx && \
    chown -R www-data:www-data /var/log/nginx

# Copy libpostal files from the builder stage (with correct ownership)
COPY --from=libpostal-builder --chown=appuser:appuser /usr/local/lib/libpostal.so* /usr/local/lib/
COPY --from=libpostal-builder --chown=appuser:appuser /usr/local/include/libpostal/ /usr/local/include/libpostal/
COPY --from=libpostal-builder --chown=appuser:appuser /usr/local/share/libpostal/ /usr/local/share/libpostal/
COPY --from=libpostal-builder --chown=appuser:appuser /usr/local/data/ /usr/local/data/

# Set environment variables
ENV LIBPOSTAL_INCLUDE_DIR=/usr/local/include \
    LIBPOSTAL_LIB_DIR=/usr/local/lib \
    LIBPOSTAL_DATA_DIR=/usr/local/data \
    LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}" \
    PORT=80 \
    APP_PORT=5000

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (with correct ownership)
COPY --chown=appuser:appuser . .

# Configure nginx
COPY --chown=www-data:www-data nginx.conf /etc/nginx/nginx.conf
COPY entrypoint.sh /
RUN chmod +x /entrypoint.sh

# Configure permissions
RUN ldconfig && \
    chown -R appuser:appuser /app

# Expose port
EXPOSE ${PORT}

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Start Nginx (your app will be started by Gunicorn or similar, proxied by Nginx)
ENTRYPOINT ["/entrypoint.sh"]
CMD ["nginx", "-g", "daemon off;"]