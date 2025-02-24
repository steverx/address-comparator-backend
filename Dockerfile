# Builder stage
FROM steverx/libpostal-builder:latest AS libpostal-builder

# Final Backend Image
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        nginx \
        ca-certificates \
        gosu \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user and required directories
RUN useradd -m -d /home/appuser -s /bin/bash appuser && \
    mkdir -p /usr/share/nginx/html && \
    mkdir -p /var/log/nginx && \
    chown -R www-data:www-data /usr/share/nginx/html && \
    chown -R www-data:www-data /var/log/nginx

# Create base directories with proper permissions
RUN mkdir -p /usr/local/lib && \
    mkdir -p /usr/local/include && \
    mkdir -p /usr/local/share && \
    chown -R appuser:appuser /usr/local/lib && \
    chown -R appuser:appuser /usr/local/include && \
    chown -R appuser:appuser /usr/local/share

# Copy libpostal files
COPY --from=libpostal-builder /usr/local/lib/* /usr/local/lib/
COPY --from=libpostal-builder /usr/local/include/* /usr/local/include/
COPY --from=libpostal-builder /usr/local/share/* /usr/local/share/

# Environment setup
ENV LIBPOSTAL_INCLUDE_DIR=/usr/local/include \
    LIBPOSTAL_LIB_DIR=/usr/local/lib \
    LIBPOSTAL_DATA_DIR=/usr/local/share \
    LD_LIBRARY_PATH=/usr/local/lib \
    PORT=80 \
    APP_PORT=5000

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application setup
COPY --chown=appuser:appuser . .
COPY --chown=www-data:www-data nginx.conf /etc/nginx/nginx.conf
COPY entrypoint.sh /
RUN chmod +x /entrypoint.sh

# Configure permissions and update library cache
RUN ldconfig && \
    chown -R appuser:appuser /app

# Expose port and health check
EXPOSE ${PORT}
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

ENTRYPOINT ["/entrypoint.sh"]
CMD ["nginx", "-g", "daemon off;"]