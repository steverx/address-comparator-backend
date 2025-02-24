# Builder stage
FROM steverx/libpostal-builder:latest as libpostal-builder

# Final Backend Image
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        python3-dev \
        pkg-config \
        nginx \
        ca-certificates \
        gosu \
        libpq-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Create directories and user
RUN useradd -m -d /home/appuser -s /bin/bash appuser && \
    mkdir -p /usr/local/{lib/pkgconfig,include/libpostal,share/libpostal}

# Copy libpostal files from builder
COPY --from=libpostal-builder /usr/local/lib/libpostal.so* /usr/local/lib/
COPY --from=libpostal-builder /usr/local/lib/pkgconfig/libpostal.pc /usr/local/lib/pkgconfig/
COPY --from=libpostal-builder /usr/local/include/libpostal/ /usr/local/include/libpostal/
COPY --from=libpostal-builder /usr/local/share/libpostal/ /usr/local/share/libpostal/

# Verify files and update library cache
RUN ls -la /usr/local/lib/libpostal* && \
    ls -la /usr/local/include/libpostal && \
    ls -la /usr/local/share/libpostal && \
    ldconfig

# Set environment variables
ENV LIBPOSTAL_DATA_DIR=/usr/local/share/libpostal \
    LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH \
    PKG_CONFIG_PATH=/usr/local/lib/pkgconfig \
    PORT=80 \
    APP_PORT=5000

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    PKG_CONFIG_PATH=/usr/local/lib/pkgconfig pip install --no-cache-dir postal==1.1.10

# Copy application code and configure nginx
COPY --chown=appuser:appuser . .
COPY --chown=www-data:www-data nginx.conf /etc/nginx/nginx.conf
COPY entrypoint.sh /

# Set permissions
RUN chmod +x /entrypoint.sh && \
    mkdir -p /usr/share/nginx/html && \
    chown -R www-data:www-data /usr/share/nginx/html && \
    chown -R appuser:appuser /app

# Expose port and health check
EXPOSE ${PORT}
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Set entrypoint and default command
ENTRYPOINT ["/entrypoint.sh"]
CMD ["gunicorn", "wsgi:app", "--bind", "0.0.0.0:5000"]