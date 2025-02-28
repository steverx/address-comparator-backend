# Use a consistent name for the builder stage
FROM steverx/libpostal-builder:latest as libpostal-builder

# --- Go Builder Stage ---
FROM golang:1.20 as go-builder

# Copy libpostal files from the first builder stage
COPY --from=libpostal-builder /usr/local/lib/libpostal.so* /usr/local/lib/
COPY --from=libpostal-builder /usr/local/include/libpostal/ /usr/local/include/libpostal/
COPY --from=libpostal-builder /usr/local/data/ /usr/local/data/

# Set environment variables for build
ENV LIBPOSTAL_INCLUDE_DIR=/usr/local/include
ENV LIBPOSTAL_LIB_DIR=/usr/local/lib
ENV LIBPOSTAL_DATA_DIR=/usr/local/data
ENV LD_LIBRARY_PATH="${LIBPOSTAL_LIB_DIR}:${LD_LIBRARY_PATH}"
ENV CGO_ENABLED=1
ENV GOOS=linux
ENV GOARCH=amd64

# Build the gopostal binary
RUN go install github.com/openvenues/gopostal/cmd/postal-rest@latest && \
    ldconfig

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
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m -d /home/appuser -s /bin/bash appuser

# Copy libpostal files from the builder stage
COPY --from=libpostal-builder --chown=appuser:appuser /usr/local/lib/libpostal.so* /usr/local/lib/
COPY --from=libpostal-builder --chown=appuser:appuser /usr/local/include/libpostal/ /usr/local/include/libpostal/
COPY --from=libpostal-builder --chown=appuser:appuser /usr/local/data/ /usr/local/data/

# Copy the built postal-rest binary
COPY --from=go-builder --chown=root:root /go/bin/postal-rest /usr/local/bin/

# Set environment variables
ENV LIBPOSTAL_INCLUDE_DIR=/usr/local/include
ENV LIBPOSTAL_LIB_DIR=/usr/local/lib
ENV LIBPOSTAL_DATA_DIR=/usr/local/data
ENV LD_LIBRARY_PATH="${LIBPOSTAL_LIB_DIR}:${LD_LIBRARY_PATH}"
ENV PORT=5000
ENV POSTAL_PORT=8080

# Run ldconfig to update shared libraries
RUN ldconfig

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gunicorn requests

# Copy application code (with correct ownership)
COPY --chown=appuser:appuser . .

# Copy nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Configure nginx
COPY --chown=www-data:www-data nginx.conf /etc/nginx/nginx.conf
RUN mkdir -p /usr/share/nginx/html && \
    chown -R www-data:www-data /usr/share/nginx/html

# Copy and set permissions for entrypoint script
COPY --chown=root:root entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Expose the port Nginx listens on
EXPOSE 80

ENTRYPOINT ["/entrypoint.sh"]

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1