#!/bin/bash
set -e

# Default configuration
APP_PORT=${APP_PORT:-5000}
NGINX_CONF="/etc/nginx/nginx.conf"
LOG_DIR="/var/log/nginx"

# Function to log messages
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Create log directory if it doesn't exist
if [ ! -d "$LOG_DIR" ]; then
    log "Creating nginx log directory"
    mkdir -p "$LOG_DIR"
    chown -R www-data:www-data "$LOG_DIR"
fi

# Check if nginx configuration exists
if [ ! -f "$NGINX_CONF" ]; then
    log "Error: Nginx configuration file not found at $NGINX_CONF"
    exit 1
fi

# Update nginx configuration with correct port
log "Configuring Nginx to proxy to port $APP_PORT"
if ! sed -i "s|proxy_pass http://127.0.0.1:5000;|proxy_pass http://127.0.0.1:${APP_PORT};|g" "$NGINX_CONF"; then
    log "Error: Failed to update Nginx configuration"
    exit 1
fi

# Verify nginx configuration
log "Verifying Nginx configuration"
if ! nginx -t; then
    log "Error: Invalid Nginx configuration"
    exit 1
fi

# Start nginx in background
log "Starting Nginx in background"
nginx

# Start application using gosu
log "Starting Gunicorn as appuser on port $APP_PORT"
exec gosu appuser gunicorn wsgi:app \
    --bind "0.0.0.0:$APP_PORT" \
    --workers 4 \
    --worker-class gthread \
    --threads 8 \
    --timeout 120 \
    --access-logfile - \
    --error-logfile - \
    --log-level debug