#!/bin/bash
set -e

# Default configuration
APP_PORT=${APP_PORT:-5000}
NGINX_CONF="/etc/nginx/nginx.conf"

# Function to log messages
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Check if nginx configuration exists
if [ ! -f "$NGINX_CONF" ]; then
    log "Error: Nginx configuration file not found at $NGINX_CONF"
    exit 1
fi

# Update nginx configuration with correct port
log "Configuring Nginx to proxy to port $APP_PORT"
if ! sed -i "s|proxy_pass http://localhost:5000;|proxy_pass http://localhost:${APP_PORT};|g" "$NGINX_CONF"; then
    log "Error: Failed to update Nginx configuration"
    exit 1
fi

# Verify nginx configuration
if ! nginx -t; then
    log "Error: Invalid Nginx configuration"
    exit 1
fi

# Start application using gosu instead of su-exec
log "Starting application as appuser"
exec gosu appuser "$@"