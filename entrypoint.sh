#!/bin/bash
set -e

# Start Nginx in the background
nginx -g "daemon on;"

# Change to app directory
cd /app

# Start the Flask application with Gunicorn
exec gunicorn --bind 0.0.0.0:${PORT} "app:create_app()"