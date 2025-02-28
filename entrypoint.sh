#!/bin/bash
# filepath: /C:/Users/Steven/Documents/projects/address-comparator-backend/entrypoint.sh

# Start Nginx
nginx -g "daemon on;"

# Start the Flask application
cd /app
gunicorn --bind 0.0.0.0:${PORT} "app:create_app()"