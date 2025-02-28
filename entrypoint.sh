#!/bin/bash
set -e

# Set the application port (use PORT from Dockerfile)
APP_PORT=${PORT:-5000}
POSTAL_PORT=${POSTAL_PORT:-8080}

# Dynamically update nginx.conf with the correct port
sed -i "s|proxy_pass http://localhost:5000;|proxy_pass http://localhost:${APP_PORT};|g" /etc/nginx/nginx.conf

# Start the libpostal REST service in the background
postal-rest -host 127.0.0.1 -port ${POSTAL_PORT} &

# Start nginx in background
nginx

# Switch to the non-root user and execute the gunicorn command
exec gosu appuser gunicorn --bind 0.0.0.0:${APP_PORT} wsgi:app