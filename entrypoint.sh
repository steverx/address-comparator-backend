#!/bin/bash

# Set the application port (default to 5000 if not provided)
APP_PORT=${APP_PORT:-5000}

# Dynamically update nginx.conf with the correct port (using sed)
sed -i "s|proxy_pass http://localhost:5000;|proxy_pass http://localhost:${APP_PORT};|g" /etc/nginx/nginx.conf

# Switch to the non-root user and execute the command
exec gosu appuser gunicorn --bind 0.0.0.0:${APP_PORT} wsgi:app