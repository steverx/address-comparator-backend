#!/bin/bash
set -e

# Initialize environment
echo "Initializing application..."

# Make sure Gunicorn is in the PATH
export PATH=$PATH:/usr/local/bin:/usr/bin

# Print diagnostic information
echo "Current directory: $(pwd)"
echo "Python version: $(python --version)"
which gunicorn || echo "Gunicorn not found in PATH"
gunicorn --version || echo "Gunicorn version check failed"

# Start Gunicorn with proper quoting
cd /app
exec gunicorn --bind "0.0.0.0:${PORT}" "app:create_app()"