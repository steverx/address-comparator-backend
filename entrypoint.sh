#!/bin/sh
# NO EXTRA SPACES OR CHARACTERS IN THIS FILE

# Start Nginx
nginx -g 'daemon on;'

# Start gunicorn
cd /app 
exec gunicorn --bind 0.0.0.0:${PORT} "app:create_app()"