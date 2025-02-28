import os
import sys
import logging
import subprocess
import psutil
from waitress import serve

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
    force=True  # Ensure our configuration takes precedence
)
logger = logging.getLogger(__name__)

# Log detailed environment information
logger.info(f"Starting WSGI application with Python {sys.version}")
logger.info(f"Environment variables: {dict(os.environ)}")
logger.info(f"Current directory: {os.getcwd()}")
logger.info(f"Directory contents: {os.listdir('.')}")

# Check for any scripts that might contain gunicorn
logger.info("Checking for files that might contain gunicorn commands:")
for root, dirs, files in os.walk('.'):
    for file in files:
        if file.endswith(('.sh', '.py', '.txt')):
            path = os.path.join(root, file)
            try:
                with open(path, 'r') as f:
                    content = f.read()
                    if 'gunicorn' in content and '--bind' in content:
                        logger.warning(f"Found gunicorn --bind in: {path}")
                        logger.warning(f"Content snippet: {content[:200]}...")
            except Exception as e:
                logger.error(f"Error reading {path}: {e}")

try:
    from app import create_app
    application = create_app()
    app = application
    logger.info("Flask application created successfully")
    logger.info(f"Available routes: {[rule.rule for rule in application.url_map.iter_rules()]}")
except Exception as e:
    logger.error(f"Failed to create Flask application: {e}", exc_info=True)
    raise

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting Waitress server on port {port}")
    serve(application, host="0.0.0.0", port=port)