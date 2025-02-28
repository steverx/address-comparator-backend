import logging
import sys
import os
from logging.handlers import RotatingFileHandler
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

# Log startup information
logger.info(f"Starting WSGI application with Python {sys.version}")
logger.info(f"Environment: {os.getenv('FLASK_ENV', 'production')}")
logger.info(f"Memory: {psutil.virtual_memory().percent}% used")

try:
    from app import create_app
    
    logger.info("Creating Flask application")
    application = create_app()  # Standard name for WSGI applications
    app = application  # Alias for direct Flask usage
    
    # Log successful creation
    logger.info("Flask application created successfully")
    logger.info(f"Available routes: {[rule.rule for rule in application.url_map.iter_rules()]}")
    
except Exception as e:
    logger.error(f"Failed to create Flask application: {e}", exc_info=True)
    raise

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting Waitress server on port {port}")
    serve(create_app(), host="0.0.0.0", port=port)