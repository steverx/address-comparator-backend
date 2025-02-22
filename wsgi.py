import logging
import sys
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,  # Changed to INFO for production
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

logger.info("Initializing WSGI application")
logger.info(f"Python version: {sys.version}")
logger.info(f"Environment: {os.getenv('FLASK_ENV', 'production')}")

try:
    from app import create_app
    
    logger.info("Creating Flask application")
    application = create_app()
    app = application  # Alias for Gunicorn
    logger.info("Flask application created successfully")
    
except Exception as e:
    logger.error(f"Failed to create Flask application: {e}")
    raise

if __name__ == "__main__":
    port = int(os.getenv('PORT', '8080'))
    logger.info(f"Starting development server on port {port}")
    application.run(host='0.0.0.0', port=port)