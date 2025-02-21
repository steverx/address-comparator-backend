import logging
import sys

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

logger.info("Initializing WSGI application")

from app import create_app

logger.info("Creating Flask application")
application = create_app()
logger.info("Flask application created successfully")

if __name__ == "__main__":
    application.run()