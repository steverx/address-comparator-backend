import os
import sys
import logging
from flask import Flask, jsonify
import psutil
import datetime

# Configure logging to stdout
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for maximum verbosity
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Log startup information
logger.info("Starting application loading process")
logger.info(f"Python version: {sys.version}")
logger.info(f"Current working directory: {os.getcwd()}")

def create_app():
    """Create minimal Flask application"""
    try:
        logger.info("Starting to create Flask app")
        app = Flask(__name__)
        
        @app.route('/')
        def root():
            logger.info("Root endpoint called")
            return jsonify({
                'status': 'running',
                'timestamp': datetime.datetime.utcnow().isoformat()
            })

        @app.route('/health')
        def health():
            try:
                logger.info("Health check endpoint called")
                memory = psutil.virtual_memory()
                health_data = {
                    'status': 'healthy',
                    'timestamp': datetime.datetime.utcnow().isoformat(),
                    'memory_usage': f"{memory.percent}%",
                    'pid': os.getpid()
                }
                logger.info(f"Health check data: {health_data}")
                return jsonify(health_data), 200
            except Exception as e:
                logger.error(f"Health check error: {str(e)}", exc_info=True)
                return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

        @app.errorhandler(500)
        def handle_500(error):
            logger.error(f"Internal server error: {str(error)}", exc_info=True)
            return jsonify({
                'status': 'error',
                'message': 'Internal server error',
                'timestamp': datetime.datetime.utcnow().isoformat()
            }), 500

        @app.errorhandler(404)
        def handle_404(error):
            logger.warning(f"Not found error: {request.url}")
            return jsonify({
                'status': 'error',
                'message': 'Resource not found',
                'path': request.url
            }), 404

        logger.info("Flask app created successfully with all routes registered")
        return app
    except Exception as e:
        logger.error(f"Error creating Flask app: {str(e)}", exc_info=True)
        raise

# Log environment variables at module level
logger.info("Environment variables:")
for key, value in os.environ.items():
    if not any(sensitive in key.lower() for sensitive in ['key', 'secret', 'password', 'token']):
        logger.info(f"{key}: {value}")

logger.info("Module loaded successfully")

if __name__ == '__main__':
    try:
        port = int(os.environ.get('PORT', 8000))
        logger.info(f"Starting development server on port {port}")
        app = create_app()
        app.run(host='0.0.0.0', port=port)
    except Exception as e:
        logger.error(f"Failed to start development server: {str(e)}", exc_info=True)
        sys.exit(1)