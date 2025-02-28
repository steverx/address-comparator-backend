import os
import logging
import sys
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash
import datetime
import uuid
import gc
import psutil
import tracemalloc
import pandas as pd
from typing import Dict, List, Optional, Union, Generator
import bleach
from celery import Celery
from utils.address_utils import AddressCorrectionModel
from utils.progress import progress_tracker
from utils.memory_manager import MemoryManager
from utils.data_processing import DataProcessor
from api.api_blueprint import register_api_blueprint

# Initialize logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)

# Environment-specific variables
DEBUG = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS", "https://address-comparator-frontend-production.up.railway.app"
).split(",")
ALLOWED_EXTENSIONS = {".csv", ".xlsx"}
DEFAULT_THRESHOLD = int(os.environ.get("DEFAULT_THRESHOLD", "80"))
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "10000"))
MAX_CONTENT_LENGTH = int(
    os.environ.get("MAX_CONTENT_LENGTH", str(500 * 1024 * 1024))
)
CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
PORT = int(os.environ.get("PORT", "5000"))

# Set up libpostal environment variables
os.environ['LIBPOSTAL_DATA_DIR'] = os.environ.get('LIBPOSTAL_DATA_DIR', '/usr/local/share')
os.environ['LIBPOSTAL_INCLUDE_DIR'] = os.environ.get('LIBPOSTAL_INCLUDE_DIR', '/usr/local/include')
os.environ['LIBPOSTAL_LIB_DIR'] = os.environ.get('LIBPOSTAL_LIB_DIR', '/usr/local/lib')

def configure_auth(app):
    """Configure HTTP authentication."""
    auth = HTTPBasicAuth()
    
    users = {
        "admin": generate_password_hash(os.environ.get("ADMIN_PASSWORD", "secret"))
    }
    
    @auth.verify_password
    def verify_password(username, password):
        if username in users and check_password_hash(users.get(username), password):
            return username
        return None
    
    @app.route('/admin')
    @auth.login_required
    def admin_route():
        return "Admin area - requires authentication"
    
    return auth

def configure_celery(app):
    """Configure Celery for the application."""
    celery = Celery(
        app.name,
        broker=app.config['CELERY_BROKER_URL'],
        backend=app.config.get('CELERY_RESULT_BACKEND')
    )
    
    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)
    
    celery.Task = ContextTask
    
    return celery

def configure_error_handlers(app):
    """Configure error handlers for the application."""
    
    @app.errorhandler(404)
    def not_found_error(error):
        """Handle 404 errors."""
        logger.error(f"Route not found: {request.url}")
        return jsonify({
            "status": "error",
            "error": "Resource not found",
            "timestamp": datetime.datetime.utcnow().isoformat()
        }), 404

    @app.errorhandler(400)
    def bad_request_error(error):
        """Handle 400 errors."""
        logger.error(f"Bad request: {str(error)}")
        return jsonify({
            "status": "error",
            "error": str(error),
            "timestamp": datetime.datetime.utcnow().isoformat()
        }), 400

    @app.errorhandler(500)
    def internal_error(error):
        """Handle 500 errors."""
        logger.exception("Internal server error")
        if hasattr(app, 'memory_manager'):
            app.memory_manager.cleanup()
        return jsonify({
            "status": "error",
            "error": "Internal server error",
            "timestamp": datetime.datetime.utcnow().isoformat()
        }), 500

def create_app(config=None):
    """Application factory pattern"""
    try:
        logger.info("Starting to create Flask app")
        app = Flask(__name__)

        # Load default configuration
        app.config.update(
            ENV="production" if not DEBUG else "development",
            DEBUG=DEBUG,
            TESTING=False,
            MAX_CONTENT_LENGTH=MAX_CONTENT_LENGTH,
            ALLOWED_ORIGINS=ALLOWED_ORIGINS,
            ALLOWED_EXTENSIONS=ALLOWED_EXTENSIONS,
            DEFAULT_THRESHOLD=DEFAULT_THRESHOLD,
            CHUNK_SIZE=CHUNK_SIZE,
            CELERY_BROKER_URL=CELERY_BROKER_URL
        )
        
        # Override with custom config if provided
        if config:
            app.config.update(config)

        # Configure CORS
        CORS(
            app,
            resources={
                r"/*": {
                    "origins": app.config['ALLOWED_ORIGINS'],
                    "methods": ["GET", "POST", "OPTIONS"],
                    "allow_headers": ["Content-Type", "Authorization"],
                    "supports_credentials": True,
                }
            },
        )

        # Initialize memory manager
        app.memory_manager = MemoryManager(enable_tracemalloc=app.config['DEBUG'])
        
        # Initialize address correction model
        app.address_model = AddressCorrectionModel()
        
        # Initialize data processor
        app.data_processor = DataProcessor(
            chunk_size=app.config['CHUNK_SIZE'],
            max_workers=4,  # Configurable if needed
            address_model=app.address_model
        )
        
        # Configure authentication
        app.auth = configure_auth(app)
        
        # Configure Celery if broker URL is available
        if app.config['CELERY_BROKER_URL']:
            app.celery = configure_celery(app)
            
            # Register Celery tasks
            from tasks.comparison_tasks import register_tasks
            register_tasks(app.celery)
        
        # Register API blueprint
        register_api_blueprint(app)
        
        # Configure error handlers
        configure_error_handlers(app)
        
        # Log request information
        @app.before_request
        def log_request():
            logger.info(f"Received request: {request.method} {request.path}")
            if app.config['DEBUG']:
                logger.debug(f"Request headers: {request.headers}")
                logger.debug(f"Request data: {request.get_data()}")
        
        logger.info("Flask app created successfully")
        return app

    except Exception as e:
        logger.error(f"Error creating Flask app: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        # Start memory profiler if in debug mode
        if DEBUG:
            tracemalloc.start()
        
        # Create and run the application
        app = create_app()
        logger.info(f"Starting server on port {PORT}")
        app.run(host="0.0.0.0", port=PORT, debug=DEBUG)
        
        # Take memory snapshot in debug mode
        if DEBUG:
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')
            print("[ Top 10 memory allocations ]")
            for stat in top_stats[:10]:
                print(stat)
                
    except Exception as e:
        logger.exception("Failed to start server:")
        sys.exit(1)
