from flask import Flask, request, jsonify
import logging
import os
import sys
import datetime
import psutil

# Initialize logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

def create_app():
    """Application factory pattern"""
    app = Flask(__name__)
    
    @app.route('/health')
    def health_check():
        """Health check endpoint for Railway."""
        try:
            memory = psutil.virtual_memory()
            health_data = {
                'status': 'healthy',
                'timestamp': datetime.datetime.utcnow().isoformat(),
                'memory_usage': f"{memory.percent}%",
                'pid': os.getpid()
            }
            logger.info(f"Health check passed: {health_data}")
            return jsonify(health_data), 200
        except Exception as e:
            logger.exception("Health check failed")
            return jsonify({
                'status': 'unhealthy',
                'error': str(e)
            }), 500

    @app.route('/')
    def root():
        """Root endpoint."""
        return jsonify({
            'message': 'Address Comparison API',
            'version': '1.0',
            'status': 'running'
        }), 200

    return app

# This is for local development only
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app = create_app()
    app.run(host='0.0.0.0', port=port)