from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
from fuzzywuzzy import fuzz
from utils.address_utils import AddressCorrectionModel
import io
import logging
import os
import datetime
import gc
import sys
from werkzeug.datastructures import FileStorage
from typing import Dict, List, Optional, Union, Generator
import psutil

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
    force=True
)
logger = logging.getLogger(__name__)

# Configuration
DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
ALLOWED_ORIGINS = ["https://address-comparator-frontend-production.up.railway.app"]
ALLOWED_EXTENSIONS = {'.csv', '.xlsx'}
DEFAULT_THRESHOLD = 80
CHUNK_SIZE = 10000

# Initialize address correction model
address_model = AddressCorrectionModel()

def cleanup_memory(dataframes: List[pd.DataFrame] = None):
    """Enhanced memory cleanup."""
    if dataframes:
        for df in dataframes:
            del df
    gc.collect()
    memory = psutil.virtual_memory()
    logger.info(f"Memory usage after cleanup: {memory.percent}%")

def create_app():
    """Application factory pattern"""
    try:
        logger.info("Starting to create Flask app")
        app = Flask(__name__)
        
        # Basic configuration
        app.config.update(
            ENV='production',
            DEBUG=DEBUG,
            MAX_CONTENT_LENGTH=500 * 1024 * 1024  # 500MB max-size
        )

        # CORS setup
        CORS(app, resources={r"/*": {
            "origins": ALLOWED_ORIGINS,
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"],
            "supports_credentials": True
        }})

        def allowed_file(filename: str) -> bool:
            """Check if file extension is allowed."""
            return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS

        def get_match_score(addr1: str, addr2: str) -> float:
            """Calculate fuzzy match score between two addresses."""
            return fuzz.ratio(addr1.lower(), addr2.lower())

        def process_dataframe_in_chunks(df: pd.DataFrame) -> Generator[pd.DataFrame, None, None]:
            """Process large DataFrames in chunks."""
            for i in range(0, len(df), CHUNK_SIZE):
                yield df.iloc[i:i + CHUNK_SIZE]

        def load_dataframe(file_storage: FileStorage) -> pd.DataFrame:
            """Load dataframe with chunked reading."""
            try:
                file_storage.seek(0)
                if file_storage.filename.endswith('.csv'):
                    return pd.read_csv(file_storage)
                elif file_storage.filename.endswith('.xlsx'):
                    return pd.read_excel(file_storage)
                else:
                    raise ValueError(f"Unsupported file type: {file_storage.filename}")
            except Exception as e:
                logger.error(f"Error reading file {file_storage.filename}: {e}")
                raise

        def combine_address_components(row: pd.Series, columns: List[str]) -> str:
            """Combine address components into a single string."""
            components = []
            for col in columns:
                if pd.notna(row[col]):
                    val = str(row[col]).strip()
                    if val:
                        components.append(val)
            return ', '.join(components) if components else ''

        def process_address_comparison(
            df1: pd.DataFrame,
            df2: pd.DataFrame,
            columns1: List[str],
            columns2: List[str],
            threshold: float
        ) -> List[Dict]:
            """Process address comparison with ML-enhanced matching."""
            try:
                results = []
                
                df1['combined_address'] = df1.apply(
                    lambda row: combine_address_components(row, columns1), axis=1
                )
                df2['combined_address'] = df2.apply(
                    lambda row: combine_address_components(row, columns2), axis=1
                )
                
                for chunk1 in process_dataframe_in_chunks(df1):
                    for chunk2 in process_dataframe_in_chunks(df2):
                        chunk1['normalized_address'] = chunk1['combined_address'].apply(
                            lambda addr: address_model.normalize_address(addr)
                        )
                        chunk2['normalized_address'] = chunk2['combined_address'].apply(
                            lambda addr: address_model.normalize_address(addr)
                        )
                        
                        for _, row1 in chunk1.iterrows():
                            matches = []
                            for _, row2 in chunk2.iterrows():
                                score = address_model.compare_addresses(
                                    row1['normalized_address'],
                                    row2['normalized_address']
                                )
                                if score >= threshold:
                                    matches.append({
                                        'address': row2['combined_address'],
                                        'score': score,
                                        'normalized': row2['normalized_address']
                                    })
                            
                            if matches:
                                results.append({
                                    'source_address': row1['combined_address'],
                                    'normalized_source': row1['normalized_address'],
                                    'matches': sorted(matches, key=lambda x: x['score'], reverse=True)
                                })
                
                return results
            
            except Exception as e:
                logger.exception("Error in address comparison")
                raise

        @app.route('/')
        def index():
            """Root endpoint for API documentation."""
            return jsonify({
                'status': 'running',
                'version': '1.0',
                'endpoints': {
                    'health': '/health',
                    'columns': '/columns',
                    'compare': '/compare',
                    'validate': '/validate'
                },
                'timestamp': datetime.datetime.utcnow().isoformat()
            }), 200

        @app.route('/health')
        def health_check():
            """Health check endpoint for Railway."""
            try:
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                health_data = {
                    'status': 'healthy',
                    'timestamp': datetime.datetime.utcnow().isoformat(),
                    'memory': {
                        'total': memory.total,
                        'available': memory.available,
                        'percent': memory.percent
                    },
                    'disk': {
                        'total': disk.total,
                        'free': disk.free,
                        'percent': disk.percent
                    },
                    'pid': os.getpid(),
                    'uptime': datetime.datetime.now().timestamp() - psutil.Process().create_time()
                }
                logger.info(f"Health check passed: {health_data}")
                return jsonify(health_data), 200
            except Exception as e:
                logger.exception("Health check failed")
                return jsonify({
                    'status': 'unhealthy',
                    'error': str(e),
                    'timestamp': datetime.datetime.utcnow().isoformat()
                }), 500

        @app.before_request
        def before_request():
            """Log incoming requests."""
            logger.info(f"Request received: {request.method} {request.path}")
            logger.debug(f"Headers: {dict(request.headers)}")

        @app.route('/columns', methods=['POST'])
        def get_columns():
            """Handle column name requests."""
            try:
                if 'file' not in request.files:
                    return jsonify({'error': 'No file provided'}), 400
                
                file = request.files['file']
                if not file or not allowed_file(file.filename):
                    return jsonify({'error': 'Invalid file type'}), 400

                df = load_dataframe(file)
                columns = df.columns.tolist()
                
                return jsonify({
                    'status': 'success',
                    'data': columns
                }), 200
            except Exception as e:
                logger.exception("Error processing columns")
                return jsonify({
                    'status': 'error',
                    'error': str(e)
                }), 500

        @app.route('/compare', methods=['POST'])
        def compare_addresses():
            """Handle address comparison requests."""
            try:
                if 'file1' not in request.files or 'file2' not in request.files:
                    return jsonify({
                        'status': 'error',
                        'error': 'Both files are required'
                    }), 400

                file1 = request.files['file1']
                file2 = request.files['file2']
                
                if not (file1 and allowed_file(file1.filename) and 
                       file2 and allowed_file(file2.filename)):
                    return jsonify({
                        'status': 'error',
                        'error': 'Invalid file type'
                    }), 400

                address_columns1 = request.form.getlist('columns1[]')
                address_columns2 = request.form.getlist('columns2[]')
                threshold = float(request.form.get('threshold', DEFAULT_THRESHOLD))

                if not address_columns1 or not address_columns2:
                    return jsonify({
                        'status': 'error',
                        'error': 'Address columns must be specified'
                    }), 400

                df1 = load_dataframe(file1)
                df2 = load_dataframe(file2)

                results = process_address_comparison(
                    df1, df2,
                    address_columns1,
                    address_columns2,
                    threshold
                )

                cleanup_memory([df1, df2])

                return jsonify({
                    'status': 'success',
                    'data': results
                }), 200

            except Exception as e:
                logger.exception("Error in compare_addresses")
                return jsonify({
                    'status': 'error',
                    'error': str(e)
                }), 500

        @app.route('/validate', methods=['POST'])
        def validate_address():
            """Validate and normalize a single address."""
            try:
                if not request.is_json:
                    return jsonify({
                        'status': 'error',
                        'error': 'Request must be JSON'
                    }), 400

                data = request.get_json()
                address = data.get('address')
                
                if not address:
                    return jsonify({
                        'status': 'error',
                        'error': 'Address is required'
                    }), 400

                normalized = address_model.normalize_address(address)
                corrections = address_model.suggest_corrections(address)
                
                return jsonify({
                    'status': 'success',
                    'data': {
                        'original': address,
                        'normalized': normalized,
                        'suggestions': corrections,
                        'valid': address_model.is_valid_address(normalized)
                    }
                }), 200

            except Exception as e:
                logger.exception("Error in address validation")
                return jsonify({
                    'status': 'error',
                    'error': str(e)
                }), 500

        @app.errorhandler(404)
        def not_found_error(error):
            """Handle 404 errors."""
            logger.error(f"Route not found: {request.url}")
            return jsonify({'error': 'Resource not found'}), 404

        @app.errorhandler(400)
        def bad_request_error(error):
            """Handle 400 errors."""
            logger.error(f"Bad request: {str(error)}")
            return jsonify({'error': str(error)}), 400

        @app.errorhandler(500)
        def internal_error(error):
            """Handle 500 errors."""
            logger.exception("Internal server error")
            cleanup_memory()
            return jsonify({
                'error': 'Internal server error',
                'status': 'error',
                'timestamp': datetime.datetime.utcnow().isoformat()
            }), 500

        logger.info("Flask app created successfully")
        return app

    except Exception as e:
        logger.error(f"Error creating Flask app: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    try:
        app = create_app()
        port = int(os.environ.get('PORT', '5000'))
        logger.info(f"Starting server on port {port}")
        app.run(
            host='0.0.0.0',
            port=port,
            debug=DEBUG
        )
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}", exc_info=True)
        sys.exit(1)