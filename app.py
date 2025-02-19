from flask import Flask, request, jsonify, send_file, make_response
from flask_cors import CORS
import pandas as pd
from fuzzywuzzy import fuzz
import utils.address_utils as address_utils
import io
import logging
import os
import re
import datetime
import gc
import sys
from werkzeug.datastructures import FileStorage
from typing import Dict, List, Optional, Union, Generator
from waitress import serve
import psutil
from uuid import uuid4
import numpy as np

# Initialize logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout  # Ensure logs go to stdout for Railway
)
logger = logging.getLogger(__name__)

# Configuration
DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
PORT = int(os.environ.get('PORT', 8000))
ALLOWED_ORIGINS = os.environ.get('ALLOWED_ORIGINS', 
    'https://address-comparator-frontend-production.up.railway.app').split(',')
ALLOWED_EXTENSIONS = {'.csv', '.xlsx'}
DEFAULT_THRESHOLD = 80
CHUNK_SIZE = 10000

def create_app():
    """Application factory pattern"""
    app = Flask(__name__)
    
    # Basic configuration
    app.config.update(
        ENV='production',
        DEBUG=DEBUG,
        MAX_CONTENT_LENGTH=500 * 1024 * 1024  # 500MB max-size
    )

    # Configure CORS
    CORS(app, 
        resources={r"/*": {
            "origins": ALLOWED_ORIGINS,
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization", "Accept", "Origin"],
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

    def load_dataframe(file_storage: FileStorage) -> Union[pd.DataFrame, pd.io.parsers.TextFileReader]:
        """Load dataframe with chunked reading."""
        try:
            file_storage.seek(0)
            if file_storage.filename.endswith('.csv'):
                try:
                    return pd.read_csv(file_storage)
                except UnicodeDecodeError:
                    file_storage.seek(0)
                    return pd.read_csv(file_storage, encoding='latin1')
            elif file_storage.filename.endswith('.xlsx'):
                return pd.read_excel(file_storage)
            else:
                raise ValueError(f"Unsupported file type: {file_storage.filename}")
        except Exception as e:
            logger.error(f"Error reading file {file_storage.filename}: {e}")
            raise

    def cleanup_memory(dataframes: List[pd.DataFrame] = None):
        """Enhanced memory cleanup function."""
        if dataframes:
            for df in dataframes:
                del df
        # Force memory cleanup
        gc.collect()
        # Clear any cached memory
        if 'linux' in sys.platform:
            os.system('sync')  # Sync filesystem

    def combine_address_components(row: pd.Series, columns: List[str], is_excel: bool = False) -> str:
        """Combine address components into a single string."""
        components = []
        
        if is_excel and any(col.endswith('1') for col in columns):
            columns = [col for col in columns if not col.endswith('1')]
        
        for col in columns:
            if pd.notna(row[col]):
                val = str(row[col]).strip()
                if val:
                    components.append(val)
        
        return ', '.join(components) if components else ''

    def validate_file(file: FileStorage) -> bool:
        """Validate file before processing."""
        if not file or not file.filename:
            raise ValueError("No file provided")
        
        if not allowed_file(file.filename):
            raise ValueError(f"Invalid file type: {file.filename}")
        
        return True

    @app.route('/health')
    def health_check():
        """Health check endpoint for Railway."""
        try:
            memory = psutil.virtual_memory()
            health_data = {
                'status': 'healthy',
                'timestamp': datetime.datetime.utcnow().isoformat(),
                'memory_usage': f"{memory.percent}%",
                'pid': os.getpid(),
                'uptime': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'ready': True
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

    @app.route('/')
    def root():
        """Root endpoint."""
        return jsonify({
            'message': 'Address Comparison API',
            'version': '1.0',
            'status': 'running'
        }), 200

    @app.route('/columns', methods=['POST'])
    def get_columns():
        """Handle column name requests."""
        try:
            logger.info("Columns request received")
            
            if 'file' not in request.files:
                return jsonify({
                    'status': 'error',
                    'error': 'No file provided'
                }), 400

            file = request.files['file']
            if not file or not allowed_file(file.filename):
                return jsonify({
                    'status': 'error',
                    'error': 'Invalid file type'
                }), 400

            df = load_dataframe(file)
            columns = df.columns.tolist()
            
            logger.info(f"Found columns: {columns}")
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
            
            address_columns1 = request.form.getlist('columns1[]')
            address_columns2 = request.form.getlist('columns2[]')
            threshold = float(request.form.get('threshold', DEFAULT_THRESHOLD))
            
            if not address_columns1 or not address_columns2:
                return jsonify({
                    'status': 'error',
                    'error': 'Address columns must be specified'
                }), 400

            try:
                validate_file(file1)
                validate_file(file2)
            except ValueError as e:
                return jsonify({
                    'status': 'error',
                    'error': str(e)
                }), 400

            df1 = load_dataframe(file1)
            df2 = load_dataframe(file2)

            is_excel1 = file1.filename.endswith('.xlsx')
            is_excel2 = file2.filename.endswith('.xlsx')

            addresses1 = []
            addresses2 = []
            results = []

            for _, row in df1.iterrows():
                addr = combine_address_components(row, address_columns1, is_excel1)
                if addr:
                    try:
                        correction = address_utils.AddressCorrectionModel().correct_address(addr)
                        if correction['validation']['is_valid']:
                            addresses1.append({
                                'original': addr,
                                'cleaned': correction['spelling_corrected'],
                                'confidence': correction['validation']['confidence']
                            })
                    except Exception as e:
                        logger.error(f"Error processing address in file1: {addr}. Error: {str(e)}")
                        continue

            for _, row in df2.iterrows():
                addr = combine_address_components(row, address_columns2, is_excel2)
                if addr:
                    try:
                        correction = address_utils.AddressCorrectionModel().correct_address(addr)
                        if correction['validation']['is_valid']:
                            addresses2.append({
                                'original': addr,
                                'cleaned': correction['spelling_corrected'],
                                'confidence': correction['validation']['confidence']
                            })
                    except Exception as e:
                        logger.error(f"Error processing address in file2: {addr}. Error: {str(e)}")
                        continue

            for addr1 in addresses1:
                best_match = None
                best_score = 0
                
                for addr2 in addresses2:
                    score = get_match_score(addr1['cleaned'], addr2['cleaned'])
                    if score > best_score:
                        best_score = score
                        best_match = addr2
                
                if best_match and best_score >= threshold:
                    avg_confidence = (addr1['confidence'] + best_match['confidence']) / 2
                    results.append({
                        'address1': addr1['original'],
                        'address2': best_match['original'],
                        'match_score': best_score,
                        'parsing_confidence': round(avg_confidence, 2)
                    })

            cleanup_memory([df1, df2])
            
            if request.form.get('export') == 'true':
                if not results:
                    return jsonify({'error': 'No results to export'}), 400
                    
                output = io.BytesIO()
                df_results = pd.DataFrame(results)
                
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df_results.to_excel(writer, sheet_name='Address Matches', index=False)
                
                output.seek(0)
                return send_file(
                    output,
                    mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    as_attachment=True,
                    download_name='address_matches.xlsx'
                )
            
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
        """Handle 500 errors explicitly."""
        logger.exception("Internal server error")
        cleanup_memory()  # Force cleanup on error
        return jsonify({
            'error': 'Internal server error',
            'status': 'error',
            'timestamp': datetime.datetime.utcnow().isoformat()
        }), 500

    return app

def main():
    """Main entry point"""
    try:
        logger.info("=== Starting Address Comparison API ===")
        logger.info(f"Port: {PORT}")
        logger.info(f"Debug mode: {DEBUG}")
        logger.info(f"Allowed origins: {ALLOWED_ORIGINS}")
        logger.info(f"Process ID: {os.getpid()}")
        logger.info(f"Working directory: {os.getcwd()}")
        logger.info(f"Python version: {sys.version}")
        
        # Log environment variables (excluding sensitive ones)
        safe_env_vars = {k: v for k, v in os.environ.items() 
                        if not any(sensitive in k.lower() 
                                 for sensitive in ['key', 'secret', 'password', 'token'])}
        logger.info(f"Environment variables: {safe_env_vars}")
        
        app = create_app()
        
        if os.environ.get('RAILWAY_ENVIRONMENT') == 'production':
            logger.info("Starting production server with Waitress")
            serve(app,
                  host='0.0.0.0',
                  port=PORT,
                  threads=4,
                  url_scheme='https')
        else:
            logger.info("Starting development server")
            app.run(host='0.0.0.0', port=PORT, debug=DEBUG)
            
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()