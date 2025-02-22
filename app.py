from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
from fuzzywuzzy import fuzz
from utils.address_utils import AddressCorrectionModel  # Import your model
import io
import logging
import os
import datetime
import gc
import sys
from werkzeug.datastructures import FileStorage
from typing import Dict, List, Optional, Union, Generator
import psutil  # For memory usage

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout,  # Output to stdout, which Railway uses
    force=True  # Force settings, useful for overriding in some environments
)
logger = logging.getLogger(__name__)

# Configuration
DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
ALLOWED_ORIGINS = ["https://address-comparator-frontend-production.up.railway.app"]  # Replace with your frontend URL
ALLOWED_EXTENSIONS = {'.csv', '.xlsx'}
DEFAULT_THRESHOLD = 80
CHUNK_SIZE = 10000  # Process in chunks to reduce memory usage

# Initialize address correction model. Make sure this is done *outside* the app context
# so it's not re-initialized on every request, which would be very inefficient.
address_model = AddressCorrectionModel()  # Initialize once, globally

def cleanup_memory(dataframes: List[pd.DataFrame] = None):
    """Enhanced memory cleanup, including explicit DataFrame deletion and garbage collection."""
    if dataframes:
        for df in dataframes:
            del df  # Explicitly delete dataframes
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
            ENV='production',  # Or 'development', as appropriate
            DEBUG=DEBUG,
            MAX_CONTENT_LENGTH=500 * 1024 * 1024  # 500MB max-size
        )

        # CORS setup
        CORS(app, resources={r"/*": {
            "origins": ALLOWED_ORIGINS,
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"],
            "supports_credentials": True  # Only if you need to support cookies/auth
        }})

        def allowed_file(filename: str) -> bool:
            """Check if file extension is allowed."""
            return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS

        def get_match_score(addr1: str, addr2: str) -> float:
            """Calculate fuzzy match score between two addresses."""
            return address_model.compare_addresses(addr1, addr2)  # use model

        def process_dataframe_in_chunks(df: pd.DataFrame) -> Generator[pd.DataFrame, None, None]:
            """Process large DataFrames in chunks to avoid memory issues."""
            for i in range(0, len(df), CHUNK_SIZE):
                yield df.iloc[i:i + CHUNK_SIZE]


        def load_dataframe(file_storage: FileStorage) -> pd.DataFrame:
            """Load dataframe, handling different file types and chunked reading for large files.
            """
            try:
                file_storage.seek(0)  # Ensure we're at the start of the file
                if file_storage.filename.endswith('.csv'):
                    return pd.read_csv(file_storage)  # Use pandas for CSV
                elif file_storage.filename.endswith('.xlsx'):
                    return pd.read_excel(file_storage)  # And for Excel
                else:
                    raise ValueError(f"Unsupported file type: {file_storage.filename}")
            except Exception as e:
                logger.error(f"Error reading file {file_storage.filename}: {e}")
                raise

        def combine_address_components(row: pd.Series, columns: List[str]) -> str:
            """Combine specified address components from a DataFrame row into a single string."""
            components = []
            for col in columns:
                if pd.notna(row[col]):  # Check for NaN/None values
                    val = str(row[col]).strip() # Convert to string and remove spaces
                    if val: # Check if value exists.
                        components.append(val)
            return ', '.join(components) if components else ''

        def process_address_comparison(df1: pd.DataFrame, df2: pd.DataFrame, columns1: List[str], columns2: List[str], threshold: float) -> List[Dict]:
            """Process address comparison using ML-enhanced matching."""
            try:
                results = []

                # Combine selected address columns into a single 'combined_address' column
                df1['combined_address'] = df1.apply(lambda row: combine_address_components(row, columns1), axis=1)
                df2['combined_address'] = df2.apply(lambda row: combine_address_components(row, columns2), axis=1)

                # Process in chunks to handle very large files efficiently.
                for chunk1 in process_dataframe_in_chunks(df1):
                    for chunk2 in process_dataframe_in_chunks(df2):
                        # Normalize addresses *within* the loop, on the chunks.
                        chunk1['normalized_address'] = chunk1['combined_address'].apply(lambda addr: address_model.normalize_address(addr))
                        chunk2['normalized_address'] = chunk2['combined_address'].apply(lambda addr: address_model.normalize_address(addr))

                        for _, row1 in chunk1.iterrows():
                            matches = []
                            for _, row2 in chunk2.iterrows():
                                # use the address model compare
                                score = address_model.compare_addresses(row1['normalized_address'], row2['normalized_address'])

                                if score >= threshold:
                                    matches.append({
                                        'address': row2['combined_address'],
                                        'score': score,
                                        'normalized': row2['normalized_address']  # Include normalized address for debugging.
                                    })
                            if matches:
                                results.append({
                                    'source_address': row1['combined_address'],
                                    'normalized_source': row1['normalized_address'], # Include normalized source for debugging.
                                    'matches': sorted(matches, key=lambda x: x['score'], reverse=True)  # Sort by score
                                    })
                return results
            except Exception as e:
                logger.exception("Error in address comparison") # More specific exception logging
                raise

        @app.route('/')
        def index():
            """Root endpoint, can serve API documentation or a basic message."""
            return jsonify({
                'status': 'running',
                'version': '1.0',
                'endpoints': {
                    'health': '/health',
                    'columns': '/columns',
                    'compare': '/compare',
                    'validate': '/validate'  # If you add a validation endpoint
                },
                'timestamp': datetime.datetime.utcnow().isoformat()
            }), 200

        @app.route('/health')
        def health_check():
            """Health check endpoint, good for Railway."""
            try:
                # Check memory and disk space
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
                    'pid': os.getpid(),  # Process ID
                    'uptime': datetime.datetime.now().timestamp() - psutil.Process().create_time()  # Uptime
                }
                logger.info(f"Health check passed: {health_data}")
                return jsonify(health_data), 200
            except Exception as e:
                logger.exception("Health check failed")  # Log full exception
                return jsonify({
                    'status': 'unhealthy',
                    'error': str(e),
                    'timestamp': datetime.datetime.utcnow().isoformat()
                }), 500

        @app.before_request
        def before_request():
            """Log incoming requests (good for debugging)."""
            logger.info(f"Request received: {request.method} {request.path}")
            logger.debug(f"Headers: {dict(request.headers)}")  # Log headers for debugging


        @app.route('/columns', methods=['POST'])
        def get_columns():
            """Handle requests for column names."""
            try:
                if 'file' not in request.files:
                    return jsonify({'error': 'No file provided'}), 400

                file = request.files['file']
                if file.filename == '' or not allowed_file(file.filename):
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
                    }), 400  # Bad Request

                file1 = request.files['file1']
                file2 = request.files['file2']

                if not (file1 and allowed_file(file1.filename) and
                        file2 and allowed_file(file2.filename)):
                    return jsonify({
                        'status': 'error',
                        'error': 'Invalid file type'
                    }), 400  # Bad Request

                address_columns1 = request.form.getlist('columns1[]')
                address_columns2 = request.form.getlist('columns2[]')
                threshold = float(request.form.get('threshold', DEFAULT_THRESHOLD))  # Default and type conversion

                if not address_columns1 or not address_columns2:
                    return jsonify({
                        'status': 'error',
                        'error': 'Address columns must be specified'
                    }), 400  # Bad Request

                df1 = load_dataframe(file1)
                df2 = load_dataframe(file2)

                # Call comparison logic
                results = process_address_comparison(df1, df2, address_columns1, address_columns2, threshold)

                # Check if export is requested
                if request.form.get('export') == 'true':
                    # Create an in-memory Excel file
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        # Convert the results to a DataFrame
                        if results: # Handle no matches result
                            # Flatten the 'matches' list within each dictionary.
                            flat_results = []
                            for result_group in results:
                                source_address = result_group['source_address']
                                normalized_source = result_group['normalized_source']
                                for match in result_group['matches']:
                                    flat_results.append({
                                        'source_address': source_address,
                                        'normalized_source' : normalized_source,
                                        'matched_address': match['address'],
                                        'normalized_match': match['normalized'],
                                        'match_score': match['score']
                                    })
                            results_df = pd.DataFrame(flat_results)
                            results_df.to_excel(writer, sheet_name='Comparison Results', index=False)
                        else: # Handle empty results
                            empty_df = pd.DataFrame(columns=['source_address', 'matched_address', 'match_score'])
                            empty_df.to_excel(writer, sheet_name='Comparison Results', index=False)

                    output.seek(0)  # Important: Rewind the buffer
                    # Generate a filename with timestamp
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"comparison_results_{timestamp}.xlsx"
                    cleanup_memory([df1, df2])
                    return send_file(output,  # Pass the in-memory buffer
                                     mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                                     as_attachment=True,
                                     download_name=filename)


                # If not export, return JSON
                 # Ensure proper response format
                response_data = {
                    'status': 'success',
                    'data': results,
                    'metadata': {
                        'file1_rows': len(df1),
                        'file2_rows': len(df2),
                        'matches_found': len(results)  # This might need adjustment for chunking
                    }
                }
                cleanup_memory([df1, df2])

                return jsonify(response_data), 200

            except Exception as e:
                logger.exception("Error in compare_addresses")  # Log the full exception
                return jsonify({
                    'status': 'error',
                    'error': str(e)
                }), 500  # Internal Server Error

        @app.route('/validate', methods=['POST'])
        def validate_address():
            """Validate and normalize a single address (example endpoint)."""
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
                corrections = address_model.suggest_corrections(address)  # Assuming you have this method

                return jsonify({
                    'status': 'success',
                    'data': {
                        'original': address,
                        'normalized': normalized,
                        'suggestions': corrections,
                        'valid': address_model.is_valid_address(normalized)  # And this one
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
            logger.error(f"Route not found: {request.url}")  # Log the requested URL
            return jsonify({'error': 'Resource not found'}), 404

        @app.errorhandler(400)
        def bad_request_error(error):
            """Handle 400 errors."""
            logger.error(f"Bad request: {str(error)}") # Log the error
            return jsonify({'error': str(error)}), 400

        @app.errorhandler(500)
        def internal_error(error):
            """Handle 500 errors."""
            logger.exception("Internal server error")  # Log full exception
            cleanup_memory() # cleanup
            return jsonify({
                'error': 'Internal server error',
                'status': 'error',
                'timestamp': datetime.datetime.utcnow().isoformat()
            }), 500

        logger.info("Flask app created successfully")
        return app

    except Exception as e:
        logger.error(f"Error creating Flask app: {str(e)}", exc_info=True)  # Log with traceback
        raise

if __name__ == '__main__':
    try:
        app = create_app()
        port = int(os.environ.get('PORT', '5000'))  # Default to 5000 if PORT is not set
        logger.info(f"Starting server on port {port}")
        app.run(
            host='0.0.0.0', # Makes the app accessible externally
            port=port,
            debug=DEBUG  # Use the DEBUG flag from config
        )
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}", exc_info=True)  # Log startup errors
        sys.exit(1)  # Exit with an error code