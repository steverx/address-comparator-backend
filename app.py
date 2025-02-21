from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
from fuzzywuzzy import fuzz, process
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
import uuid  # For generating job IDs
from utils.progress import progress_tracker  # Import progress tracker


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
ALLOWED_ORIGINS = ["https://address-comparator-frontend-production.up.railway.app"]  # Replace with your frontend URL
ALLOWED_EXTENSIONS = {'.csv', '.xlsx'}
DEFAULT_THRESHOLD = 80
CHUNK_SIZE = 10000  # Process in chunks

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

        app.config.update(
            ENV='production',
            DEBUG=DEBUG,
            MAX_CONTENT_LENGTH=500 * 1024 * 1024  # 500MB max-size
        )

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
            """Calculate fuzzy match score."""
            return address_model.compare_addresses(addr1, addr2)

        def process_dataframe_in_chunks(df: pd.DataFrame) -> Generator[pd.DataFrame, None, None]:
            """Process large DataFrames in chunks."""
            for i in range(0, len(df), CHUNK_SIZE):
                yield df.iloc[i:i + CHUNK_SIZE]

        def load_dataframe(file_storage: FileStorage) -> pd.DataFrame:
            """Load dataframe, handling different file types."""
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

        def validate_request_files(request_files):
            """Validate uploaded files."""
            if not all(key in request_files for key in ['file1', 'file2']):
                raise ValueError("Missing required files")

            files = {key: request_files[key] for key in ['file1', 'file2']}
            for name, file in files.items():
                if not file or not allowed_file(file.filename):
                    raise ValueError(f"Invalid file type for {name}: {file.filename}")
            return files

        def validate_columns(form_data):
            """Validate column selections."""
            columns1 = form_data.getlist('columns1[]')
            columns2 = form_data.getlist('columns2[]')

            if not columns1 or not columns2:
                raise ValueError("Column selections are required")
            return columns1, columns2

        def validate_comparison_results(results: List[Dict]) -> bool:
            """Validate comparison results structure."""
            required_keys = {'source_address', 'matched_address', 'match_score'}

            for result in results:
                if not all(key in result for key in required_keys):
                    return False
                if not isinstance(result['match_score'], (int, float)):
                    return False
            return True
        def process_chunk(chunk_df1: pd.DataFrame, df2: pd.DataFrame, columns1: List[str], columns2: List[str], threshold: float) -> List[Dict]:
            """Process a single chunk of df1 against the entire df2."""
            chunk_results = []
            for _, row1 in chunk_df1.iterrows():
                best_match = None
                best_score = threshold  # Initialize with the threshold

                for _, row2 in df2.iterrows():
                    score = address_model.compare_addresses(row1['normalized_address'], row2['normalized_address'])
                    if score > best_score:
                        best_score = score
                        best_match = {
                            'source_address': row1['combined_address'],
                            'normalized_source': row1['normalized_address'],
                            'matched_address': row2['combined_address'],
                            'normalized_match': row2['normalized_address'],
                            'match_score': score
                        }
                if best_match:
                    chunk_results.append(best_match)
            return chunk_results


        def process_address_comparison(df1: pd.DataFrame, df2: pd.DataFrame,
                                        columns1: List[str], columns2: List[str],
                                        threshold: float) -> List[Dict]:
            """Process address comparison with optimized memory usage."""
            try:
                # Combine selected address columns
                df1['combined_address'] = df1.apply(lambda row: combine_address_components(row, columns1), axis=1)
                df2['combined_address'] = df2.apply(lambda row: combine_address_components(row, columns2), axis=1)

                # Normalize addresses
                df1['normalized_address'] = df1['combined_address'].apply(lambda addr: address_model.normalize_address(addr))
                df2['normalized_address'] = df2['combined_address'].apply(lambda addr: address_model.normalize_address(addr))

                results = []
                total_chunks = (len(df1) // CHUNK_SIZE) + 1  # Use the global constant

                for i, chunk_df1 in enumerate(process_dataframe_in_chunks(df1)):
                    chunk_results = process_chunk(chunk_df1, df2, columns1, columns2, threshold) # Pass the necessary arguments
                    results.extend(chunk_results)
                    logger.info(f"Processed chunk {i + 1}/{total_chunks}")  # Simplified logging

                return results

            except Exception as e:
                logger.exception("Error in address comparison")
                raise




        @app.route('/')
        def index():
            """Root endpoint."""
            return jsonify({
                'status': 'running', 'version': '1.0', 'endpoints': {'health': '/health', 'columns': '/columns',
                                                                     'compare': '/compare', 'validate': '/validate'},
                'timestamp': datetime.datetime.utcnow().isoformat()}), 200

        @app.route('/health')
        def health_check():
            """Health check endpoint."""
            try:
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                health_data = {'status': 'healthy', 'timestamp': datetime.datetime.utcnow().isoformat(),
                               'memory': {'total': memory.total, 'available': memory.available, 'percent': memory.percent},
                               'disk': {'total': disk.total, 'free': disk.free, 'percent': disk.percent},
                               'pid': os.getpid(),
                               'uptime': datetime.datetime.now().timestamp() - psutil.Process().create_time()}
                logger.info(f"Health check passed: {health_data}")
                return jsonify(health_data), 200
            except Exception as e:
                logger.exception("Health check failed")
                return jsonify({'status': 'unhealthy', 'error': str(e),
                                'timestamp': datetime.datetime.utcnow().isoformat()}), 500

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
                return jsonify({'status': 'success', 'data': columns}), 200
            except Exception as e:
                logger.exception("Error processing columns")
                return jsonify({'status': 'error', 'error': str(e)}), 500


        @app.route('/compare', methods=['POST'])
        def compare_addresses():
            """Handle address comparison requests with validation."""
            try:
                # Generate a unique job ID
                job_id = str(uuid.uuid4())
                progress_tracker.update_progress(job_id, { # Start tracking
                    'status': 'started',
                    'progress': 0
                })
                # Log incoming request
                logger.info('Received comparison request: %s', {
                    'files': list(request.files.keys()),  # Safer way to get file keys
                    'form': list(request.form.keys()),   # Safer way to get form keys
                    'content_type': request.content_type,
                    'job_id': job_id
                })


                # Validate request data
                files = validate_request_files(request.files)
                columns1, columns2 = validate_columns(request.form)
                threshold = float(request.form.get('threshold', DEFAULT_THRESHOLD))/100 #Getting it in the 0-1 range.
                logger.info(f'Files: {files}')

                df1 = load_dataframe(files['file1'])
                df2 = load_dataframe(files['file2'])

                results = process_address_comparison(df1, df2, columns1, columns2, threshold)


                if not validate_comparison_results(results):
                    raise ValueError("Invalid results structure")
                # Check if export is requested
                if request.form.get('export') == 'true':
                    # Create an in-memory Excel file
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        # Convert the results to a DataFrame
                        if results:
                            results_df = pd.DataFrame(results)
                            results_df.to_excel(writer, sheet_name='Comparison Results', index=False)
                        else:  # Handle the case where there are no results
                            empty_df = pd.DataFrame(
                                columns=['source_address', 'normalized_source', 'matched_address',
                                         'normalized_match', 'match_score'])
                            empty_df.to_excel(writer, sheet_name='Comparison Results',
                                              index=False)  # Create empty sheet

                    output.seek(0)  # Important: Rewind the buffer to the beginning
                    # Generate dynamic filename
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"comparison_results_{timestamp}.xlsx"
                    cleanup_memory([df1, df2])
                    return send_file(output,  # Send the in-memory file
                                     mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                                     as_attachment=True,
                                     download_name=filename)

                response_data = {
                    'status': 'success',
                    'job_id': job_id,  # Return the job ID
                    'data': results,
                    'metadata': {  # Include metadata
                        'file1_rows': len(df1),
                        'file2_rows': len(df2),
                        'matches_found': len(results)
                    }
                }
                cleanup_memory([df1, df2])
                return jsonify(response_data), 200

            except ValueError as ve:  # Catch validation errors
                return jsonify({'status': 'error', 'error': str(ve)}), 400  # Bad Request
            except Exception as e:
                logger.exception('Error processing comparison request:')
                return jsonify({'status': 'error', 'error': str(e)}), 500

        @app.route('/validate', methods=['POST'])
        def validate_address():
            """Validate and normalize a single address."""
            try:
                if not request.is_json:
                    return jsonify({'status': 'error', 'error': 'Request must be JSON'}), 400
                data = request.get_json()
                address = data.get('address')
                if not address:
                    return jsonify({'status': 'error', 'error': 'Address is required'}), 400
                normalized = address_model.normalize_address(address)
                corrections = address_model.suggest_corrections(address)  # Assuming you have it
                return jsonify({
                    'status': 'success',
                    'data': {'original': address, 'normalized': normalized, 'suggestions': corrections,
                             'valid': address_model.is_valid_address(normalized)}
                }), 200
            except Exception as e:
                logger.exception("Error in address validation")
                return jsonify({'status': 'error', 'error': str(e)}), 500

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
            return jsonify({'error': 'Internal server error', 'status': 'error',
                            'timestamp': datetime.datetime.utcnow().isoformat()}), 500

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
        app.run(host='0.0.0.0', port=port, debug=DEBUG)
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}", exc_info=True)
        sys.exit(1)