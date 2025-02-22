from flask import Blueprint, request, jsonify, send_file
from utils.address_utils import AddressCorrectionModel
from utils.data_processing import DataProcessor  # Import DataProcessor
from utils.progress import progress_tracker
import pandas as pd
import io
import logging
import os
import datetime
import gc
import uuid
from werkzeug.datastructures import FileStorage
from typing import Dict, List, Optional, Union, Generator
import psutil


address_bp = Blueprint('address_bp', __name__)
# address_model = AddressCorrectionModel()  # Don't initialize here; do it in DataProcessor
DEFAULT_THRESHOLD = 80
CHUNK_SIZE = 10000
logger = logging.getLogger(__name__)
data_processor = DataProcessor()  # Initialize DataProcessor *here*


# -- Helper functions (moved into routes file and simplified)--
def allowed_file(filename: str) -> bool:
    ALLOWED_EXTENSIONS = {'.csv', '.xlsx'}
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS

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


# -- Routes --
@address_bp.route('/columns', methods=['POST'])
def get_columns():
    """Handle column name requests."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        file = request.files['file']
        if not file or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400

        # Use DataProcessor to load the file.  No validation needed here.
        df = data_processor.load_and_validate_file(file, file.filename) # Pass filename for validation
        columns = df.columns.tolist()
        return jsonify({'status': 'success', 'data': columns}), 200
    except Exception as e:
        logger.exception("Error processing columns")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@address_bp.route('/compare', methods=['POST'])
def compare_addresses():
    """Handle address comparison requests."""
    try:
        # Generate a unique job ID
        job_id = str(uuid.uuid4())
        progress_tracker.update_progress(job_id, {  # Start tracking
            'status': 'started',
            'progress': 0
        })

        # Log request
        logger.info('Received comparison request: %s', {
            'files': list(request.files.keys()),
            'form': list(request.form.keys()),
            'content_type': request.content_type,
            'job_id': job_id
        })

        # Validate request (files and columns)
        files = validate_request_files(request.files)
        columns1, columns2 = validate_columns(request.form)
        threshold = float(request.form.get('threshold', DEFAULT_THRESHOLD)) / 100

        # Load DataFrames using DataProcessor (handles file types and errors)
        df1 = data_processor.load_and_validate_file(files['file1'], files['file1'].filename)
        df2 = data_processor.load_and_validate_file(files['file2'], files['file2'].filename)


        # Process the comparison using DataProcessor
        results = data_processor.process_address_comparison(df1, df2, columns1, columns2, threshold, job_id)

        if not validate_comparison_results(results):
            raise ValueError("Invalid results structure")

        # Check if export is requested
        if request.form.get('export') == 'true':
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                if results:
                    results_df = pd.DataFrame(results)
                    results_df.to_excel(writer, sheet_name='Comparison Results', index=False)
                else:
                    empty_df = pd.DataFrame(
                        columns=['source_address', 'normalized_source', 'matched_address',
                                 'normalized_match', 'match_score'])
                    empty_df.to_excel(writer, sheet_name='Comparison Results', index=False)

            output.seek(0)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"comparison_results_{timestamp}.xlsx"

            # Cleanup memory *before* sending the file
            data_processor.cleanup_memory([df1, df2])  # Use DataProcessor's cleanup
            return send_file(output,
                             mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                             as_attachment=True,
                             download_name=filename)


        # Return JSON response
        response_data = {
            'status': 'success',
            'job_id': job_id,
            'data': results,
            'metadata': {
                'file1_rows': len(df1),
                'file2_rows': len(df2),
                'matches_found': len(results)
            }
        }
        data_processor.cleanup_memory([df1, df2]) #Cleanup
        return jsonify(response_data), 200

    except ValueError as ve:
        return jsonify({'status': 'error', 'error': str(ve)}), 400
    except Exception as e:
        logger.exception('Error processing comparison request:')
        return jsonify({'status': 'error', 'error': str(e)}), 500

@address_bp.route('/validate', methods=['POST'])  # Add if you create it
def validate_address():
  pass

@address_bp.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors."""
    logger.error(f"Route not found: {request.url}")
    return jsonify({'error': 'Resource not found'}), 404

@address_bp.errorhandler(400)
def bad_request_error(error):
    """Handle 400 errors."""
    logger.error(f"Bad request: {str(error)}")
    return jsonify({'error': str(error)}), 400