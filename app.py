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
from werkzeug.datastructures import FileStorage
from typing import Dict, List, Optional, Union
from waitress import serve 

# Environment Configuration
DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
PORT = int(os.environ.get('PORT', 8000))
ALLOWED_ORIGINS = os.environ.get('ALLOWED_ORIGINS', 
    'https://address-comparator-frontend-production.up.railway.app').split(',')

# Constants
ALLOWED_EXTENSIONS = {'.csv', '.xlsx'}
DEFAULT_PARSER = 'usaddress'
DEFAULT_THRESHOLD = 80

# Initialize Flask app
app = Flask(__name__)

# Production configuration
app.config.update(
    ENV='production',
    DEBUG=DEBUG,
    MAX_CONTENT_LENGTH=16 * 1024 * 1024  # 16MB max-size
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Simplified logging level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]  # Remove file handler for Railway
)
logger = logging.getLogger(__name__)

# CORS configuration
CORS(app, 
    resources={r"/*": {
        "origins": ALLOWED_ORIGINS,
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Accept", "Origin"],
        "supports_credentials": True
    }})

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint."""
    return jsonify({'status': 'healthy'}), 200

def get_match_score(addr1: str, addr2: str) -> float:
    """Calculate match score using multiple metrics."""
    addr1 = addr1.lower().strip()
    addr2 = addr2.lower().strip()
    
    parts1 = addr1.split(',')
    parts2 = addr2.split(',')
    
    street_score = fuzz.token_sort_ratio(parts1[0], parts2[0]) if parts1 and parts2 else 0
    
    nums1 = set(re.findall(r'\d+', parts1[0])) if parts1 else set()
    nums2 = set(re.findall(r'\d+', parts2[0])) if parts2 else set()
    num_score = 100 if nums1 and nums2 and nums1.intersection(nums2) else 0
    
    zip1 = re.search(r'\b\d{5}\b', addr1)
    zip2 = re.search(r'\b\d{5}\b', addr2)
    zip_score = 100 if (zip1 and zip2 and zip1.group(0) == zip2.group(0)) else 0
    
    state1 = re.search(r'\b[A-Za-z]{2}\b', parts1[-1]) if len(parts1) > 1 else None
    state2 = re.search(r'\b[A-Za-z]{2}\b', parts2[-1]) if len(parts2) > 1 else None
    state_score = 100 if (state1 and state2 and state1.group(0).upper() == state2.group(0).upper()) else 0
    
    score = (
        (street_score * 0.5) +
        (num_score * 0.2) +
        (zip_score * 0.2) +
        (state_score * 0.1)
    )
    
    return round(score, 1)

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS

def load_dataframe(file_storage: FileStorage) -> pd.DataFrame:
    """Load dataframe with proper encoding handling."""
    try:
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

@app.before_request
def log_request_info():
    """Log request details for debugging."""
    logger.info('Headers: %s', dict(request.headers))
    if request.method != 'OPTIONS':
        logger.info('Body: %s', request.get_data())

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
        file1 = request.files.get('file1')
        file2 = request.files.get('file2')
        
        if not file1 or not file2:
            return jsonify({'error': 'Missing files'}), 400
            
        if not (allowed_file(file1.filename) and allowed_file(file2.filename)):
            return jsonify({'error': 'Invalid file types'}), 400
            
        address_columns1 = request.form.getlist('addressColumns1')
        address_columns2 = request.form.getlist('addressColumns2')
        parser = request.form.get('parser', DEFAULT_PARSER)
        threshold = int(request.form.get('threshold', DEFAULT_THRESHOLD))
        
        if not address_columns1 or not address_columns2:
            return jsonify({'error': 'No address columns selected'}), 400
            
        df1 = load_dataframe(file1)
        df2 = load_dataframe(file2)
        
        results = []
        is_excel1 = file1.filename.endswith('.xlsx')
        is_excel2 = file2.filename.endswith('.xlsx')
        
        addresses1 = []
        addresses2 = []
        
        # Process first file
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
        
        # Process second file
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
        
        logger.info(f"Processed {len(addresses1)} addresses from file1")
        logger.info(f"Processed {len(addresses2)} addresses from file2")
        
        # Find matches
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
        
        # Handle export request
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

@app.errorhandler(Exception)
def handle_error(error):
    """Handle unhandled exceptions."""
    logger.exception("Unhandled error")
    return jsonify({'error': str(error)}), 500

if __name__ == '__main__':
    logger.info(f"Starting server on port {PORT}")
    logger.info(f"Debug mode: {DEBUG}")
    logger.info(f"Allowed origins: {ALLOWED_ORIGINS}")
    
    if os.environ.get('RAILWAY_ENVIRONMENT') == 'production':
        logger.info("Starting production server with Waitress")
        serve(app, host='0.0.0.0', port=PORT, threads=4)
    else:
        logger.info("Starting development server")
        app.run(host='0.0.0.0', port=PORT, debug=DEBUG)