from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
from fuzzywuzzy import fuzz
import utils.address_utils as address_utils
import io
import logging
import os
import re

# --- Constants ---
ALLOWED_EXTENSIONS = {'.csv', '.xlsx'}
DEFAULT_PARSER = 'usaddress'
DEFAULT_THRESHOLD = 80

# --- Setup Flask App ---
app = Flask(__name__)
CORS(app, resources={r"/*": {
    "origins": [
        "https://address-comparator-frontend-production.up.railway.app",
        "http://localhost:3000",
        "*"
    ], 
    "allow_headers": ["Content-Type", "Authorization", "Origin"],
    "methods": ["GET", "POST", "OPTIONS"]
}})

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add OPTIONS method handler for preflight requests
@app.route('/columns', methods=['OPTIONS'])
@app.route('/compare', methods=['OPTIONS'])
def handle_options():
    response = jsonify()
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,Origin')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

def get_match_score(addr1, addr2):
    """Calculate match score using multiple metrics."""
    # Basic cleaning
    addr1 = addr1.lower().strip()
    addr2 = addr2.lower().strip()
    
    # Extract components
    parts1 = addr1.split(',')
    parts2 = addr2.split(',')
    
    # Street address comparison (50%)
    street_score = fuzz.token_sort_ratio(parts1[0], parts2[0]) if parts1 and parts2 else 0
    
    # Number comparison (20%)
    nums1 = set(re.findall(r'\d+', parts1[0])) if parts1 else set()
    nums2 = set(re.findall(r'\d+', parts2[0])) if parts2 else set()
    num_score = 100 if nums1 and nums2 and nums1.intersection(nums2) else 0
    
    # ZIP code comparison (20%)
    zip1 = re.search(r'\b\d{5}\b', addr1)
    zip2 = re.search(r'\b\d{5}\b', addr2)
    zip_score = 100 if (zip1 and zip2 and zip1.group(0) == zip2.group(0)) else 0
    
    # State comparison (10%)
    state1 = re.search(r'\b[A-Za-z]{2}\b', parts1[-1]) if len(parts1) > 1 else None
    state2 = re.search(r'\b[A-Za-z]{2}\b', parts2[-1]) if len(parts2) > 1 else None
    state_score = 100 if (state1 and state2 and state1.group(0).upper() == state2.group(0).upper()) else 0
    
    # Calculate weighted score
    score = (
        (street_score * 0.5) +
        (num_score * 0.2) +
        (zip_score * 0.2) +
        (state_score * 0.1)
    )
    
    return round(score, 1)

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS

def load_dataframe(file_storage):
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

def combine_address_components(row, columns, is_excel=False):
    """Combine address components."""
    components = []
    
    # Remove duplicate columns for Excel files
    if is_excel and any(col.endswith('1') for col in columns):
        columns = [col for col in columns if not col.endswith('1')]
    
    # Process address components
    for col in columns:
        if pd.notna(row[col]):
            val = str(row[col]).strip()
            if val:
                components.append(val)
    
    return ', '.join(components) if components else ''

# --- Routes ---
@app.route('/columns', methods=['POST'])
def get_columns():
    """Handle column name requests."""
    try:
        # Log incoming request details
        logger.info("Columns request received")
        logger.info(f"Request files: {request.files}")
        
        if 'file' not in request.files:
            logger.error("No file in request")
            return jsonify({'error': 'Missing file'}), 400
            
        file = request.files['file']
        logger.info(f"Received file: {file.filename}")
        
        if not file or not allowed_file(file.filename):
            logger.error(f"Invalid file type: {file.filename}")
            return jsonify({'error': 'Invalid file type'}), 400
            
        df = load_dataframe(file)
        columns = df.columns.tolist()
        logger.info(f"Columns found: {columns}")
        
        return jsonify({'columns': columns}), 200
        
    except Exception as e:
        logger.exception("Error processing columns request")
        return jsonify({'error': str(e)}), 500

@app.route('/compare', methods=['POST'])
def compare_addresses():
    """Handle address comparison requests."""
    try:
        # Validate inputs
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
            
        # Load and process files
        df1 = load_dataframe(file1)
        df2 = load_dataframe(file2)
        
        # Process addresses
        results = []
        is_excel1 = file1.filename.endswith('.xlsx')
        is_excel2 = file2.filename.endswith('.xlsx')
        
        # Process first file
        addresses1 = []
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
        addresses2 = []
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
            output = io.BytesIO()
            if results:
                df_results = pd.DataFrame(results)
                
                # Use openpyxl engine
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df_results.to_excel(writer, sheet_name='Address Matches', index=False)
                
                output.seek(0)
                
                return send_file(
                    output,
                    mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    as_attachment=True,
                    download_name='address_matches.xlsx'
                )
            else:
                return jsonify({'error': 'No results to export'}), 400
        
        return jsonify(results), 200
        
    except Exception as e:
        logger.exception("Comprehensive error in compare_addresses")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)