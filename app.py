from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
from fuzzywuzzy import fuzz, process
import utils.address_utils as address_utils
import io
import hashlib
import logging
import os

# --- Constants ---
ALLOWED_EXTENSIONS = {'.csv', '.xlsx'}
DEFAULT_PARSER = 'usaddress'
DEFAULT_THRESHOLD = 80
CACHE = {}  # Simple in-memory cache: {file_hash: {address: (cleaned_address, confidence)}}

# --- Setup Flask App ---
app = Flask(__name__)
CORS(app)

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---
def allowed_file(filename):
    """Checks if the file extension is allowed."""
    return '.' in filename and os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS

def generate_file_hash(file_storage):
    """Generates a SHA-256 hash of the file content."""
    file_storage.seek(0)  # Ensure we're at the beginning of the file
    content = file_storage.read()
    if isinstance(content, str):  # Handle text-mode files (like CSV)
        content = content.encode('utf-8')
    file_hash = hashlib.sha256(content).hexdigest()
    file_storage.seek(0)  # Reset file pointer after reading
    return file_hash

def load_dataframe(file_storage):
    """Loads a DataFrame from a file storage object (CSV or Excel)."""
    try:
        if file_storage.filename.endswith('.csv'):
            return pd.read_csv(file_storage)
        elif file_storage.filename.endswith('.xlsx'):
            return pd.read_excel(file_storage)
        else:
            raise ValueError(f"Unsupported file type: {file_storage.filename}")
    except Exception as e:
        logging.error(f"Error reading file {file_storage.filename}: {e}")
        raise

def combine_and_clean_address_csv(row, columns, parser):
    """Combines and cleans address components for CSV files."""
    combined_address = ' '.join(str(row[col]) for col in columns if pd.notna(row[col]) and str(row[col]).strip() != '')
    cleaned_address, confidence = address_utils.clean_address(combined_address, parser=parser)
    return combined_address, cleaned_address, confidence

def clean_excel_address(address, parser):
    """Cleans and parses the address from the Excel file's combined address column."""
    cleaned_address, confidence = address_utils.clean_address(address, parser=parser)
    return cleaned_address, confidence

def get_cached_address(file_hash, combined_address):
    """Retrieves cleaned address and confidence from cache, if available."""
    if file_hash in CACHE and combined_address in CACHE[file_hash]:
        return CACHE[file_hash][combined_address]
    return None, None

def cache_address(file_hash, combined_address, cleaned_address, confidence):
    """Stores the cleaned address and confidence in the cache."""
    if file_hash not in CACHE:
        CACHE[file_hash] = {}
    CACHE[file_hash][combined_address] = (cleaned_address, confidence)

# --- Flask Routes ---

@app.route('/columns', methods=['POST'])
def get_columns():
    """Handles requests for column names from uploaded files."""
    logging.info("Received request at /columns")
    try:
        if 'file' not in request.files:
            logging.error("Missing file in request")
            return jsonify({'error': 'Missing file'}), 400

        file = request.files['file']
        logging.info(f"File received: {file.filename}")

        if not allowed_file(file.filename):
            logging.error(f"Invalid file type: {file.filename}")
            return jsonify({'error': 'Invalid file type'}), 400

        try:
            if file.filename.endswith('.csv'):
                df = pd.read_csv(file)
                logging.info("Successfully read CSV file")
            elif file.filename.endswith('.xlsx'):
                df = pd.read_excel(file)
                logging.info("Successfully read Excel file")
            else:
                logging.error(f"invalid file type: {file.filename}")
                return jsonify({'error': 'Invalid File Type'}), 400
            columns = df.columns.tolist()
            logging.info(f"Columns found: {columns}")
            return jsonify({'columns': columns}), 200

        except pd.errors.ParserError as e:
            logging.error(f"Pandas ParserError reading file: {e}")
            return jsonify({'error': f'Error parsing file: {e}'}), 400
        except pd.errors.EmptyDataError as e:
            logging.error(f"Pandas EmptyDataError reading file: {e}")
            return jsonify({'error': f'Error, Empty file: {e}'}), 400
        except FileNotFoundError as e:
            logging.error(f"FileNotFoundError: {e}")
            return jsonify({'error': f'File Not Found: {e}'}), 400
        except Exception as e:
            logging.error(f"General error reading file: {e}")
            return jsonify({'error': f'Error reading file: {e}'}), 400

    except Exception as e:
        logging.exception("An unexpected error occurred in /columns")
        return jsonify({'error': 'An unexpected error occurred.'}), 500


@app.route('/compare', methods=['POST'])
def compare_addresses():
    """Handles address comparison requests."""
    logging.info("Received request at /compare")
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
            return jsonify({'error': 'No address columns selected.'}), 400

        try:
            df1 = load_dataframe(file1)
            df2 = load_dataframe(file2)
        except Exception as e:
            return jsonify({'error': str(e)}), 400

        if not all(col in df1.columns for col in address_columns1) or \
           not all(col in df2.columns for col in address_columns2):
            return jsonify({'error': 'Invalid column names selected.'}), 400


        file1_hash = generate_file_hash(file1)
        file2_hash = generate_file_hash(file2)

        # --- File 1 Processing (Excel) ---
        if file1.filename.endswith(".xlsx") and 'GPOEndUserAddress1' in df1.columns:
            logging.info("Using GPOEndUserAddress1 for Excel file.")
            df1['combined_address'] = df1['GPOEndUserAddress1'].fillna('').astype(str)
            df1['cleaned_address'], df1['confidence1'] = zip(*df1['combined_address'].apply(lambda x: address_utils.clean_address(x, parser=parser)))

        # --- File 1 Processing (CSV) ---
        else:
            logging.info("Combining selected columns for CSV file.")
            if not all(col in df1.columns for col in address_columns1):
                return jsonify({'error': 'Invalid column names selected for File 1.'}), 400
            df1[['combined_address', 'cleaned_address', 'confidence1']] = df1.apply(
                lambda row: combine_and_clean_address_csv(row, address_columns1, parser), axis=1
            ).apply(pd.Series)
        for index, row in df1.iterrows():
            cache_address(file1_hash, row['combined_address'], row['cleaned_address'], row['confidence1'])

        # --- File 2 Processing (Excel) ---
        if file2.filename.endswith(".xlsx") and 'GPOEndUserAddress1' in df2.columns:
            logging.info("Using GPOEndUserAddress1 for Excel file.")
            df2['combined_address'] = df2['GPOEndUserAddress1'].fillna('').astype(str)
            df2['cleaned_address'], df2['confidence2'] = zip(*df2['combined_address'].apply(lambda x: address_utils.clean_address(x, parser=parser)))

        # --- File 2 Processing (CSV) ---
        else:
            logging.info("Combining selected columns for CSV file.")
            if not all(col in df2.columns for col in address_columns2):
              return jsonify({'error': 'Invalid column names selected for File 2.'}), 400
            df2[['combined_address', 'cleaned_address', 'confidence2']] = df2.apply(
                lambda row: combine_and_clean_address_csv(row, address_columns2, parser), axis=1
            ).apply(pd.Series)

        for index, row in df2.iterrows():
            cache_address(file2_hash, row['combined_address'], row['cleaned_address'], row['confidence2'])

        results = []
        for _, row1 in df1.iterrows():
            best_match = process.extractOne(row1['cleaned_address'], df2['cleaned_address'], scorer=fuzz.token_sort_ratio)
            if best_match:
                match_score = best_match[1]
                matched_index = best_match[2]
                matched_row = df2.iloc[matched_index]
                parsing_confidence = (row1['confidence1'] + matched_row['confidence2']) / 2
                weighted_match_score = int(round(match_score * parsing_confidence))

                results.append({
                    'address1': row1['combined_address'],  # Use combined address
                    'address2': matched_row['combined_address'],  # Use combined address
                    'original_match_score': match_score,
                    'match_score': weighted_match_score,
                    'parsing_confidence': round(parsing_confidence, 2),
                    'original_address2': matched_row['combined_address']  # Keep for reference
                })

        filtered_results = [r for r in results if r['match_score'] >= threshold]

        # --- Export to Excel ---
        if request.form.get('export') == 'true':
            logging.info("Exporting results to Excel...")
            output = io.BytesIO()
            try:
                df_results = pd.DataFrame(filtered_results)
                if not df_results.empty:
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        df_results.to_excel(writer, sheet_name='Address Matches', index=False)
                    output.seek(0)

                logging.info("Sending Excel file...")
                return send_file(
                    output,
                    mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    download_name='address_matches.xlsx'
                )
            except Exception as e:
                logging.exception("Error during Excel export:")
                return jsonify({'error': f'Error exporting to Excel: {str(e)}'}), 500
        else:
            return jsonify(filtered_results), 200

    except Exception as e:
        logging.exception("An unexpected error occurred during comparison.")
        return jsonify({'error': 'An unexpected error occurred.'}), 500

if __name__ == '__main__':
    app.run(debug=True)