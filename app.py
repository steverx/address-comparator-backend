# --- Configuration ---
import os
import logging
import sys

# Initialize logger before other imports
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)

DEBUG = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS", "https://address-comparator-frontend-production.up.railway.app"
).split(",")
ALLOWED_EXTENSIONS = {".csv", ".xlsx"}
DEFAULT_THRESHOLD = int(os.environ.get("DEFAULT_THRESHOLD", "80"))
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "10000"))  # Process in chunks
MAX_CONTENT_LENGTH = int(
    os.environ.get("MAX_CONTENT_LENGTH", str(500 * 1024 * 1024))
)  # 500MB max-size
CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
PORT = int(os.environ.get("PORT", "5000"))

# --- End Configuration ---

# The rest of your imports and code follow
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
from fuzzywuzzy import fuzz, process
from utils.address_utils import AddressCorrectionModel  # Import your model
import io
import datetime
import gc
from werkzeug.datastructures import FileStorage
from typing import Dict, List, Optional, Union, Generator
import psutil  # For memory usage
import uuid  # For generating job IDs
from utils.progress import progress_tracker  # Import progress tracker
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from celery import Celery  # Import Celery
import bleach  # For input sanitization
from postal.parser import parse_address  # Import libpostal
import tracemalloc  # Import memory profiler
import subprocess
import json

app = Flask(__name__)
CORS(app, origins="https://address-comparator-frontend-production.up.railway.app")

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)

# Configuration
DEBUG = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS", "https://address-comparator-frontend-production.up.railway.app"
).split(",")
ALLOWED_EXTENSIONS = {".csv", ".xlsx"}
DEFAULT_THRESHOLD = int(os.environ.get("DEFAULT_THRESHOLD", "80"))
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "10000"))  # Process in chunks
MAX_CONTENT_LENGTH = int(
    os.environ.get("MAX_CONTENT_LENGTH", str(500 * 1024 * 1024))
)  # 500MB max-size

# Initialize address correction model
address_model = AddressCorrectionModel()

# Configure Celery
celery = Celery(app.name, broker=CELERY_BROKER_URL)

# Start memory profiler
tracemalloc.start()


def cleanup_memory(dataframes: List[pd.DataFrame] = None):
    """Enhanced memory cleanup."""
    if dataframes:
        for df in dataframes:
            del df
    gc.collect()
    memory = psutil.virtual_memory()
    logger.info(f"Memory usage after cleanup: {memory.percent}%")


def sanitize_input(text):
    """Sanitize user input to prevent XSS attacks."""
    return bleach.clean(text)


def normalize_address(address):
    """Normalize address using libpostal."""
    parsed_address = parse_address(address)
    # ... (code to format the parsed address) ...
    return normalized_address


def compile_typescript():
    """Compile TypeScript code to JavaScript."""
    try:
        logger.info("Compiling TypeScript code...")
        subprocess.run(["npm", "install"], cwd="./services", check=True, capture_output=True, text=True)
        subprocess.run(["npm", "run", "build"], cwd="./services", check=True, capture_output=True, text=True)
        logger.info("TypeScript code compiled successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to compile TypeScript code: {e.stderr}")
        raise


def create_app():
    """Application factory pattern"""
    try:
        logger.info("Starting to create Flask app")
        app = Flask(__name__)

        app.config.update(
            ENV="production",
            DEBUG=DEBUG,
            MAX_CONTENT_LENGTH=500 * 1024 * 1024,  # 500MB max-size
        )

        CORS(
            app,
            resources={
                r"/*": {
                    "origins": ALLOWED_ORIGINS,
                    "methods": ["GET", "POST", "OPTIONS"],
                    "allow_headers": ["Content-Type", "Authorization"],
                    "supports_credentials": True,
                }
            },
        )

        # Compile TypeScript code
        compile_typescript()

        def allowed_file(filename: str) -> bool:
            """Check if file extension is allowed."""
            return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS

        def get_match_score(addr1: str, addr2: str) -> float:
            """Calculate fuzzy match score."""
            return address_model.compare_addresses(addr1, addr2)

        def process_dataframe_in_chunks(
            df: pd.DataFrame,
        ) -> Generator[pd.DataFrame, None, None]:
            """Process large DataFrames in chunks."""
            for i in range(0, len(df), CHUNK_SIZE):
                yield df.iloc[i : i + CHUNK_SIZE]

        def load_dataframe(file_storage: FileStorage) -> pd.DataFrame:
            """Load dataframe, handling different file types."""
            try:
                logger.info(f"Loading dataframe from file: {file_storage.filename}")
                file_storage.seek(0)
                if file_storage.filename.endswith(".csv"):
                    return pd.read_csv(file_storage)
                elif file_storage.filename.endswith(".xlsx"):
                    return pd.read_excel(file_storage)
                else:
                    raise ValueError(f"Unsupported file type: {file_storage.filename}")
            except FileNotFoundError:
                logger.error(f"File not found: {file_storage.filename}")
                raise
            except pd.errors.ParserError:
                logger.error(f"Error parsing file: {file_storage.filename}")
                raise
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
            return ", ".join(components) if components else ""

        def validate_request_files(request_files):
            """Validate uploaded files."""
            if not all(key in request_files for key in ["file1", "file2"]):
                raise ValueError("Missing required files")

            files = {key: request_files[key] for key in ["file1", "file2"]}
            for name, file in files.items():
                if not file or not allowed_file(file.filename):
                    raise ValueError(f"Invalid file type for {name}: {file.filename}")
            return files

        def validate_columns(form_data):
            """Validate column selections."""
            columns1 = form_data.getlist('columns1[]')
            columns2 = form_data.getlist('columns2[]')

            if not columns1 or not columns2:
                logger.warning("Missing column selections")
                raise ValueError("Column selections are required")
            if not all(isinstance(col, str) for col in columns1 + columns2):
                logger.warning("Invalid column names")
                raise ValueError("Column names must be strings")
            return columns1, columns2

        def validate_comparison_results(results: List[Dict]) -> bool:
            """Validate comparison results structure."""
            required_keys = {"source_address", "matched_address", "match_score"}

            for result in results:
                if not all(key in result for key in required_keys):
                    return False
                if not isinstance(result["match_score"], (int, float)):
                    return False
            return True

        def process_chunk(
            chunk_df1: pd.DataFrame,
            df2: pd.DataFrame,
            columns1: List[str],
            columns2: List[str],
            threshold: float,
        ) -> List[Dict]:
            """Process a single chunk of df1 against the entire df2."""
            chunk_results = []
            for _, row1 in chunk_df1.iterrows():
                best_match = None
                best_score = threshold  # Initialize with the threshold

                for _, row2 in df2.iterrows():
                    score = address_model.compare_addresses(
                        row1["normalized_address"], row2["normalized_address"]
                    )
                    if score > best_score:
                        best_score = score
                        best_match = {
                            "source_address": row1["combined_address"],
                            "normalized_source": row1["normalized_address"],
                            "matched_address": row2["combined_address"],
                            "normalized_match": row2["normalized_address"],
                            "match_score": score,
                        }
                if best_match:
                    chunk_results.append(best_match)
            return chunk_results

        def process_address_comparison(
            df1: pd.DataFrame,
            df2: pd.DataFrame,
            columns1: List[str],
            columns2: List[str],
            threshold: float,
        ) -> List[Dict]:
            """Process address comparison with optimized memory usage."""
            try:
                logger.info(
                    f"Starting address comparison for {len(df1)} x {len(df2)} addresses"
                )
                # Combine selected address columns
                df1["combined_address"] = df1.apply(
                    lambda row: combine_address_components(row, columns1), axis=1
                )
                df2["combined_address"] = df2.apply(
                    lambda row: combine_address_components(row, columns2), axis=1
                )

                # Normalize addresses
                df1["normalized_address"] = df1["combined_address"].apply(
                    lambda addr: normalize_address(addr)
                )
                df2["normalized_address"] = df2["combined_address"].apply(
                    lambda addr: normalize_address(addr)
                )

                # Extract addresses for ML service
                addresses1 = df1["normalized_address"].tolist()
                addresses2 = df2["normalized_address"].tolist()

                # Combine addresses for vectorization
                all_addresses = addresses1 + addresses2

                # Vectorize addresses
                vectorizer = TfidfVectorizer()
                tfidf_matrix = vectorizer.fit_transform(all_addresses)

                # Split vectorized addresses
                tfidf_matrix_1 = tfidf_matrix[:len(addresses1)]
                tfidf_matrix_2 = tfidf_matrix[len(addresses1):]

                results = []
                total_comparisons = len(addresses1) * len(addresses2)
                comparison_count = 0

                for i, vec1 in enumerate(tfidf_matrix_1):
                    for j, vec2 in enumerate(tfidf_matrix_2):
                        similarity_score = cosine_similarity(vec1.toarray().flatten(), vec2.toarray().flatten())
                        comparison_count += 1

                        if similarity_score >= threshold:
                            results.append({
                                "source_address": df1["combined_address"].iloc[i],
                                "normalized_source": df1["normalized_address"].iloc[i],
                                "matched_address": df2["combined_address"].iloc[j],
                                "normalized_match": df2["normalized_address"].iloc[j],
                                "match_score": similarity_score,
                            })

                        progress = int((comparison_count / total_comparisons) * 100)
                        progress_tracker.update_progress(job_id, {'progress': progress})

                return results

            except Exception as e:
                logger.exception("Error in address comparison")
                raise

        @celery.task
        def process_address_comparison_task(df1, df2, columns1, columns2, threshold):
            """Asynchronous task for address comparison."""
            results = process_address_comparison(
                df1, df2, columns1, columns2, threshold
            )
            return results

        @app.route("/")
        def index():
            """Root endpoint."""
            return (
                jsonify(
                    {
                        "status": "running",
                        "version": "1.0",
                        "endpoints": {
                            "health": "/health",
                            "columns": "/columns",
                            "compare": "/compare",
                            "validate": "/validate",
                        },
                        "timestamp": datetime.datetime.utcnow().isoformat(),
                    }
                ),
                200,
            )

        @app.route("/health")
        def health_check():
            """Health check endpoint."""
            try:
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage("/")
                health_data = {
                    "status": "healthy",
                    "timestamp": datetime.datetime.utcnow().isoformat(),
                    "memory": {
                        "total": memory.total,
                        "available": memory.available,
                        "percent": memory.percent,
                    },
                    "disk": {
                        "total": disk.total,
                        "free": disk.free,
                        "percent": disk.percent,
                    },
                    "pid": os.getpid(),
                    "uptime": datetime.datetime.now().timestamp()
                    - psutil.Process().create_time(),
                }
                logger.info(f"Health check passed: {health_data}")
                return jsonify(health_data), 200
            except Exception as e:
                logger.exception("Health check failed")
                return (
                    jsonify(
                        {
                            "status": "unhealthy",
                            "error": str(e),
                            "timestamp": datetime.datetime.utcnow().isoformat(),
                        }
                    ),
                    500,
                )

        @app.before_request
        def before_request():
            """Log incoming requests."""
            logger.info(f"Request received: {request.method} {request.path}")
            logger.debug(f"Headers: {dict(request.headers)}")

        @app.before_request
        def log_request():
            logger.info(f"Received request: {request.method} {request.url}")
            logger.debug(f"Request headers: {request.headers}")
            logger.debug(f"Request data: {request.get_data()}")

        @app.route("/columns", methods=["POST"])
        def get_columns():
            """Handle column name requests."""
            try:
                if "file" not in request.files:
                    return jsonify({"error": "No file provided"}), 400
                file = request.files["file"]
                if not file or not allowed_file(file.filename):
                    return jsonify({"error": "Invalid file type"}), 400
                df = load_dataframe(file)
                columns = df.columns.tolist()
                return jsonify({"status": "success", "data": columns}), 200
            except Exception as e:
                logger.exception("Error processing columns")
                return jsonify({"status": "error", "error": str(e)}), 500

        @app.route('/compare', methods=['POST'])
        def compare_addresses():
            try:
                logger.info("Received address comparison request")
                if 'file1' not in request.files or 'file2' not in request.files:
                    logger.warning("Missing file(s) in request")
                    return jsonify({'status': 'error', 'error': 'Missing file(s)'}), 400

                file1 = request.files['file1']
                file2 = request.files['file2']

                logger.info(f"Uploaded files: {file1.filename}, {file2.filename}")

                # Generate a unique job ID
                job_id = str(uuid.uuid4())
                progress_tracker.update_progress(
                    job_id, {"status": "started", "progress": 0}  # Start tracking
                )
                # Log incoming request
                logger.info(
                    "Received comparison request: %s",
                    {
                        "files": list(
                            request.files.keys()
                        ),  # Safer way to get file keys
                        "form": list(request.form.keys()),  # Safer way to get form keys
                        "content_type": request.content_type,
                        "job_id": job_id,
                    },
                )
                logger.info(f"Processing comparison request with job ID: {job_id}")
                logger.debug(
                    f"Request data: files={list(request.files.keys())}, form={request.form}"
                )

                # Validate request data
                files = validate_request_files(request.files)
                columns1, columns2 = validate_columns(request.form)
                threshold = (
                    float(request.form.get("threshold", DEFAULT_THRESHOLD)) / 100
                )  # Getting it in the 0-1 range.
                logger.info(f"Files: {files}")

                df1 = load_dataframe(files["file1"])
                df2 = load_dataframe(files["file2"])

                task = process_address_comparison_task.delay(
                    df1.to_dict(), df2.to_dict(), columns1, columns2, threshold
                )
                return jsonify({"status": "success", "task_id": task.id}), 200

            except ValueError as ve:  # Catch validation errors
                logger.warning(f"Validation error: {str(ve)}")
                return (
                    jsonify({"status": "error", "error": str(ve)}),
                    400,
                )  # Bad Request
            except Exception as e:
                logger.exception("Error processing comparison request:")
                # Optionally, include more details in the response
                return jsonify({"status": "error", "error": str(e), "details": traceback.format_exc()}), 500

        @app.route("/validate", methods=["POST"])
        def validate_address():
            """Validate and normalize a single address."""
            try:
                if not request.is_json:
                    return (
                        jsonify({"status": "error", "error": "Request must be JSON"}),
                        400,
                    )
                data = request.get_json()
                address = sanitize_input(data.get("address"))
                if not address:
                    return (
                        jsonify({"status": "error", "error": "Address is required"}),
                        400,
                    )
                normalized = address_model.normalize_address(address)
                corrections = address_model.suggest_corrections(
                    address
                )  # Assuming you have it
                return (
                    jsonify(
                        {
                            "status": "success",
                            "data": {
                                "original": address,
                                "normalized": normalized,
                                "suggestions": corrections,
                                "valid": address_model.is_valid_address(normalized),
                            },
                        }
                    ),
                    200,
                )
            except Exception as e:
                logger.exception("Error in address validation")
                return jsonify({"status": "error", "error": str(e)}), 500

        @app.errorhandler(404)
        def not_found_error(error):
            """Handle 404 errors."""
            logger.error(f"Route not found: {request.url}")
            return jsonify({"error": "Resource not found"}), 404

        @app.errorhandler(400)
        def bad_request_error(error):
            """Handle 400 errors."""
            logger.error(f"Bad request: {str(error)}")
            return jsonify({"error": str(error)}), 400

        @app.errorhandler(500)
        def internal_error(error):
            """Handle 500 errors."""
            logger.exception("Internal server error")
            cleanup_memory()
            return (
                jsonify(
                    {
                        "error": "Internal server error",
                        "status": "error",
                        "timestamp": datetime.datetime.utcnow().isoformat(),
                    }
                ),
                500,
            )

        logger.info("Flask app created successfully")
        return app

    except Exception as e:
        logger.error(f"Error creating Flask app: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    try:
        app = create_app()
        port = int(os.environ.get("PORT", "5000"))
        logger.info(f"Starting server on port {port}")
        app.run(host="0.0.0.0", port=port, debug=DEBUG)
        
        # Take memory snapshot
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')

        print("[ Top 10 ]")
        for stat in top_stats[:10]:
            print(stat)
    except Exception as e:
        logger.exception("Failed to start server:")  # Log the exception with traceback
        sys.exit(1)
