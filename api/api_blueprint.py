from flask import Blueprint, jsonify, request, current_app, g
import datetime
import logging
import functools
import traceback
import time
from typing import Dict, Any, Callable, TypeVar, Tuple, List, Optional
import bleach
from werkzeug.exceptions import HTTPException
import pandas as pd
import json
import uuid
from utils.progress import progress_tracker

logger = logging.getLogger(__name__)
F = TypeVar('F', bound=Callable[..., Any])  # For function type annotations

# Create blueprint
api_bp = Blueprint('api', __name__, url_prefix='/api/v1')

# ============ Error Handling ============

class APIError(Exception):
    """Base class for API errors."""
    
    def __init__(self, message: str, status_code: int = 400, details: Optional[Dict] = None):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(message)
    
    def to_dict(self) -> Dict:
        """Convert error to dictionary for JSON response."""
        result = {
            'status': 'error',
            'error': self.message,
            'code': self.status_code,
            'timestamp': datetime.datetime.utcnow().isoformat()
        }
        
        if self.details:
            result['details'] = self.details
            
        return result


class ValidationError(APIError):
    """Error for validation failures."""
    
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, 400, details)


class ResourceNotFoundError(APIError):
    """Error for missing resources."""
    
    def __init__(self, message: str, resource_type: str, resource_id: Optional[str] = None):
        details = {'resource_type': resource_type}
        if resource_id:
            details['resource_id'] = resource_id
        super().__init__(message, 404, details)


class ProcessingError(APIError):
    """Error for processing failures."""
    
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, 500, details)


@api_bp.errorhandler(APIError)
def handle_api_error(error: APIError):
    """Handle custom API errors."""
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


@api_bp.errorhandler(HTTPException)
def handle_http_exception(error: HTTPException):
    """Handle standard HTTP exceptions."""
    response = jsonify({
        'status': 'error',
        'error': error.description,
        'code': error.code,
        'timestamp': datetime.datetime.utcnow().isoformat()
    })
    response.status_code = error.code
    return response


@api_bp.errorhandler(Exception)
def handle_generic_exception(error: Exception):
    """Handle all other exceptions."""
    logger.exception("Unhandled exception")
    
    response = jsonify({
        'status': 'error',
        'error': "Internal server error",
        'code': 500,
        'timestamp': datetime.datetime.utcnow().isoformat()
    })
    response.status_code = 500
    return response


# ============ Decorators ============

def api_route(rule: str, **options):
    """
    Decorator for API routes with standardized response format and error handling.
    
    Args:
        rule: URL rule
        **options: Options to pass to Blueprint.route
        
    Returns:
        Route decorator
    """
    def decorator(f: F) -> F:
        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            try:
                start_time = time.time()
                result = f(*args, **kwargs)
                
                # If result is a tuple, assume it's (data, status_code)
                if isinstance(result, tuple) and len(result) == 2:
                    data, status_code = result
                else:
                    data, status_code = result, 200
                
                # Standardize response
                response = {
                    'status': 'success',
                    'data': data,
                    'execution_time': f"{(time.time() - start_time):.3f}s"
                }
                
                return jsonify(response), status_code
                
            except APIError as e:
                # Let the errorhandler handle this
                raise
            except Exception as e:
                logger.exception(f"Error in API route {rule}")
                # Convert to APIError for consistent error handling
                raise ProcessingError(str(e), {
                    'traceback': traceback.format_exc() if current_app.config.get('DEBUG', False) else None
                })
        
        # Register route with the blueprint
        endpoint = options.pop('endpoint', None)
        api_bp.route(rule, endpoint=endpoint, **options)(wrapped)
        return wrapped
    
    return decorator


def validate_json_payload(*required_fields: str):
    """
    Decorator to validate JSON payload.
    
    Args:
        *required_fields: Required fields in JSON payload
        
    Returns:
        Decorator function
    """
    def decorator(f: F) -> F:
        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            if not request.is_json:
                raise ValidationError("Request must be JSON")
            
            data = request.get_json()
            
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                raise ValidationError(
                    "Missing required fields",
                    {'missing_fields': missing_fields}
                )
            
            return f(*args, **kwargs)
        
        return wrapped
    
    return decorator


def validate_files(*required_files: str, allowed_extensions: Optional[List[str]] = None):
    """
    Decorator to validate file uploads.
    
    Args:
        *required_files: Required file keys
        allowed_extensions: Allowed file extensions (e.g., ['.csv', '.xlsx'])
        
    Returns:
        Decorator function
    """
    def decorator(f: F) -> F:
        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            if not request.files:
                raise ValidationError("No files uploaded")
            
            missing_files = [file for file in required_files if file not in request.files]
            if missing_files:
                raise ValidationError(
                    "Missing required files",
                    {'missing_files': missing_files}
                )
            
            if allowed_extensions:
                invalid_files = []
                for file_key in required_files:
                    file = request.files[file_key]
                    ext = '.' + file.filename.split('.')[-1].lower() if '.' in file.filename else ''
                    if ext not in allowed_extensions:
                        invalid_files.append({
                            'file': file_key,
                            'filename': file.filename,
                            'extension': ext,
                            'allowed': allowed_extensions
                        })
                
                if invalid_files:
                    raise ValidationError(
                        "Invalid file type",
                        {'invalid_files': invalid_files}
                    )
            
            return f(*args, **kwargs)
        
        return wrapped
    
    return decorator


def sanitize_input(f: F) -> F:
    """
    Decorator to sanitize user input.
    
    Args:
        f: Function to decorate
        
    Returns:
        Decorated function
    """
    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        # Sanitize request form data
        if request.form:
            sanitized_form = {}
            for key, value in request.form.items():
                if isinstance(value, str):
                    sanitized_form[key] = bleach.clean(value)
                else:
                    sanitized_form[key] = value
            request.sanitized_form = sanitized_form
        
        # Sanitize JSON data
        if request.is_json:
            data = request.get_json()
            if isinstance(data, dict):
                sanitized_data = {}
                for key, value in data.items():
                    if isinstance(value, str):
                        sanitized_data[key] = bleach.clean(value)
                    else:
                        sanitized_data[key] = value
                request.sanitized_json = sanitized_data
        
        return f(*args, **kwargs)
    
    return wrapped


def log_execution_time(f: F) -> F:
    """
    Decorator to log function execution time.
    
    Args:
        f: Function to decorate
        
    Returns:
        Decorated function
    """
    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        execution_time = time.time() - start_time
        
        logger.info(f"Function {f.__name__} executed in {execution_time:.3f} seconds")
        
        return result
    
    return wrapped


# ============ API Routes ============

@api_route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return {
        'status': 'healthy',
        'version': '1.0',
        'timestamp': datetime.datetime.utcnow().isoformat()
    }


@api_route('/columns', methods=['POST'])
@validate_files('file', allowed_extensions=['.csv', '.xlsx'])
@log_execution_time
def get_columns():
    """Get columns from a file."""
    file = request.files['file']
    
    try:
        # Use data processor from app context
        df = current_app.data_processor.load_and_validate_file(file, 'file')
        columns = df.columns.tolist()
        
        return jsonify({"status": "success", "data": columns}), 200
            
    except Exception as e:
        logger.exception("Error processing columns")
        return jsonify({"status": "error", "error": str(e)}), 500


@api_route('/compare', methods=['POST'])
def compare_addresses():
    try:
        logger.info("Received address comparison request")
        
        # Validate files in request
        if 'file1' not in request.files or 'file2' not in request.files:
            logger.warning("Missing file(s) in request")
            return jsonify({'status': 'error', 'error': 'Missing file(s)'}), 400
            
        file1 = request.files['file1']
        file2 = request.files['file2']
        logger.info(f"Uploaded files: {file1.filename}, {file2.filename}")
        
        # Generate a unique job ID
        job_id = str(uuid.uuid4())
        progress_tracker.update_progress(
            job_id, {"status": "started", "progress": 0}
        )
        
        # Validate request data
        if not current_app.data_processor.allowed_file(file1.filename) or not current_app.data_processor.allowed_file(file2.filename):
            return jsonify({"status": "error", "error": "Invalid file type"}), 400
            
        # Get column selections
        columns1 = request.form.getlist('columns1[]')
        columns2 = request.form.getlist('columns2[]')
        
        if not columns1 or not columns2:
            return jsonify({"status": "error", "error": "Column selections are required"}), 400
            
        # Get threshold
        threshold_str = request.form.get("threshold", str(current_app.config["DEFAULT_THRESHOLD"]))
        try:
            threshold = float(threshold_str) / 100
            if not 0 <= threshold <= 1:
                raise ValueError("Threshold must be between 0 and 100")
        except ValueError:
            logger.warning(f"Invalid threshold value: {threshold_str}")
            return jsonify({'status': 'error', 'error': 'Invalid threshold value'}), 400
            
        # Load dataframes
        df1 = current_app.data_processor.load_dataframe(file1)
        df2 = current_app.data_processor.load_dataframe(file2)
        
        # Submit task to Celery if available
        if hasattr(current_app, 'celery'):
            from tasks.comparison_tasks import process_address_comparison_task
            task = process_address_comparison_task.delay(
                df1.to_dict(), df2.to_dict(), columns1, columns2, threshold, job_id
            )
            return jsonify({"status": "success", "job_id": job_id, "task_id": task.id}), 200
        else:
            # Synchronous processing (fallback if Celery not available)
            results = current_app.data_processor.compare_addresses(df1, df2, columns1, columns2, threshold)
            return jsonify({"status": "success", "data": results}), 200
            
    except Exception as e:
        logger.exception("Error processing comparison request:")
        return jsonify({"status": "error", "error": str(e)}), 500


@api_route("/validate", methods=["POST"])
def validate_address():
    """Validate a single address."""
    try:
        if not request.is_json:
            return jsonify({"status": "error", "error": "Request must be JSON"}), 400
            
        data = request.get_json()
        address = data.get("address", "")
        
        if not address:
            return jsonify({"status": "error", "error": "Address is required"}), 400
            
        # Simple response for testing
        return jsonify({
            "status": "success",
            "data": {
                "original": address,
                "normalized": address.lower().strip(),
                "suggestions": [],
                "valid": len(address) > 10
            }
        }), 200
        
    except Exception as e:
        logger.exception("Error in address validation")
        return jsonify({"status": "error", "error": str(e)}), 500


def register_api_blueprint(app):
    """Register API routes with Flask app."""
    
    # Create a blueprint
    api_bp = Blueprint('api', __name__)
    
    @api_bp.route("/")
    def index():
        """Root endpoint."""
        return jsonify({
            "status": "running",
            "version": "1.0",
            "endpoints": {
                "health": "/health",
                "columns": "/columns",
                "compare": "/compare",
                "validate": "/validate",
            },
            "timestamp": datetime.datetime.utcnow().isoformat()
        }), 200
    
    @api_bp.route("/health")
    def health_check():
        """Health check endpoint."""
        return jsonify({"status": "healthy"}), 200
    
    @api_bp.route("/columns", methods=["POST"])
    def get_columns():
        """Get columns from uploaded file."""
        try:
            if "file" not in request.files:
                return jsonify({"error": "No file provided"}), 400
                
            file = request.files["file"]
            if not file or not file.filename:
                return jsonify({"error": "Invalid file"}), 400
                
            # Simple file loading
            if file.filename.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.filename.endswith('.xlsx'):
                df = pd.read_excel(file)
            else:
                return jsonify({"error": "Unsupported file type"}), 400
                
            columns = df.columns.tolist()
            
            return jsonify({"status": "success", "data": columns}), 200
            
        except Exception as e:
            logger.exception("Error processing columns")
            return jsonify({"status": "error", "error": str(e)}), 500
    
    # Add validate and compare endpoints here too...
    
    # Register the blueprint with the app
    app.register_blueprint(api_bp)
    
    return api_bp