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
import os

logger = logging.getLogger(__name__)
F = TypeVar('F', bound=Callable[..., Any])  # For function type annotations

# Create blueprint
api_bp = Blueprint('api', __name__, url_prefix='/api/v1')  # ADD THIS LINE

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
    """Compare addresses from uploaded file to database."""
    try:
        # Check if file exists in request
        if "file" not in request.files:
            return jsonify({"status": "error", "error": "No file provided"}), 400
            
        file = request.files["file"]
        if not file or not file.filename:
            return jsonify({"status": "error", "error": "Invalid file"}), 400
        
        # Get columns from form data
        columns = request.form.getlist("columns[]") or []
        if not columns:
            columns = json.loads(request.form.get("columns", "[]"))
        
        # Get threshold (default 80%)
        threshold = float(request.form.get("threshold", "0.8"))
        
        # Load dataframe using data processor
        df = current_app.data_processor.load_dataframe(file)
        
        # Process comparison against database
        results = current_app.data_processor.compare_addresses(df, columns, threshold)
        
        return {"data": results}, 200
        
    except Exception as e:
        logger.exception("Error processing address comparison")
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
    # Configure CORS
    from flask_cors import CORS
    CORS(app, resources={r"/api/*": {
        "origins": os.environ.get('ALLOWED_ORIGINS', '*').split(','),
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }})
    
    # Register the blueprint with the app
    app.register_blueprint(api_bp)
    
    return api_bp