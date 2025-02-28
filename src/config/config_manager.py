import os
import json
import logging
import yaml
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class ConfigManager:
    """
    Unified configuration management system for address comparison application.
    Centralizes configuration from environment variables, files, and defaults.
    """
    
    # Default configuration values
    DEFAULT_CONFIG = {
        # Server configuration
        'server': {
            'debug': False,
            'port': 5000,
            'host': '0.0.0.0',
            'workers': 2,
            'timeout': 600,  # 10 minutes
            'max_content_length': 500 * 1024 * 1024,  # 500MB
        },
        
        # Security configuration
        'security': {
            'allowed_origins': ['https://address-comparator-frontend-production.up.railway.app'],
            'cors_methods': ['GET', 'POST', 'OPTIONS'],
            'cors_headers': ['Content-Type', 'Authorization'],
            'supports_credentials': True,
            'admin_username': 'admin',
            'admin_password': None,  # Must be set via env var
        },
        
        # Data processing configuration
        'processing': {
            'chunk_size': 10000,
            'default_threshold': 80,  # percentage
            'max_workers': 4,
            'allowed_extensions': ['.csv', '.xlsx'],
        },
        
        # Address model configuration
        'address_model': {
            'model_type': 'random_forest',
            'training_data_path': None,  # Optional path to training data
            'use_libpostal': True,
            'enable_ml': True,
            'similarity_method': 'combined',
        },
        
        # Celery configuration
        'celery': {
            'broker_url': 'redis://localhost:6379/0',
            'result_backend': 'redis://localhost:6379/0',
            'task_serializer': 'json',
            'result_serializer': 'json',
            'accept_content': ['json'],
            'timezone': 'UTC',
            'enable_utc': True,
            'worker_concurrency': 2,
        },
        
        # Logging configuration
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file': None,  # If None, log to stdout
            'force': True,
        },
        
        # Address keywords for auto-detection
        'address_keywords': [
            'address', 'addr', 'street', 'avenue', 'road', 'boulevard', 'lane',
            'city', 'town', 'state', 'province', 'zip', 'postal', 'country'
        ],
    }
    
    def __init__(self, 
                env_prefix: str = 'ADDR_CMP_', 
                config_file: Optional[str] = None):
        """
        Initialize ConfigManager.
        
        Args:
            env_prefix: Prefix for environment variables
            config_file: Optional path to config file (YAML or JSON)
        """
        self.env_prefix = env_prefix
        self.config = self.DEFAULT_CONFIG.copy()
        
        # Load configuration in order (later sources override earlier ones):
        # 1. Default values (already loaded)
        # 2. Configuration file (if provided)
        # 3. Environment variables
        
        # Load from config file if provided
        if config_file:
            self._load_from_file(config_file)
        
        # Load environment variables
        self._load_from_env()
        
        # Validate critical settings
        self._validate_config()
        
        logger.info("Configuration loaded successfully")
        
    def _load_from_file(self, config_file: str) -> None:
        """Load configuration from file."""
        try:
            if not os.path.exists(config_file):
                logger.warning(f"Config file not found: {config_file}")
                return
                
            logger.info(f"Loading configuration from {config_file}")
            file_ext = os.path.splitext(config_file)[1].lower()
            
            if file_ext == '.json':
                with open(config_file, 'r') as f:
                    file_config = json.load(f)
            elif file_ext in ('.yaml', '.yml'):
                with open(config_file, 'r') as f:
                    file_config = yaml.safe_load(f)
            else:
                logger.warning(f"Unsupported config file format: {file_ext}")
                return
                
            # Update config with file values
            self._update_nested_dict(self.config, file_config)
            logger.info(f"Loaded configuration from {config_file}")
            
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
    
    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        # Load variables from .env file if it exists
        load_dotenv()
        
        # Process all environment variables with the prefix
        for key, value in os.environ.items():
            if key.startswith(self.env_prefix):
                # Remove prefix and convert to lowercase
                config_key = key[len(self.env_prefix):].lower()
                
                # Split by double underscore to identify nested keys
                path = config_key.split('__')
                
                # Convert value to appropriate type
                typed_value = self._convert_value_type(value)
                
                # Update configuration
                self._set_nested_value(self.config, path, typed_value)
    
    def _convert_value_type(self, value: str) -> Any:
        """Convert string value to appropriate type."""
        # Boolean values
        if value.lower() in ('true', 'yes', '1'):
            return True
        if value.lower() in ('false', 'no', '0'):
            return False
            
        # Numbers
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            pass
            
        # Lists (comma-separated)
        if ',' in value:
            return [self._convert_value_type(item.strip()) for item in value.split(',')]
            
        # Default to string
        return value
    
    def _set_nested_value(self, d: Dict, path: List[str], value: Any) -> None:
        """Set a value in a nested dictionary based on path."""
        current = d
        for part in path[:-1]:
            if part not in current:
                current[part] = {}
            if not isinstance(current[part], dict):
                # If the path conflicts with an existing non-dict value, log and return
                logger.warning(f"Cannot set nested key {'.'.join(path)} because {part} is not a dictionary")
                return
            current = current[part]
        current[path[-1]] = value
    
    def _update_nested_dict(self, target: Dict, source: Dict) -> None:
        """Update a nested dictionary with values from another nested dictionary."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._update_nested_dict(target[key], value)
            else:
                target[key] = value
    
    def _validate_config(self) -> None:
        """Validate critical configuration values."""
        # Check security settings
        if not self.config['security']['allowed_origins']:
            logger.warning("No allowed origins specified, this may cause CORS issues")
            
        # Check admin password
        if self.config['security']['admin_username'] and not self.config['security']['admin_password']:
            logger.warning("Admin username specified but no password set")
        
        # Check celery configuration
        if not self.config['celery']['broker_url']:
            logger.warning("Celery broker URL not specified, background tasks will not work")
        
        # Check processing settings
        if self.config['processing']['chunk_size'] <= 0:
            logger.warning(f"Invalid chunk size: {self.config['processing']['chunk_size']}, using default")
            self.config['processing']['chunk_size'] = self.DEFAULT_CONFIG['processing']['chunk_size']
    
    def get_flask_config(self) -> Dict:
        """Get Flask-specific configuration."""
        return {
            'DEBUG': self.config['server']['debug'],
            'MAX_CONTENT_LENGTH': self.config['server']['max_content_length'],
            'ENV': 'development' if self.config['server']['debug'] else 'production',
        }
    
    def get_celery_config(self) -> Dict:
        """Get Celery-specific configuration."""
        return self.config['celery']
    
    def get_cors_config(self) -> Dict:
        """Get CORS configuration."""
        return {
            'origins': self.config['security']['allowed_origins'],
            'methods': self.config['security']['cors_methods'],
            'allow_headers': self.config['security']['cors_headers'],
            'supports_credentials': self.config['security']['supports_credentials'],
        }
    
    def get_logging_config(self) -> Dict:
        """Get logging configuration."""
        return self.config['logging']
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by dot-notation key.
        
        Args:
            key: Dot-notation key (e.g., 'server.port')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        parts = key.split('.')
        value = self.config
        
        for part in parts:
            if not isinstance(value, dict) or part not in value:
                return default
            value = value[part]
            
        return value
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access to configuration."""
        result = self.get(key)
        if result is None:
            raise KeyError(f"Configuration key not found: {key}")
        return result


# Usage in application factory:
def create_app(config_file: Optional[str] = None):
    """
    Application factory function with unified configuration.
    
    Args:
        config_file: Optional path to config file
        
    Returns:
        Flask application
    """
    from flask import Flask
    from flask_cors import CORS
    
    # Initialize configuration
    config_manager = ConfigManager(config_file=config_file)
    
    # Create Flask app
    app = Flask(__name__)
    
    # Apply Flask configuration
    app.config.update(config_manager.get_flask_config())
    
    # Setup CORS
    CORS(app, **config_manager.get_cors_config())
    
    # Configure logging
    logging_config = config_manager.get_logging_config()
    logging.basicConfig(
        level=getattr(logging, logging_config['level']),
        format=logging_config['format'],
        filename=logging_config['file'],
        force=logging_config['force']
    )
    
    # Initialize services with configuration
    init_services(app, config_manager)
    
    # Register routes
    register_routes(app, config_manager)
    
    return app


def init_services(app, config_manager):
    """Initialize application services with configuration."""
    from utils.address_utils import AddressCorrectionModel
    from utils.data_processing import DataProcessor
    
    # Initialize address correction model
    app.address_model = AddressCorrectionModel(
        training_data_path=config_manager.get('address_model.training_data_path'),
        model_type=config_manager.get('address_model.model_type')
    )
    
    # Initialize data processor
    app.data_processor = DataProcessor(
        chunk_size=config_manager.get('processing.chunk_size'),
        max_workers=config_manager.get('processing.max_workers'),
        address_model=app.address_model
    )
    
    # Initialize Celery if needed
    if config_manager.get('celery.broker_url'):
        from celery import Celery
        
        app.celery = Celery(app.name)
        app.celery.conf.update(config_manager.get_celery_config())


def register_routes(app, config_manager):
    """Register application routes with configuration."""
    from flask import jsonify, request
    from flask_httpauth import HTTPBasicAuth
    from werkzeug.security import generate_password_hash, check_password_hash
    
    # Setup authentication if needed
    if config_manager.get('security.admin_username') and config_manager.get('security.admin_password'):
        auth = HTTPBasicAuth()
        
        users = {
            config_manager.get('security.admin_username'): generate_password_hash(
                config_manager.get('security.admin_password')
            )
        }
        
        @auth.verify_password
        def verify_password(username, password):
            if username in users and check_password_hash(users.get(username), password):
                return username
            return None
        
        @app.route('/admin')
        @auth.login_required
        def admin_route():
            return "Admin area - requires authentication"
    
    # Register API routes (simplified example)
    @app.route('/health')
    def health_check():
        return jsonify({'status': 'healthy'}), 200
    
    # Additional route registration here...