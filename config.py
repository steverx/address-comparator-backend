class Config:
    """Configuration settings for the Flask application."""
    ALLOWED_EXTENSIONS = {'.csv', '.xlsx'}
    DEFAULT_PARSER = 'usaddress'
    DEFAULT_THRESHOLD = 80
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
    CACHE_MAXSIZE = 100
    CACHE_TTL = 3600  # 1 hour in seconds
    DEBUG = True  # Enable Flask's debug mode
    LOG_LEVEL = "INFO"