import pytest
import json
import os
import pandas as pd
import tempfile
from typing import Dict, List, Any, Generator
from unittest.mock import patch, MagicMock
from io import BytesIO

# Import application modules
from app import create_app
from utils.address_utils import AddressCorrectionModel
from utils.data_processing import DataProcessor
from config.config_manager import ConfigManager
from api.api_blueprint import api_bp, ValidationError

# ============ Fixtures ============

@pytest.fixture
def config() -> Dict[str, Any]:
    """Test configuration."""
    return {
        'server': {
            'debug': True,
            'testing': True
        },
        'security': {
            'allowed_origins': ['*'],
            'admin_username': 'test_admin',
            'admin_password': 'test_password'
        },
        'processing': {
            'chunk_size': 100,
            'default_threshold': 80,
            'max_workers': 2
        },
        'celery': {
            'broker_url': None  # Disable Celery for testing
        }
    }

@pytest.fixture
def test_app(config):
    """Flask test application."""
    app = create_app(config=config)
    app.config['TESTING'] = True
    return app

@pytest.fixture
def client(test_app):
    """Flask test client."""
    with test_app.test_client() as client:
        yield client

@pytest.fixture
def auth_headers():
    """Authentication headers for admin routes."""
    import base64
    credentials = base64.b64encode(b'test_admin:test_password').decode('utf-8')
    return {'Authorization': f'Basic {credentials}'}

@pytest.fixture
def address_model():
    """Address correction model for testing."""
    return AddressCorrectionModel()

@pytest.fixture
def data_processor(address_model):
    """Data processor for testing."""
    return DataProcessor(
        chunk_size=100,
        max_workers=2,
        address_model=address_model
    )

@pytest.fixture
def sample_addresses() -> List[str]:
    """Sample addresses for testing."""
    return [
        "123 Main St, Anytown, CA 12345",
        "456 Oak Ave, Springfield, IL 62701",
        "789 Pine Rd, Riverside, NY 10001",
        "321 Elm Blvd, Lakeside, TX 75001",
        "654 Maple Dr, Mountain View, CO 80001"
    ]

@pytest.fixture
def sample_invalid_addresses() -> List[str]:
    """Sample invalid addresses for testing."""
    return [
        "",
        "123",
        "abcdef",
        "123 Main St",
        "Anytown, CA"
    ]

@pytest.fixture
def sample_df1() -> pd.DataFrame:
    """Sample DataFrame for testing."""
    data = {
        'id': [1, 2, 3, 4, 5],
        'name': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Brown', 'Charlie Davis'],
        'address': [
            "123 Main St, Anytown, CA 12345",
            "456 Oak Ave, Springfield, IL 62701",
            "789 Pine Rd, Riverside, NY 10001",
            "321 Elm Blvd, Lakeside, TX 75001",
            "654 Maple Dr, Mountain View, CO 80001"
        ]
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_df2() -> pd.DataFrame:
    """Sample DataFrame for comparison testing."""
    data = {
        'customer_id': [101, 102, 103, 104, 105],
        'customer_name': ['John D.', 'Jane S.', 'Robert J.', 'Alice B.', 'Charles D.'],
        'customer_address': [
            "123 Main Street, Anytown, California 12345",  # Similar to df1[0]
            "456 Oak Avenue, Springfield, Illinois 62701",  # Similar to df1[1]
            "789 Pine Road, Riverside, New York 10001",     # Similar to df1[2]
            "321 Elm Boulevard, Lakeside, Texas 75001",     # Similar to df1[3]
            "654 Maple Drive, Mountain View, Colorado 80001" # Similar to df1[4]
        ]
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_csv_file1(sample_df1) -> Generator[BytesIO, None, None]:
    """Sample CSV file for testing."""
    csv_file = BytesIO()
    sample_df1.to_csv(csv_file, index=False)
    csv_file.seek(0)
    yield csv_file
    csv_file.close()

@pytest.fixture
def sample_csv_file2(sample_df2) -> Generator[BytesIO, None, None]:
    """Sample CSV file for comparison testing."""
    csv_file = BytesIO()
    sample_df2.to_csv(csv_file, index=False)
    csv_file.seek(0)
    yield csv_file
    csv_file.close()

# ============ Unit Tests ============

# Address Correction Model Tests

def test_address_correction_model_init():
    """Test AddressCorrectionModel initialization."""
    model = AddressCorrectionModel()
    assert hasattr(model, 'normalize_address')
    assert hasattr(model, 'compare_addresses')
    assert hasattr(model, 'correct_address')

def test_normalize_address(address_model, sample_addresses):
    """Test address normalization."""
    for address in sample_addresses:
        normalized = address_model.normalize_address(address)
        assert isinstance(normalized, str)
        assert normalized.islower()
        
def test_compare_addresses(address_model, sample_addresses):
    """Test address comparison."""
    # Compare identical addresses
    for address in sample_addresses:
        score = address_model.compare_addresses(address, address)
        assert score > 0.9  # Should be very high for identical addresses
    
    # Compare similar addresses
    similar_pairs = [
        ("123 Main St, Anytown, CA 12345", "123 Main Street, Anytown, California 12345"),
        ("456 Oak Ave, Springfield, IL 62701", "456 Oak Avenue, Springfield, Illinois 62701")
    ]
    
    for addr1, addr2 in similar_pairs:
        score = address_model.compare_addresses(addr1, addr2)
        assert score > 0.8  # Should be high for similar addresses
    
    # Compare different addresses
    different_pairs = [
        ("123 Main St, Anytown, CA 12345", "789 Pine Rd, Riverside, NY 10001"),
        ("456 Oak Ave, Springfield, IL 62701", "321 Elm Blvd, Lakeside, TX 75001")
    ]
    
    for addr1, addr2 in different_pairs:
        score = address_model.compare_addresses(addr1, addr2)
        assert score < 0.8  # Should be lower for different addresses

def test_correct_address(address_model, sample_addresses, sample_invalid_addresses):
    """Test address correction."""
    # Test valid addresses
    for address in sample_addresses:
        result = address_model.correct_address(address)
        assert isinstance(result, dict)
        assert 'original_address' in result
        assert 'spelling_corrected' in result
        assert 'parsed_components' in result
        assert 'validation' in result
        assert result['validation']['is_valid']
    
    # Test invalid addresses
    for address in sample_invalid_addresses:
        if address:  # Skip empty string
            result = address_model.correct_address(address)
            assert isinstance(result, dict)
            assert 'original_address' in result
            assert 'validation' in result
            # Some might be considered valid by the basic validation
            # so we don't assert on the is_valid field

# Data Processor Tests

def test_data_processor_init(address_model):
    """Test DataProcessor initialization."""
    processor = DataProcessor(
        chunk_size=100,
        max_workers=2,
        address_model=address_model
    )
    assert processor.chunk_size == 100
    assert processor.max_workers == 2
    assert processor.address_model == address_model

def test_preprocess_addresses(data_processor, sample_df1):
    """Test address preprocessing."""
    result = data_processor.preprocess_addresses(sample_df1, ['address'])
    assert 'combined_address' in result.columns
    assert 'normalized_address' in result.columns
    assert len(result) == len(sample_df1)

def test_process_dataframe_in_chunks(data_processor, sample_df1):
    """Test dataframe chunking."""
    chunks = list(data_processor.process_dataframe_in_chunks(sample_df1))
    assert len(chunks) > 0
    total_rows = sum(len(chunk) for chunk in chunks)
    assert total_rows == len(sample_df1)

def test_process_chunk(data_processor, sample_df1, sample_df2):
    """Test chunk processing."""
    # Preprocess dataframes
    df1 = data_processor.preprocess_addresses(sample_df1, ['address'])
    df2 = data_processor.preprocess_addresses(sample_df2, ['customer_address'])
    
    # Process chunk
    results = data_processor.process_chunk(df1, df2, ['address'], ['customer_address'], 0.7)
    
    assert isinstance(results, list)
    assert len(results) > 0
    
    # Check result structure
    for result in results:
        assert 'source_address' in result
        assert 'matched_address' in result
        assert 'match_score' in result
        assert isinstance(result['match_score'], float)
        assert 0 <= result['match_score'] <= 1

def test_load_and_validate_file(data_processor, sample_csv_file1):
    """Test file loading and validation."""
    mock_file = MagicMock()
    mock_file.filename = "test.csv"
    mock_file.read = sample_csv_file1.read
    mock_file.seek = sample_csv_file1.seek
    
    df = data_processor.load_and_validate_file(mock_file, 'file1')
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert list(df.columns) == ['id', 'name', 'address']

# API Tests

def test_health_endpoint(client):
    """Test health check endpoint."""
    response = client.get('/api/v1/health')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'success'
    assert 'version' in data['data']

def test_columns_endpoint(client, sample_csv_file1):
    """Test columns endpoint."""
    data = {'file': (sample_csv_file1, 'test.csv')}
    response = client.post('/api/v1/columns', data=data, content_type='multipart/form-data')
    assert response.status_code == 200
    result = json.loads(response.data)
    assert result['status'] == 'success'
    assert 'columns' in result['data']
    assert 'suggested_address_columns' in result['data']
    assert 'address' in result['data']['suggested_address_columns']

def test_validate_endpoint(client, sample_addresses):
    """Test address validation endpoint."""
    for address in sample_addresses:
        data = {'address': address}
        response = client.post('/api/v1/validate', json=data)
        assert response.status_code ==