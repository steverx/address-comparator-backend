import os
import sys

# Print current working directory and Python path for debugging
print("Current Working Directory:", os.getcwd())
print("Python Path:", sys.path)

# Get the absolute path of the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the project root to PYTHONPATH
sys.path.insert(0, project_root)

# Print updated Python path
print("Updated Python Path:", sys.path)

# Import the AddressCorrectionModel
from utils.address_utils import AddressCorrectionModel

import pandas as pd

def create_verified_addresses():
    # Initialize the address correction model
    model = AddressCorrectionModel()

    # Sample addresses
    addresses = [
        '123 Main Street, Anytown, CA 12345',
        '456 Oak Avenue, Springfield, IL 62701',
        '789 Pine Road, Rivertown, NY 10001',
        '1010 Corporate Drive, Industryville, TX 75001',
        '250 Maple Lane, Suburbia, WA 98001',
        '742 Evergreen Terrace, Springfield, OR 97001'
    ]

    # Clean and validate addresses
    verified_addresses = []
    for addr in addresses:
        # Use the correct_address method
        correction = model.correct_address(addr)
        
        # Check if the address is valid
        if correction['validation']['is_valid'] and correction['validation']['confidence'] > 0.7:
            verified_addresses.append({
                'address': correction['spelling_corrected'],
                'type': 'residential' if 'lane' in correction['spelling_corrected'].lower() or 'terrace' in correction['spelling_corrected'].lower() else 'commercial',
                'verified_date': '2024-02-11',
                'confidence': correction['validation']['confidence']
            })

    # Create DataFrame
    df = pd.DataFrame(verified_addresses)

    # Ensure data directory exists
    os.makedirs(os.path.join(project_root, 'Data'), exist_ok=True)

    # Save to CSV
    file_path = os.path.join(project_root, 'Data', 'verified_addresses.csv')
    df.to_csv(file_path, index=False)
    print(f"Verified addresses saved to {file_path}")
    print(df)

# Ensure the script runs only when directly executed
if __name__ == '__main__':
    create_verified_addresses()