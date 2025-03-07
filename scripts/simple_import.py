# scripts/simple_import.py
import pandas as pd
import psycopg2
import psycopg2.extras
import os
import sys
import json
import re

def normalize_address(address):
    """Basic address normalization without dependencies."""
    if not address:
        return ""
    
    # Convert to lowercase
    address = address.lower()
    
    # Replace common abbreviations
    replacements = {
        'avenue': 'ave',
        'boulevard': 'blvd',
        'circle': 'cir',
        'court': 'ct',
        'drive': 'dr',
        'lane': 'ln',
        'place': 'pl',
        'road': 'rd',
        'square': 'sq',
        'street': 'st',
        'terrace': 'ter'
    }
    
    for full, abbr in replacements.items():
        address = re.sub(r'\b' + full + r'\b', abbr, address)
        
    # Remove punctuation except commas for readability
    address = re.sub(r'[^\w\s,]', '', address)
    
    # Normalize whitespace
    address = re.sub(r'\s+', ' ', address).strip()
    
    return address

def import_csv_to_database(csv_file_path):
    """Import addresses from CSV file into PostgreSQL database."""
    # Load CSV file
    print(f"Loading CSV file: {csv_file_path}")
    df = pd.read_csv(csv_file_path)
    total_rows = len(df)
    print(f"Found {total_rows} rows to import")
    
    # Guess address column
    address_col = None
    for col in df.columns:
        if 'address' in col.lower():
            address_col = col
            break
    
    if not address_col and len(df.columns) > 0:
        address_col = df.columns[0]
        print(f"No column with 'address' found. Using first column: {address_col}")
        
    if not address_col:
        print("Error: Could not identify address column")
        return
        
    # Connect to database
    print("Connecting to database...")
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    cursor = conn.cursor()
    
    # Create table if it doesn't exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS addresses (
            id SERIAL PRIMARY KEY,
            raw_address TEXT NOT NULL,
            normalized_address TEXT NOT NULL,
            components JSONB DEFAULT '{}',
            metadata JSONB DEFAULT '{}',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_normalized_address 
        ON addresses(normalized_address);
    """)
    
    # Process and insert data
    batch_size = 100
    total_imported = 0
    
    for i in range(0, total_rows, batch_size):
        batch = df.iloc[i:i+batch_size]
        values = []
        
        for _, row in batch.iterrows():
            # Get the raw address
            raw_address = str(row.get(address_col, ''))
            if not raw_address.strip():
                continue
            
            # Basic normalization
            normalized_address = normalize_address(raw_address)
            
            # Create metadata from other columns
            metadata = {}
            for col in df.columns:
                if col != address_col and pd.notna(row[col]):
                    metadata[col] = str(row[col])
            
            # Create the record
            values.append((
                raw_address,
                normalized_address,
                json.dumps({}),  # Empty components for now
                json.dumps({
                    "member_id": str(row.get('member_id', row.get('id', ''))), 
                    "member_name": str(row.get('name', row.get('member_name', ''))),
                    "original_record": metadata
                })
            ))
        
        # Batch insert
        if values:
            psycopg2.extras.execute_values(
                cursor,
                """
                INSERT INTO addresses (raw_address, normalized_address, components, metadata)
                VALUES %s
                """,
                values
            )
            total_imported += len(values)
            
        print(f"Imported {min(i + batch_size, total_rows)}/{total_rows} rows")
    
    # Commit and close
    conn.commit()
    cursor.close()
    conn.close()
    print(f"Import completed successfully! {total_imported} addresses imported.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python simple_import.py <csv_file>")
        sys.exit(1)
    
    import_csv_to_database(sys.argv[1])