# scripts/direct_import.py
import pandas as pd
import psycopg2
import psycopg2.extras
import sys
import json
import re

def normalize_address(address):
    """Simple address normalization."""
    if not address:
        return ""
    address = str(address).lower().strip()
    address = re.sub(r'[^\w\s,]', '', address)
    address = re.sub(r'\s+', ' ', address)
    return address

# Use your actual PostgreSQL connection URL here
DATABASE_URL = "postgresql://postgres:DgFDrKcpwiqvSNGKxuCnvfqaXRiyxfws@monorail.proxy.rlwy.net:47823/railway"

def import_csv(csv_path):
    print(f"Loading CSV file: {csv_path}")
    
    # Read with low_memory=False to avoid dtype warnings
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"Successfully loaded {len(df)} rows")
    
    # Find address column
    address_col = None
    for col in df.columns:
        if 'address' in col.lower():
            address_col = col
            print(f"Found address column: {address_col}")
            break
    
    if not address_col:
        print("No address column found. Using first column.")
        address_col = df.columns[0]
    
    # Connect to database
    print("Connecting to database...")
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    
    # Create table
    print("Creating table if it doesn't exist...")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS addresses (
            id SERIAL PRIMARY KEY,
            raw_address TEXT NOT NULL,
            normalized_address TEXT NOT NULL,
            components JSONB DEFAULT '{}',
            metadata JSONB DEFAULT '{}'
        );
        
        CREATE INDEX IF NOT EXISTS idx_normalized_address 
        ON addresses(normalized_address);
    """)
    
    # Import in batches
    batch_size = 1000
    total_imported = 0
    
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        values = []
        
        for _, row in batch.iterrows():
            raw_address = str(row.get(address_col, '')).strip()
            if not raw_address:
                continue
                
            # Create metadata
            metadata = {}
            for col in df.columns:
                if pd.notna(row[col]) and col != address_col:
                    metadata[col] = str(row[col])
            
            values.append((
                raw_address,
                normalize_address(raw_address),
                json.dumps({}),
                json.dumps(metadata)
            ))
        
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
            print(f"Imported {total_imported}/{len(df)} rows")
    
    conn.commit()
    cursor.close()
    conn.close()
    print("Import completed successfully!")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python direct_import.py <csv_file>")
        sys.exit(1)
        
    import_csv(sys.argv[1])