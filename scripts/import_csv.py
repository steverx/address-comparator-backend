import pandas as pd
import psycopg2
import psycopg2.extras
import os
import argparse
import json
from utils.address_utils import AddressCorrectionModel

def import_csv_to_database(csv_file_path):
    """Import addresses from CSV file into PostgreSQL database."""
    # Initialize address model
    address_model = AddressCorrectionModel()
    
    # Load CSV file
    print(f"Loading CSV file: {csv_file_path}")
    df = pd.read_csv(csv_file_path)
    total_rows = len(df)
    print(f"Found {total_rows} addresses to import")
    
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
    batch_size = 1000
    for i in range(0, total_rows, batch_size):
        batch = df.iloc[i:i+batch_size]
        values = []
        
        for _, row in batch.iterrows():
            # Get the raw address
            raw_address = str(row.get('address', ''))
            if not raw_address.strip():
                continue
            
            # Normalize the address
            normalized_address = address_model.normalize_address(raw_address)
            
            # Create metadata from other columns
            metadata = {}
            for col in df.columns:
                if col != 'address' and pd.notna(row[col]):
                    metadata[col] = str(row[col])
            
            # Create the record
            values.append((
                raw_address,
                normalized_address,
                json.dumps({}),  # Empty components for now
                json.dumps({"member_id": row.get('member_id', ''), 
                           "member_name": row.get('name', ''),
                           "original_record": metadata})
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
            
        print(f"Imported {min(i + batch_size, total_rows)}/{total_rows} addresses")
    
    # Commit and close
    conn.commit()
    cursor.close()
    conn.close()
    print("Import completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import CSV file to PostgreSQL database")
    parser.add_argument("csv_file", help="Path to the CSV file")
    args = parser.parse_args()
    
    import_csv_to_database(args.csv_file)