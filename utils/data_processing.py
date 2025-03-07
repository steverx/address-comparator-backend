from typing import List, Dict, Generator, Optional
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import gc
from utils.address_utils import AddressCorrectionModel
from utils.progress import progress_tracker
from config.excel_config import EXCEL_CONFIG, ADDRESS_KEYWORDS  # Import config
import os  # Import os
import requests
from werkzeug.datastructures import FileStorage

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self,
                 chunk_size: int = 10000,  # Use config, but keep default
                 max_workers: int = 4,
                 address_model: Optional[AddressCorrectionModel] = None):
        self.chunk_size = chunk_size  # Or:  self.chunk_size = chunk_size if chunk_size else EXCEL_CONFIG['chunk_size']
        self.max_workers = max_workers  # Number of threads
        self.address_model = address_model or AddressCorrectionModel() # Use provided model or create one
        self.allowed_extensions = {".csv", ".xlsx"}

    def allowed_file(self, filename: str) -> bool:
        """Check if file extension is allowed."""
        return os.path.splitext(filename)[1].lower() in self.allowed_extensions

    def load_dataframe(self, file_storage: FileStorage) -> pd.DataFrame:
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
        except Exception as e:
            logger.error(f"Error reading file {file_storage.filename}: {e}")
            raise

    def combine_address_components(self, row: pd.Series, columns: List[str]) -> str:
        """Combine address components into a single string."""
        components = []
        for col in columns:
            if pd.notna(row[col]):
                val = str(row[col]).strip()
                if val:
                    components.append(val)
        return ", ".join(components) if components else ""

    def process_dataframe_in_chunks(self,
                                  df: pd.DataFrame,
                                  job_id: str = None) -> Generator[pd.DataFrame, None, None]:
        """Process dataframe in chunks with progress tracking."""
        total_chunks = (len(df) + self.chunk_size - 1) // self.chunk_size

        for chunk_num, i in enumerate(range(0, len(df), self.chunk_size)):
            chunk = df.iloc[i:i + self.chunk_size].copy()  # Create a copy

            if job_id:
                progress = (chunk_num + 1) / total_chunks * 100
                progress_tracker.update_progress(job_id, {
                    'status': 'processing',
                    'progress': progress,
                    'current_chunk': chunk_num + 1,
                    'total_chunks': total_chunks
                })

            yield chunk

            # Clean up memory
            del chunk
            gc.collect()

    def preprocess_addresses(self,
                           df: pd.DataFrame,
                           address_columns: List[str]) -> pd.DataFrame:
        """Preprocess addresses for comparison."""
        # Combine, handling missing columns
        df['combined_address'] = ""  # Initialize
        valid_columns = [col for col in address_columns if col in df.columns] # Only existing columns
        if valid_columns:
            df['combined_address'] = df[valid_columns].fillna('').astype(str).agg(', '.join, axis=1)

        df['normalized_address'] = df['combined_address'].apply(self.address_model.normalize_address)
        return df


    def process_chunk(self,
                     chunk_df1: pd.DataFrame,
                     df2: pd.DataFrame,
                     columns1: List[str],
                     columns2: List[str],
                     threshold: float) -> List[Dict]:
        """Process chunks with parallel comparison."""

        # Preprocessing *inside* process_chunk (but on the entire df2 only once).
        chunk_df1 = self.preprocess_addresses(chunk_df1, columns1)
        # Only preprocess df2 *once*, outside the inner loop
        if not hasattr(df2, 'normalized_address'):
           df2 = self.preprocess_addresses(df2, columns2)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for _, row1 in chunk_df1.iterrows():
                future = executor.submit(
                    self._find_best_match,
                    row1,
                    df2,
                    threshold
                )
                futures.append(future)

            results = []
            for future in as_completed(futures):
                try:
                    match = future.result()
                    if match:
                        results.append(match)
                except Exception as e:
                    logger.error(f"Error processing match: {e}")

        return results


    def _find_best_match(self,
                        row1: pd.Series,
                        df2: pd.DataFrame,
                        threshold: float) -> Optional[Dict]:
        """Find best matching address with vectorized operations."""
        try:
            # Vectorized comparison using apply and Series.
            scores = df2['normalized_address'].apply(
                lambda x: self.address_model.compare_addresses(
                    row1['normalized_address'],
                    x
                )
            )

            # Find the index of the best score.
            best_idx = scores.idxmax()
            best_score = scores.loc[best_idx]

            if best_score > threshold:
                return {
                    'source_address': row1['combined_address'],
                    'normalized_source': row1['normalized_address'],
                    'matched_address': df2.loc[best_idx]['combined_address'],
                    'normalized_match': df2.loc[best_idx]['normalized_address'],
                    'match_score': float(best_score)
                }

            return None

        except Exception as e:
            logger.error(f"Error finding best match: {e}")
            return None

    def load_and_validate_file(self, file, file_key: str) -> pd.DataFrame:
        """Loads and validates an uploaded file."""
        ALLOWED_EXTENSIONS = {'.csv', '.xlsx'}
        if not file:
            raise ValueError(f"Missing file: {file_key}")

        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            raise ValueError(f"Invalid file type for {file_key}: {file.filename}")

        try:
            if file_ext == '.csv':
                df = pd.read_csv(file)
            elif file_ext == '.xlsx':
                df = pd.read_excel(file, engine='openpyxl')
            else:  # Should never get here
                raise ValueError(f"Unsupported file type: {file.filename}")
            return df

        except Exception as e:
            logger.exception(f"Error loading file {file.filename}:")
            raise

    def parse_address(self, address):
        """Parse address using libpostal REST API."""
        try:
            response = requests.post(
                'http://localhost:8080/parser',
                json={'query': address}
            )
            return response.json()
        except Exception as e:
            logger.error(f"Error parsing address: {e}")
            return {}

    def expand_address(self, address):
        """Expand address using libpostal REST API."""
        try:
            response = requests.post(
                'http://localhost:8080/expand',
                json={'query': address}
            )
            return response.json()
        except Exception as e:
            logger.error(f"Error expanding address: {e}")
            return []

    def normalize_address(self, address):
        """Normalize address using libpostal."""
        parsed = self.parse_address(address)
        # Format the parsed address based on your requirements
        # This is a placeholder - implement your logic here
        return address  # Return original for now

    def compare_addresses(self, df1, columns1, threshold):
        """Compare addresses between dataframe and database optimized for very large databases."""
        import psycopg2
        from psycopg2.extras import RealDictCursor
        import time
        
        conn = None
        cursor = None
        results = []
        
        try:
            logger.info(f"Comparing {len(df1)} addresses against large address database")
            start_time = time.time()
            
            # Get connection from environment variable
            conn = psycopg2.connect(os.environ["DATABASE_URL"]) 
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Preprocess and normalize addresses first
            df1 = self.preprocess_addresses(df1, columns1)
            
            # Process addresses in chunks
            for chunk_idx, chunk in enumerate(self.process_dataframe_in_chunks(df1)):
                logger.info(f"Processing chunk {chunk_idx+1} with {len(chunk)} addresses")
                chunk_results = []
                
                # STRATEGY 1: PostgreSQL similarity for all addresses in chunk (fastest)
                # Build a batch query for better performance with large datasets
                placeholders = []
                query_params = []
                for idx, row in chunk.iterrows():
                    norm_addr = row.get('normalized_address', '').strip()
                    if not norm_addr:
                        continue
                        
                    placeholders.append(f"(%s, %s, {idx})")
                    query_params.extend([norm_addr, threshold])
                    
                if not placeholders:
                    continue  # Skip if no valid addresses
                    
                # Execute batch similarity query using a temporary table for better performance
                batch_query = f"""
                    WITH address_inputs(address, threshold, row_idx) AS (
                        VALUES {", ".join(placeholders)}
                    )
                    SELECT 
                        a.id,
                        a.raw_address,
                        a.normalized_address,
                        i.row_idx as source_idx,
                        a.metadata->>'member_id' as member_id, 
                        a.metadata->>'member_name' as member_name,
                        a.metadata->'original_record'->>'LIC' as lic,
                        similarity(a.normalized_address, i.address) as score
                    FROM addresses a
                    JOIN address_inputs i ON similarity(a.normalized_address, i.address) > i.threshold
                    ORDER BY i.row_idx, score DESC;
                """
                
                cursor.execute(batch_query, query_params)
                all_matches = cursor.fetchall()
                
                # Group matches by source_idx
                matches_by_row = {}
                for match in all_matches:
                    source_idx = match['source_idx']
                    if source_idx not in matches_by_row:
                        matches_by_row[source_idx] = []
                    if len(matches_by_row[source_idx]) < 5:  # Limit to top 5 per address
                        matches_by_row[source_idx].append(match)
                
                # STRATEGY 2: For addresses with insufficient matches, use vectorized approach
                rows_needing_more = [idx for idx, matches in matches_by_row.items() 
                                   if len(matches) < 3 and idx < len(chunk)]
                
                if rows_needing_more and len(df1) > 100:
                    # Only load the vectorized matcher when needed
                    if not hasattr(self, 'address_matcher'):
                        from utils.optimized_matcher import OptimizedAddressMatcher
                        self.address_matcher = OptimizedAddressMatcher(threshold=threshold)
                    
                    # Fetch more addresses for vectorized comparison
                    cursor.execute("""
                        SELECT id, raw_address, normalized_address 
                        FROM addresses 
                        ORDER BY id  -- Use database index for efficiency
                        LIMIT 100000 -- Manageable batch size for vectorization
                    """)
                    db_addresses = cursor.fetchall()
                    
                    if db_addresses:
                        # Convert to DataFrame for vectorized operations
                        db_df = pd.DataFrame(db_addresses)
                        
                        # Create subset of source addresses needing more matches
                        source_subset = chunk.iloc[rows_needing_more].copy()
                        
                        # Get vectorized matches
                        vector_matches = self.address_matcher.batch_find_matches(
                            source_df=source_subset,
                            target_df=db_df,
                            chunk_size=len(source_subset)
                        )
                        
                        # Merge vectorized matches with database matches
                        for vm in vector_matches:
                            source_idx = vm['source_idx']
                            if source_idx in matches_by_row and len(matches_by_row[source_idx]) < 5:
                                # Get detailed match data from database
                                cursor.execute("""
                                    SELECT 
                                        id,
                                        raw_address,
                                        normalized_address,
                                        metadata->>'member_id' as member_id, 
                                        metadata->>'member_name' as member_name,
                                        metadata->'original_record'->>'LIC' as lic,
                                        %s as score
                                    FROM addresses 
                                    WHERE id = %s
                                """, (vm['score'], vm['target_id']))
                                
                                db_match = cursor.fetchone()
                                if db_match:
                                    db_match['match_type'] = 'vectorized'
                                    # Only add if not already in matches
                                    if not any(m['id'] == db_match['id'] for m in matches_by_row[source_idx]):
                                        matches_by_row[source_idx].append(dict(db_match))
                
                # Format results for return
                for idx, matches in matches_by_row.items():
                    if idx >= len(chunk):
                        continue
                        
                    row = chunk.iloc[idx]
                    address = row.get('combined_address', '')
                    
                    # Ensure score is float for JSON serialization
                    for match in matches:
                        if 'match_type' not in match:
                            match['match_type'] = 'database'
                        match['score'] = float(match['score'])
                    
                    chunk_results.append({
                        'source_address': address,
                        'source_record': row.to_dict(),
                        'match_count': len(matches),
                        'matches': sorted(matches, key=lambda x: x['score'], reverse=True)
                    })
                
                # Add chunk results to overall results
                results.extend(chunk_results)
            
            elapsed_time = time.time() - start_time
            logger.info(f"Found {len(results)} matches from {len(df1)} addresses in {elapsed_time:.2f} seconds")
            return results
            
        except Exception as e:
            logger.exception(f"Database error in compare_addresses: {e}")
            raise
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()