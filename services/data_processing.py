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

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self,
                 chunk_size: int = 10000,  # Use config, but keep default
                 max_workers: int = 4,
                 address_model: Optional[AddressCorrectionModel] = None):
        self.chunk_size = chunk_size  # Or:  self.chunk_size = chunk_size if chunk_size else EXCEL_CONFIG['chunk_size']
        self.max_workers = max_workers  # Number of threads
        self.address_model = address_model or AddressCorrectionModel() # Use provided model or create one

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