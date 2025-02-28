import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class OptimizedAddressMatcher:
    """
    Optimized address matching using vectorized operations for better performance.
    """
    
    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
        self.vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=(2, 3),
            min_df=1,
            max_df=0.9
        )
        
    def batch_find_matches(self, 
                         source_df: pd.DataFrame, 
                         target_df: pd.DataFrame,
                         chunk_size: int = 1000) -> List[Dict]:
        """
        Find matches between two dataframes using vectorized operations.
        
        Args:
            source_df: DataFrame with normalized_address and combined_address columns
            target_df: DataFrame with normalized_address and combined_address columns
            chunk_size: Number of source addresses to process at once
            
        Returns:
            List of match dictionaries
        """
        if 'normalized_address' not in source_df.columns or 'normalized_address' not in target_df.columns:
            raise ValueError("Both dataframes must have 'normalized_address' column")
            
        # Fit vectorizer on all target addresses (reference dataset)
        logger.info(f"Fitting vectorizer on {len(target_df)} target addresses")
        target_addresses = target_df['normalized_address'].fillna('').tolist()
        self.vectorizer.fit(target_addresses)
        
        # Transform target addresses once
        target_matrix = self.vectorizer.transform(target_addresses)
        
        results = []
        
        # Process source addresses in chunks
        for i in range(0, len(source_df), chunk_size):
            chunk = source_df.iloc[i:i+chunk_size]
            logger.info(f"Processing chunk {i//chunk_size + 1}, addresses {i} to {min(i+chunk_size, len(source_df))}")
            
            # Transform source chunk
            source_addresses = chunk['normalized_address'].fillna('').tolist()
            source_matrix = self.vectorizer.transform(source_addresses)
            
            # Calculate similarity matrix (source_chunk × target)
            # This gives a matrix of shape (len(chunk), len(target_df))
            similarity_matrix = cosine_similarity(source_matrix, target_matrix)
            
            # For each source address, find the best match
            for idx, row in enumerate(similarity_matrix):
                best_match_idx = np.argmax(row)
                best_score = row[best_match_idx]
                
                if best_score >= self.threshold:
                    match = {
                        'source_address': chunk['combined_address'].iloc[idx],
                        'normalized_source': chunk['normalized_address'].iloc[idx],
                        'matched_address': target_df['combined_address'].iloc[best_match_idx],
                        'normalized_match': target_df['normalized_address'].iloc[best_match_idx],
                        'match_score': float(best_score)
                    }
                    results.append(match)
            
        return results
        
    def find_top_k_matches(self, 
                         source_df: pd.DataFrame, 
                         target_df: pd.DataFrame,
                         k: int = 3,
                         chunk_size: int = 1000) -> Dict[int, List[Dict]]:
        """
        Find top k matches for each source address.
        
        Args:
            source_df: DataFrame with normalized_address and combined_address columns
            target_df: DataFrame with normalized_address and combined_address columns
            k: Number of top matches to return for each source address
            chunk_size: Number of source addresses to process at once
            
        Returns:
            Dictionary mapping source index to list of match dictionaries
        """
        if 'normalized_address' not in source_df.columns or 'normalized_address' not in target_df.columns:
            raise ValueError("Both dataframes must have 'normalized_address' column")
            
        # Fit vectorizer on all target addresses
        target_addresses = target_df['normalized_address'].fillna('').tolist()
        self.vectorizer.fit(target_addresses)
        
        # Transform target addresses once
        target_matrix = self.vectorizer.transform(target_addresses)
        
        results = {}
        
        # Process source addresses in chunks
        for i in range(0, len(source_df), chunk_size):
            chunk = source_df.iloc[i:i+chunk_size]
            
            # Transform source chunk
            source_addresses = chunk['normalized_address'].fillna('').tolist()
            source_matrix = self.vectorizer.transform(source_addresses)
            
            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(source_matrix, target_matrix)
            
            # For each source address, find the top k matches
            for idx, row in enumerate(similarity_matrix):
                # Get indices of top k matches
                top_k_indices = np.argsort(row)[-k:][::-1]
                
                # Filter by threshold
                matches = []
                for match_idx in top_k_indices:
                    score = row[match_idx]
                    if score >= self.threshold:
                        match = {
                            'source_address': chunk['combined_address'].iloc[idx],
                            'normalized_source': chunk['normalized_address'].iloc[idx],
                            'matched_address': target_df['combined_address'].iloc[match_idx],
                            'normalized_match': target_df['normalized_address'].iloc[match_idx],
                            'match_score': float(score)
                        }
                        matches.append(match)
                
                # Store results for this source address
                source_idx = i + idx
                if matches:
                    results[source_idx] = matches
        
        return results

# Integration with DataProcessor
def integrate_with_data_processor(self):
    """
    Replace DataProcessor._find_best_match with optimized version
    """
    # In DataProcessor class:
    def _find_best_matches_optimized(self, 
                                   chunk_df1: pd.DataFrame, 
                                   df2: pd.DataFrame, 
                                   threshold: float) -> List[Dict]:
        # Ensure dataframes are preprocessed
        if 'normalized_address' not in chunk_df1.columns:
            chunk_df1 = self.preprocess_addresses(chunk_df1, self.address_columns1)
        if 'normalized_address' not in df2.columns:
            df2 = self.preprocess_addresses(df2, self.address_columns2)
            
        # Create matcher if not exists
        if not hasattr(self, 'address_matcher'):
            self.address_matcher = OptimizedAddressMatcher(threshold=threshold)
            
        # Find matches using vectorized operations
        return self.address_matcher.batch_find_matches(
            chunk_df1, df2, chunk_size=len(chunk_df1)
        )