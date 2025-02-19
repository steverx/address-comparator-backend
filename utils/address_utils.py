import re
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import jellyfish
import usaddress
import logging

class AddressCorrectionModel:
    """
    Advanced machine learning-based address correction system.
    
    Key Features:
    - Spelling correction
    - Address component normalization
    - Similarity-based suggestion
    - Supervised learning for address validation
    """
    
    def __init__(self, 
                 training_data_path: Optional[str] = None, 
                 log_level: int = logging.INFO):
        """
        Initialize the address correction model.
        
        :param training_data_path: Path to CSV with verified addresses
        :param log_level: Logging level
        """
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Components for different correction strategies
        self.spelling_corrector = SpellingCorrector()
        self.similarity_matcher = AddressSimilarityMatcher()
        
        # Supervised learning model
        self.validation_model = AddressValidationModel()
        
        # Load training data if provided
        if training_data_path:
            self.load_training_data(training_data_path)
        
    def load_training_data(self, data_path: str):
        """
        Load and preprocess training data for address validation.
        
        :param data_path: Path to CSV with address training data
        """
        try:
            # Load training data
            df = pd.read_csv(data_path)
            
            # Validate required columns
            required_cols = ['address', 'is_valid']
            if not all(col in df.columns for col in required_cols):
                raise ValueError("Training data must contain 'address' and 'is_valid' columns")
            
            # Train validation model
            self.validation_model.train(df['address'], df['is_valid'])
            
            self.logger.info(f"Loaded training data from {data_path}")
        except Exception as e:
            self.logger.error(f"Error loading training data: {e}")
    
    def correct_address(self, address: str) -> Dict[str, Any]:
        """
        Comprehensive address correction pipeline.
        
        :param address: Raw input address
        :return: Corrected address details
        """
        # Spelling correction
        spelling_corrected = self.spelling_corrector.correct_address(address)
        
        # Component parsing
        try:
            parsed_components = usaddress.tag(spelling_corrected)[0]
        except Exception:
            parsed_components = {}
        
        # Similarity-based suggestions
        similar_addresses = self.similarity_matcher.find_similar_addresses(address)
        
        # Validation prediction
        is_valid, confidence = self.validation_model.predict(address)
        
        return {
            'original_address': address,
            'spelling_corrected': spelling_corrected,
            'parsed_components': parsed_components,
            'similar_addresses': similar_addresses,
            'validation': {
                'is_valid': is_valid,
                'confidence': confidence
            }
        }

class SpellingCorrector:
    """
    Advanced spelling correction for addresses using multiple strategies.
    """
    def __init__(self):
        # Common address abbreviations and corrections
        self.abbreviation_map = {
            'st': 'street', 'ave': 'avenue', 'rd': 'road', 
            'blvd': 'boulevard', 'dr': 'drive', 'apt': 'apartment',
            'n': 'north', 's': 'south', 'e': 'east', 'w': 'west'
        }
        
        # Common misspellings
        self.common_misspellings = {
            'streat': 'street', 'steet': 'street',
            'avenu': 'avenue', 'avanue': 'avenue',
            'rooad': 'road', 'raod': 'road'
        }
    
    def correct_address(self, address: str) -> str:
        """
        Correct spelling and standardize address.
        
        :param address: Raw address string
        :return: Corrected address
        """
        # Convert to lowercase
        addr = address.lower()
        
        # Replace common misspellings
        for misspell, correct in self.common_misspellings.items():
            addr = addr.replace(misspell, correct)
        
        # Expand abbreviations
        words = addr.split()
        corrected_words = [
            self.abbreviation_map.get(word, word) 
            for word in words
        ]
        
        # Apply advanced spelling correction using Levenshtein distance
        corrected_words = [
            self._find_closest_match(word) 
            for word in corrected_words
        ]
        
        return ' '.join(corrected_words)
    
    def _find_closest_match(self, word: str, threshold: float = 0.8) -> str:
        """
        Find closest match for a word using Levenshtein distance.
        
        :param word: Input word
        :param threshold: Similarity threshold
        :return: Corrected word
        """
        # Predefined dictionary of correct address-related words
        address_dictionary = set([
            'street', 'avenue', 'road', 'boulevard', 'drive', 
            'apartment', 'suite', 'north', 'south', 'east', 'west'
        ])
        
        # Find best match
        best_match = min(
            address_dictionary, 
            key=lambda x: jellyfish.levenshtein_distance(word, x)
        )
        
        # Only return if length difference is small
        max_distance = max(len(word), len(best_match)) * (1 - threshold)
        if jellyfish.levenshtein_distance(word, best_match) <= max_distance:
            return best_match
        
        return word

class AddressSimilarityMatcher:
    """
    Find similar addresses using TF-IDF and cosine similarity.
    """
    def __init__(self, max_suggestions: int = 5):
        """
        Initialize similarity matcher.
        
        :param max_suggestions: Maximum number of similar addresses to return
        """
        self.max_suggestions = max_suggestions
        self.vectorizer = TfidfVectorizer()
        self.reference_addresses = []
    
    def add_reference_addresses(self, addresses: List[str]):
        """
        Add reference addresses for similarity matching.
        
        :param addresses: List of reference addresses
        """
        self.reference_addresses.extend(addresses)
        
        # Refit vectorizer
        self._fit_vectorizer()
    
    def _fit_vectorizer(self):
        """
        Fit TF-IDF vectorizer on reference addresses.
        """
        if self.reference_addresses:
            self.vectorizer.fit(self.reference_addresses)
    
    def find_similar_addresses(self, address: str) -> List[Tuple[str, float]]:
        """
        Find similar addresses based on cosine similarity.
        
        :param address: Input address to find similar matches for
        :return: List of similar addresses with similarity scores
        """
        if not self.reference_addresses:
            return []
        
        # Vectorize input address and reference addresses
        input_vector = self.vectorizer.transform([address])
        ref_vectors = self.vectorizer.transform(self.reference_addresses)
        
        # Calculate cosine similarity
        similarities = cosine_similarity(input_vector, ref_vectors)[0]
        
        # Sort and return top suggestions
        similar_indices = similarities.argsort()[::-1][:self.max_suggestions]
        
        return [
            (self.reference_addresses[idx], similarities[idx]) 
            for idx in similar_indices 
            if similarities[idx] > 0.5  # Minimum similarity threshold
        ]

class AddressValidationModel:
    """
    Supervised learning model for address validation.
    """
    def __init__(self):
        """
        Initialize address validation model.
        """
        self.model = RandomForestClassifier(n_estimators=100)
        self.vectorizer = TfidfVectorizer()
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def _extract_features(self, addresses: List[str]) -> np.ndarray:
        """
        Extract features from addresses.
        
        :param addresses: List of address strings
        :return: Extracted feature matrix
        """
        # TF-IDF vectorization
        tfidf_features = self.vectorizer.transform(addresses)
        
        # Additional handcrafted features
        features = []
        for addr in addresses:
            addr_features = [
                len(addr),  # Address length
                len(addr.split()),  # Number of words
                bool(re.search(r'\d', addr)),  # Contains number
                bool(re.search(r'[A-Z]', addr)),  # Contains uppercase
                bool(re.search(r'\b(st|ave|rd|blvd)\b', addr))  # Contains street abbreviation
            ]
            features.append(addr_features)
        
        # Combine TF-IDF and handcrafted features
        combined_features = np.hstack([
            tfidf_features.toarray(), 
            np.array(features)
        ])
        
        return self.scaler.fit_transform(combined_features)
    
    def train(self, addresses: List[str], labels: List[bool]):
        """
        Train the address validation model.
        
        :param addresses: Training address strings
        :param labels: Corresponding validity labels
        """
        try:
            # Prepare feature matrix
            X = self._extract_features(addresses)
            y = np.array(labels)
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Evaluate model
            accuracy = self.model.score(X_test, y_test)
            print(f"Model Accuracy: {accuracy:.2f}")
            
            self.is_trained = True
        
        except Exception as e:
            print(f"Training failed: {e}")
    
    def predict(self, address: str) -> Tuple[bool, float]:
        """
        Predict address validity.
        
        :param address: Address to validate
        :return: Tuple of (is_valid, confidence)
        """
        if not self.is_trained:
            # Fallback if no training has occurred
            return self._basic_validation(address)
        
        try:
            # Extract features
            features = self._extract_features([address])
            
            # Predict
            prediction = self.model.predict(features)
            proba = self.model.predict_proba(features)[0]
            
            # Confidence is the probability of the predicted class
            confidence = proba[prediction[0]]
            
            return bool(prediction[0]), float(confidence)
        
        except Exception:
            # Fallback to basic validation
            return self._basic_validation(address)
    
    def _basic_validation(self, address: str) -> Tuple[bool, float]:
        """
        Basic address validation when no trained model is available.
        
        :param address: Address to validate
        :return: Tuple of (is_valid, confidence)
        """
        # Basic heuristics
        checks = [
            len(address) > 10,  # Minimum length
            bool(re.search(r'\d', address)),  # Contains number
            bool(re.search(r'\b[A-Z]{2}\b', address)),  # Contains state abbreviation
            bool(re.search(r'\d{5}', address))  # Contains ZIP code
        ]
        
        is_valid = all(checks)
        confidence = sum(checks) / len(checks)
        
        return is_valid, confidence

# Demonstration
def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize address correction model
    # Assume you have a training data CSV with columns: address, is_valid
    model = AddressCorrectionModel(
        training_data_path='address_training_data.csv'
    )
    
    # Add some reference addresses for similarity matching
    model.similarity_matcher.add_reference_addresses([
        '123 Main Street, Anytown, CA 12345',
        '456 Oak Avenue, Springfield, IL 62701',
        '789 Pine Road, Riverside, NY 10001'
    ])
    
    # Test addresses
    test_addresses = [
        '123 Main Streat, Anytowm, CA 12345',  # Misspelled
        '456 Oak Ave, Springfeild, IL 627001',  # Abbreviated
        'Invalid Address'  # Completely wrong
    ]
    
    for address in test_addresses:
        print("\nInput Address:", address)
        
        # Correct and validate address
        correction = model.correct_address(address)
        
        print("Correction Results:")
        print(json.dumps(correction, indent=2))

if __name__ == "__main__":
    main()