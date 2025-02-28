import re
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, precision_recall_fscore_support
import usaddress
from fuzzywuzzy import fuzz
import jellyfish

try:
    import pycountry
except ImportError:
    pycountry = None

logger = logging.getLogger(__name__)

class EnhancedAddressValidator:
    """
    Advanced address validation system using multiple ML techniques and feature engineering.
    """
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize enhanced address validator.
        
        Args:
            model_type: Type of model to use ('random_forest' or 'gradient_boosting')
        """
        self.logger = logging.getLogger(__name__)
        self.is_trained = False
        self.model_type = model_type
        
        # Initialize feature extraction components
        self.text_vectorizer = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(2, 5),
            max_features=500
        )
        
        # Initialize country code database
        self.countries = {}
        if pycountry:
            self.countries = {country.alpha_2: country.name for country in pycountry.countries}
        
        # Define regex patterns for address components
        self.patterns = {
            'zip_code_us': r'\b\d{5}(-\d{4})?\b',
            'zip_code_ca': r'\b[A-Za-z]\d[A-Za-z]\s?\d[A-Za-z]\d\b',
            'zip_code_uk': r'\b[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}\b',
            'phone': r'\b(\+\d{1,3}[\s-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'po_box': r'\b[P|p]\.?[O|o]\.?\s*[B|b][O|o][X|x]?\s+\d+\b',
            'building_identifiers': r'\b(suite|apt|apartment|unit|room|floor|ste|fl|#)\s*[#]?\s*\w+\b',
            'street_number': r'\b\d+\s+[A-Za-z]',
            'street_suffix': r'\b(street|st|avenue|ave|boulevard|blvd|drive|dr|lane|ln|road|rd|court|ct|plaza|plz|parkway|pkwy)\b',
            'compass_dir': r'\b(north|south|east|west|n|s|e|w|ne|nw|se|sw)\b',
            'state_abbr': r'\b[A-Z]{2}\b'
        }

        # Common address abbreviations dictionary
        self.common_abbreviations = {
            'st': 'street',
            'rd': 'road',
            'dr': 'drive',
            'ave': 'avenue',
            'blvd': 'boulevard',
            'ln': 'lane',
            'ct': 'court',
            'pl': 'place',
            'ter': 'terrace',
            'apt': 'apartment'
        }
        
        # Initialize model
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the ML model based on selected type."""
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=10,
                random_state=42,
                class_weight='balanced'
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def normalize_address(self, address: Optional[str]) -> str:
        """Normalize address string by removing special characters and standardizing format."""
        if not address or not isinstance(address, str):
            return ""

        try:
            # Convert to lowercase
            address = address.lower().strip()

            # Remove special characters and extra whitespace
            address = re.sub(r'[^\w\s]', ' ', address)
            address = re.sub(r'\s+', ' ', address)

            # Expand common abbreviations
            words = address.split()
            normalized_words = [
                self.common_abbreviations.get(word, word)
                for word in words
            ]

            return ' '.join(normalized_words)

        except Exception as e:
            self.logger.error(f"Error normalizing address: {e}")
            return address
    
    def extract_features(self, addresses: List[str]) -> pd.DataFrame:
        """
        Extract comprehensive features from addresses.
        
        Args:
            addresses: List of address strings
            
        Returns:
            DataFrame with extracted features
        """
        features = []
        
        for address in addresses:
            if not address or not isinstance(address, str):
                # Handle empty or invalid addresses
                features.append(self._get_empty_features())
                continue
                
            try:
                # Basic text features
                text_features = {
                    'length': len(address),
                    'word_count': len(address.split()),
                    'avg_word_length': np.mean([len(w) for w in address.split()]) if address.split() else 0,
                    'digit_ratio': sum(c.isdigit() for c in address) / len(address) if address else 0,
                    'uppercase_ratio': sum(c.isupper() for c in address) / len(address) if address else 0,
                    'alpha_ratio': sum(c.isalpha() for c in address) / len(address) if address else 0,
                    'space_ratio': sum(c.isspace() for c in address) / len(address) if address else 0
                }
                
                # Regular expression-based features
                regex_features = {
                    f'has_{name}': bool(re.search(pattern, address, re.IGNORECASE))
                    for name, pattern in self.patterns.items()
                }
                
                # Parse using usaddress
                try:
                    parsed_usaddress, address_type = usaddress.tag(address)
                    usaddress_features = {
                        'usaddress_type': address_type,
                        'usaddress_components': len(parsed_usaddress),
                        'has_number': 'AddressNumber' in parsed_usaddress,
                        'has_street': 'StreetName' in parsed_usaddress,
                        'has_state': 'StateName' in parsed_usaddress or 'StateNameAbbreviation' in parsed_usaddress,
                        'has_zip': 'ZipCode' in parsed_usaddress
                    }
                except (usaddress.RepeatedLabelError, usaddress.BadLabelError):
                    usaddress_features = {
                        'usaddress_type': 'parsing_error',
                        'usaddress_components': 0,
                        'has_number': False,
                        'has_street': False,
                        'has_state': False,
                        'has_zip': False
                    }
                
                # Instead of using libpostal, add advanced text analysis features
                norm_address = self.normalize_address(address)
                advanced_features = {
                    'has_valid_format': bool(re.search(r'\d+\s+[\w\s]+,\s+[\w\s]+,\s+[A-Z]{2}\s+\d{5}', address, re.IGNORECASE)),
                    'normalized_length': len(norm_address),
                    'contains_directions': bool(re.search(r'\b(north|south|east|west|n|s|e|w)\b', address, re.IGNORECASE)),
                    'contains_floor': bool(re.search(r'\b(fl\b|floor\b)', address, re.IGNORECASE)),
                    'levenshtein_ratio': 0  # Will be filled in for similarity checks
                }
                
                # Country detection
                found_countries = []
                for code, name in self.countries.items():
                    if code in address.upper() or name.lower() in address.lower():
                        found_countries.append(code)
                
                country_features = {
                    'countries_detected': len(found_countries),
                    'has_country': len(found_countries) > 0
                }
                
                # Merge all features
                all_features = {
                    **text_features,
                    **regex_features,
                    **usaddress_features,
                    **advanced_features,
                    **country_features
                }
                
                features.append(all_features)
                
            except Exception as e:
                self.logger.error(f"Error extracting features from address: {e}")
                features.append(self._get_empty_features())
        
        # Convert to DataFrame
        return pd.DataFrame(features)
    
    def _get_empty_features(self) -> Dict:
        """Generate empty features for invalid addresses."""
        return {
            # Text features
            'length': 0,
            'word_count': 0,
            'avg_word_length': 0,
            'digit_ratio': 0,
            'uppercase_ratio': 0,
            'alpha_ratio': 0,
            'space_ratio': 0,
            
            # Regex features
            **{f'has_{name}': False for name in self.patterns},
            
            # usaddress features
            'usaddress_type': 'invalid',
            'usaddress_components': 0,
            'has_number': False,
            'has_street': False,
            'has_state': False,
            'has_zip': False,
            
            # advanced features
            'has_valid_format': False,
            'normalized_length': 0,
            'contains_directions': False,
            'contains_floor': False,
            'levenshtein_ratio': 0,
            
            # country features
            'countries_detected': 0,
            'has_country': False
        }
    
    def train(self, addresses: List[str], labels: List[bool], tune_hyperparameters: bool = False) -> Dict:
        """
        Train the address validation model.
        
        Args:
            addresses: List of addresses for training
            labels: Corresponding validity labels (True for valid, False for invalid)
            tune_hyperparameters: Whether to perform hyperparameter tuning
            
        Returns:
            Dictionary with training metrics
        """
        try:
            self.logger.info(f"Training model on {len(addresses)} addresses")
            
            # Extract features
            X_features = self.extract_features(addresses)
            X_text = addresses
            y = np.array(labels)
            
            # Split data
            X_features_train, X_features_test, X_text_train, X_text_test, y_train, y_test = train_test_split(
                X_features, X_text, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Define preprocessing pipeline
            categorical_features = ['usaddress_type']
            categorical_transformer = OneHotEncoder(handle_unknown='ignore')
            
            # Define feature pipeline
            preprocessor = ColumnTransformer(
                transformers=[
                    ('cat', categorical_transformer, categorical_features),
                    ('num', StandardScaler(), X_features.select_dtypes(include=['float64', 'int64']).columns)
                ],
                remainder='drop'
            )
            
            # Extract text features
            self.text_vectorizer.fit(X_text_train)
            X_text_train_vec = self.text_vectorizer.transform(X_text_train)
            X_text_test_vec = self.text_vectorizer.transform(X_text_test)
            
            # Process structured features
            X_features_train_proc = preprocessor.fit_transform(X_features_train)
            X_features_test_proc = preprocessor.transform(X_features_test)
            
            # Combine features
            X_train = np.hstack([X_features_train_proc.toarray(), X_text_train_vec.toarray()])
            X_test = np.hstack([X_features_test_proc.toarray(), X_text_test_vec.toarray()])
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average='binary'
            )
            
            # Store training artifacts
            self.preprocessor = preprocessor
            self.is_trained = True
            
            return {
                'accuracy': self.model.score(X_test, y_test),
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'classification_report': classification_report(y_test, y_pred)
            }
            
        except Exception as e:
            self.logger.error(f"Error training model: {e}")
            return {'error': str(e)}
    
    def predict(self, address: str) -> Dict[str, Any]:
        """
        Predict if an address is valid and provide confidence scores.
        
        Args:
            address: Address string to validate
            
        Returns:
            Dictionary with validation results
        """
        if not self.is_trained:
            return self._fallback_validation(address)
        
        try:
            # Extract features
            X_features = self.extract_features([address])
            X_text = [address]
            
            # Process features
            X_features_proc = self.preprocessor.transform(X_features)
            X_text_vec = self.text_vectorizer.transform(X_text)
            
            # Combine features
            X = np.hstack([X_features_proc.toarray(), X_text_vec.toarray()])
            
            # Predict
            is_valid = bool(self.model.predict(X)[0])
            probabilities = self.model.predict_proba(X)[0]
            confidence = float(probabilities[1] if is_valid else probabilities[0])
            
            # Parse components
            try:
                parsed, address_type = usaddress.tag(address)
            except:
                parsed = {}
                address_type = "parsing_error"
            
            return {
                'is_valid': is_valid,
                'confidence': confidence,
                'parsed_components': parsed,
                'address_type': address_type,
                'detected_features': {k: v for k, v in X_features.iloc[0].items() if v == True and k.startswith('has_')}
            }
            
        except Exception as e:
            self.logger.error(f"Error during prediction: {e}")
            return self._fallback_validation(address)
    
    def _fallback_validation(self, address: str) -> Dict[str, Any]:
        """Fallback validation when model is not trained."""
        # Basic heuristics for address validation
        if not address or not isinstance(address, str):
            return {'is_valid': False, 'confidence': 1.0, 'reason': 'Empty or invalid input'}
            
        # Extract basic features for heuristic validation
        try:
            parsed, address_type = usaddress.tag(address)
        except:
            parsed = {}
            address_type = "parsing_error"
            
        # Check for key address components
        has_number = bool(re.search(r'\d+', address))
        has_street = bool(re.search(self.patterns['street_suffix'], address, re.IGNORECASE))
        has_zip = bool(re.search(r'\b\d{5}(-\d{4})?\b', address))
        
        # Calculate validity score
        validity_checks = [
            len(address) > 10,
            has_number,
            has_street,
            has_zip,
            'AddressNumber' in parsed,
            'StreetName' in parsed,
            address_type != "parsing_error"
        ]
        
        score = sum(validity_checks) / len(validity_checks)
        is_valid = score > 0.6
        
        return {
            'is_valid': is_valid,
            'confidence': score,
            'parsed_components': parsed,
            'address_type': address_type,
            'heuristic_score': score,
            'method': 'fallback'
        }

    def compare_addresses(self, addr1: str, addr2: str, method: str = 'combined') -> float:
        """
        Compare two addresses and return a similarity score.
        
        Args:
            addr1: The first address string.
            addr2: The second address string.
            method: The comparison method ('ratio', 'partial_ratio', 'combined')
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not isinstance(addr1, str) or not isinstance(addr2, str):
            return 0.0
            
        try:
            # Normalize addresses
            norm_addr1 = self.normalize_address(addr1)
            norm_addr2 = self.normalize_address(addr2)
            
            # Calculate similarity using fuzzywuzzy
            if method == 'ratio':
                return fuzz.ratio(norm_addr1, norm_addr2) / 100.0
            elif method == 'partial_ratio':
                return fuzz.partial_ratio(norm_addr1, norm_addr2) / 100.0
            elif method == 'combined':
                ratio = fuzz.ratio(norm_addr1, norm_addr2)
                partial = fuzz.partial_ratio(norm_addr1, norm_addr2)
                token_sort = fuzz.token_sort_ratio(norm_addr1, norm_addr2)
                return (ratio * 0.4 + partial * 0.4 + token_sort * 0.2) / 100.0
            else:
                raise ValueError("Invalid method. Choose 'ratio', 'partial_ratio', or 'combined'")
                
        except Exception as e:
            self.logger.error(f"Error comparing addresses: {e}")
            return 0.0

# Create an alias/wrapper class for compatibility with app.py
class AddressCorrectionModel:
    """
    Wrapper class that implements the expected interface using EnhancedAddressValidator.
    """
    
    def __init__(self, api_base_url=None):
        """Initialize with EnhancedAddressValidator."""
        self.validator = EnhancedAddressValidator()
        self.api_base_url = api_base_url
    
    def normalize_address(self, address):
        """Normalize an address using internal validator."""
        return self.validator.normalize_address(address)
    
    def parse_address(self, address):
        """Parse address components."""
        try:
            import usaddress
            parsed, _ = usaddress.tag(address)
            return parsed
        except Exception as e:
            logger.error(f"Error parsing address: {e}")
            return {}
    
    def expand_address(self, address):
        """Return address variations."""
        # Use fallback since we're not actually using libpostal
        return [address] if address else []
    
    def suggest_corrections(self, address):
        """Suggest corrections for an address."""
        # Since we don't have direct equivalent, return empty for now
        return []
    
    def is_valid_address(self, address):
        """Check if an address appears valid."""
        result = self.validator.predict(address)
        return result.get('is_valid', False)