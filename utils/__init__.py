"""Utility modules for address comparison."""

# Import and expose EnhancedAddressValidator
from utils.address_utils import EnhancedAddressValidator

# Create alias for backward compatibility with app.py
class AddressCorrectionModel:
    """
    Wrapper class that provides the interface expected by app.py
    while using EnhancedAddressValidator underneath.
    """
    
    def __init__(self, api_base_url=None):
        """Initialize with EnhancedAddressValidator."""
        self.validator = EnhancedAddressValidator()
        self.api_base_url = api_base_url
        
    def normalize_address(self, address):
        """Normalize address using internal validator."""
        if not address:
            return ""
        # Just return the cleaned address if no specific normalization method
        return address.lower().strip()
    
    def parse_address(self, address):
        """Parse address components."""
        try:
            import usaddress
            parsed, _ = usaddress.tag(address)
            return parsed
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Error parsing address: {e}")
            return {}
    
    def expand_address(self, address):
        """Return address variations."""
        # Fallback implementation
        return [address] if address else []
    
    def suggest_corrections(self, address):
        """Suggest corrections for an address."""
        # Since EnhancedAddressValidator doesn't provide direct equivalent
        return []
    
    def is_valid_address(self, address):
        """Check if an address appears valid using validator."""
        # Basic validation fallback if validator doesn't have predict method
        if hasattr(self.validator, 'predict'):
            result = self.validator.predict(address)
            return result.get('is_valid', False)
        return bool(address and len(address) > 5)