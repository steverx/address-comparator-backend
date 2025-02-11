import re
import usaddress
import pyap


def clean_address_basic(address):
    address = address.lower()
    address = re.sub(r'[^\w\s\'-]', '', address)  # Keep apostrophes and hyphens
    address = re.sub(r'(?<!\w)-\s|\s-(?!\w)', ' ', address)  # Remove hyphens not between words
    address = re.sub(r'\b(st|str)\b', 'street', address)
    address = re.sub(r'\b(ave|av)\b', 'avenue', address)
    address = re.sub(r'\b(rd)\b', 'road', address)
    address = re.sub(r'\b(ln)\b', 'lane', address)
    address = re.sub(r'\b(dr)\b', 'drive', address)
    address = re.sub(r'\b(blvd)\b', 'boulevard', address)
    address = re.sub(r'\b(apt|ste)\b', 'suite', address)
    address = re.sub(r'\b(n|no)\b', 'north', address)
    address = re.sub(r'\b(s)\b', 'south', address)
    address = re.sub(r'\b(e)\b', 'east', address)
    address = re.sub(r'\b(w)\b', 'west', address)
    address = ' '.join(address.split())  # Remove extra whitespace
    return address


def parse_address_usaddress(address):
    """Parses and returns standardized address and confidence (US)."""
    try:
        tagged_address, address_type = usaddress.tag(address)
        standardized = []
        confidence = 0.8  # Base confidence

        if address_type == 'Ambiguous':
            confidence -= 0.3

        for part, label in tagged_address.items():
            # Include ALL parts of the address
            if label == 'PlaceName':  # City
                standardized.append(part.lower())
            elif label == 'StateName':  # State
                standardized.append(part.upper())  # Uppercase state for consistency
            elif label == 'ZipCode':  # ZIP Code
                standardized.append(part)
            elif label in ('AddressNumber', 'StreetNamePreDirectional', 'StreetName', 'StreetNamePostType',
                           'StreetNamePostDirectional', 'OccupancyType', 'OccupancyIdentifier'):
                standardized.append(part.lower())
            # Keep this for confidence calculation (optional parts)
            elif label in ('StreetNamePreDirectional', 'StreetNamePostDirectional', 'OccupancyType'):
                confidence -= 0.05

        return " ".join(standardized), confidence

    except usaddress.RepeatedLabelError:
        # usaddress can't confidently parse
        return clean_address_basic(address), 0.3
    except Exception as e:
        # Catch any other parsing errors
        print(f"Error in parse_address_usaddress: {e}") # added for testing
        return clean_address_basic(address), 0.2


def parse_address_pyap(address):
    """Parses and returns standardized address and confidence (US/CA/UK)."""
    try:
        addresses = pyap.parse(address, country='US')  # Specify country
        if addresses:
            addr = addresses[0]  # pyap returns a list; take the first (best) parse
            standardized = []
            confidence = 0.7  # Base confidence

            # Include ALL parts of the address
            if addr.street_number: standardized.append(addr.street_number.lower())
            if addr.street_name_pre_directional: standardized.append(addr.street_name_pre_directional.lower())
            if addr.street_name: standardized.append(addr.street_name.lower())
            if addr.street_name_post_type: standardized.append(addr.street_name_post_type.lower())
            if addr.street_name_post_directional: standardized.append(addr.street_name_post_directional.lower())
            if addr.city: standardized.append(addr.city.lower())
            if addr.region1: standardized.append(addr.region1.upper()) #state or province
            if addr.postal_code: standardized.append(addr.postal_code)
            if addr.occupancy_type: standardized.append(addr.occupancy_type.lower())
            if addr.occupancy_id: standardized.append(addr.occupancy_id.lower())

            # Adjust confidence based on presence of key components
            if not addr.street_number: confidence -= 0.1
            if not addr.street_name: confidence -= 0.2
            # we are now parsing all address components

            return " ".join(standardized), max(0, confidence)  # Ensure confidence is not negative
        else:
            return clean_address_basic(address), 0.3  # Lower confidence if parsing fails
    except Exception as e:
        print(f"Error in parse_address_pyap: {e}")
        return clean_address_basic(address), 0.2



def clean_address(address, parser='usaddress'):
    """Cleans, standardizes, and returns address and confidence."""
    cleaned = clean_address_basic(address)  # Always do basic cleaning first

    if parser == 'usaddress':
        return parse_address_usaddress(cleaned)
    elif parser == 'pyap':
        return parse_address_pyap(cleaned)
    else:
        return cleaned, 0.5 # Return basic cleaned address with moderate confidence