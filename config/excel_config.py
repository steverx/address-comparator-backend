from typing import Dict, List

# Excel file processing configuration
EXCEL_CONFIG: Dict = {
    'chunk_size': 1000,
    'supported_extensions': ['.xlsx', '.xls'],
    'max_file_size': 100 * 1024 * 1024,  # 100MB
    'sheet_name': 0  # First sheet
}

# Keywords to help identify potential address columns
ADDRESS_KEYWORDS: List[str] = [
    'address',
    'street',
    'avenue',
    'road',
    'lane',
    'boulevard',
    'city',
    'state',
    'province',
    'zip',
    'postal',
    'code'
]