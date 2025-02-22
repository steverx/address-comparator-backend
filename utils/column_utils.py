# utils/column_utils.py
import pandas as pd
from typing import List
import re

def identify_address_columns(df: pd.DataFrame, keywords: List[str]) -> List[str]:
    """Identifies potential address columns based on keywords."""
    address_columns = []
    for col in df.columns:
        col_lower = col.lower()
        # Check if any keyword is present in the column name
        if any(keyword in col_lower for keyword in keywords):
            address_columns.append(col)
        # Also, check a few sample values from that column (heuristic approach)
        else:
            sample_values = df[col].dropna().astype(str).head(3)  # Get first 3 non-null string values
            if len(sample_values) > 0: #Check if values were found.
                # Basic checks (you can add more sophisticated checks here)
                if any(re.search(r'\d', val) for val in sample_values):  # Check for digits
                    address_columns.append(col)

    return address_columns


def get_column_preview(df: pd.DataFrame, column_name: str, num_rows: int = 5) -> List:
    """Gets a preview of the values in a specified column (first num_rows)."""
    return df[column_name].head(num_rows).tolist()