import pandas as pd

def load_data(file_path):
    """Load data from a file."""
    if file_path.name.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.name.endswith('.xlsx'):
        return pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format")


