# data_handler.py
import pandas as pd

def load_data(file_path):
    """Load data from a file."""
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        return pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format")

def load_from_sql(connection_string, query):
    """Load data from a SQL database."""
    import sqlalchemy
    engine = sqlalchemy.create_engine(connection_string)
    return pd.read_sql(query, con=engine)
