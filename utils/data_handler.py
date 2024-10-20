
import pandas as pd

"""
* upload file 
* result from uploading file => UploadedFile(
file_id='97501b1d-54a9-4de9-b61e-fab85ff631c0',
name='telecom_churn.csv', type='text/csv', size=310007, _file_urls=file_id: "97501b1d-54a9-4de9-b61e-fab85ff631c0"
upload_url: "/_stcore/upload_file/96d8e23b-8203-4228-982e-14834e11a9b7/97501b1d-54a9-4de9-b61e-fab85ff631c0"        
delete_url: "/_stcore/upload_file/96d8e23b-8203-4228-982e-14834e11a9b7/97501b1d-54a9-4de9-b61e-fab85ff631c0"        
)
"""
def load_data(file_path):
    """Load data from a file."""
    if file_path.type.endswith('text/csv'):
        return pd.read_csv(file_path)
    elif file_path.type.endswith('.xlsx'):
        return pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format")

def load_from_sql(connection_string, query):
    """Load data from a SQL database."""
    import sqlalchemy
    engine = sqlalchemy.create_engine(connection_string)
    return pd.read_sql(query, con=engine)
