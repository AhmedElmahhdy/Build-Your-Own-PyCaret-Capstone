
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def summary_statistics(df):
    """Return summary statistics of the DataFrame."""
    return df.describe() , df.isna().sum()
def plot_distributions(df, columns):
    """Plot distributions of specified columns."""
    for col in columns:
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.show()


# Handle Categorical Data
def convert_categorical_to_numeric(df):
    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    if len(categorical_cols) > 0:
        print(f"Categorical columns found: {list(categorical_cols)}")
        # Convert categorical columns to numeric using one-hot encoding
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    else:
        print("No categorical columns found.")
    
    return df
