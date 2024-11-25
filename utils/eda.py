
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def summary_statistics(df):
    """Return summary statistics of the DataFrame."""
    return df.describe() 
def plot_distributions(df, columns):
    """Plot distributions of specified columns."""
    sns.pairplot(df, hue="BodyFat")
    plt.title(f'Distribution of ')
    plt.show()


# Handle Categorical Data
def handle_categorical_data(df , handle_way):
    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    handle_way = handle_way.lower()
    if len(categorical_cols) > 0 and handle_way == 'additional class':
        # Convert categorical columns to numeric using one-hot encoding
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    elif len(categorical_cols) > 0 and handle_way == 'mode':
        # Convert categorical columns to numeric using mode
        df[categorical_cols] = df[categorical_cols].apply(lambda col: col.fillna(col.mode().iloc[0]))
    else:
        pass

    
    return df



def handle_missing_continuous_values(df, handle_way):
    # Handle missing values
    handle_way = handle_way.lower()
    if handle_way == 'mean':
        df = df.fillna(df.mean())
    elif handle_way == 'median':
        df = df.fillna(df.median())
    elif handle_way == 'mode':
        df = df.fillna(df.mode().iloc[0])
    return df

def continuous_variables(df):
    # Identify continuous columns
    continuous_cols = df.select_dtypes(include=['float64', 'int64']).columns
    return continuous_cols , df.isnull().sum().sum()

def categorical_variables(df):
    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    return categorical_cols , df.isnull().sum().sum()



def drop_columns(df, columns_to_drop):
    # Drop specified columns
    df = df.drop(columns=columns_to_drop)
    return df