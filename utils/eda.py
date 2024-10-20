
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def summary_statistics(df):
    """Return summary statistics of the DataFrame."""
    return df.describe()

def plot_distributions(df, columns):
    """Plot distributions of specified columns."""
    for col in columns:
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.show()
