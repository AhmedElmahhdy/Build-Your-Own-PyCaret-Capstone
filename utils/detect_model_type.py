import pandas as pd
import numpy as np

def detect_task_type(y):
    """
    Detects whether the task is classification or regression based on the target variable.

    Parameters:
        y (pd.Series): The target column.

    Returns:
        str: "classification" or "regression"
    """
      # Ensure the input is a pandas Series
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    # Check if target variable is categorical
    if y.dtype.name == 'object' or y.dtype.name == 'category':
        return "classifier"
    else:
        return "regressor"

