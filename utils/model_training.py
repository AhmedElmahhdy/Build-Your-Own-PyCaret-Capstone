# model_training.py
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

def train_model(df, target_column, model_type='classifier', model_name='RandomForest'):
    """Train a machine learning model."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if model_type == 'classifier':
        model = RandomForestClassifier() if model_name == 'RandomForest' else None
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        return accuracy_score(y_test, predictions), model
    
    elif model_type == 'regressor':
        model = RandomForestRegressor() if model_name == 'RandomForest' else None
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        return mean_squared_error(y_test, predictions), model

    else:
        raise ValueError("Unsupported model type")
