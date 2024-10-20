
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

def automl_classifier(X, y):
    """Automatically select the best classifier model."""
    models = {
        'RandomForest': RandomForestClassifier(),
        'GradientBoosting': GradientBoostingClassifier()
    }
    
    best_score = 0
    best_model = None

    for model_name, model in models.items():
        score = cross_val_score(model, X, y, cv=5).mean()
        if score > best_score:
            best_score = score
            best_model = model

    return best_model, best_score
