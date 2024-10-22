

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

def automl_classifier(features, target):
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    # Standardize features for SVM
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
 
    models = {
        'RandomForest': RandomForestClassifier(),
        'SVM': SVC(),
        'LogisticRegression': LogisticRegression(),
        'KNN': KNeighborsClassifier(),
        'DecisionTree': DecisionTreeClassifier(),
        'NaiveBayes': GaussianNB()
    }
    
    best_model = None
    best_score = 0
    model_scores = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        model_scores[name] = score
        
        if score > best_score:
            best_score = score
            best_model = model
            
    print("Model scores:", model_scores)
    return best_model, best_score, model_scores

