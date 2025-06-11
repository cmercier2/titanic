"""
Train the Titanic model.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

def train_model(X_train, y_train):
    """ 
    Initiate the model and train it on the Titanic dataset.

    """
    lin = LogisticRegression(max_iter=1000, random_state=42)
    lin.fit(X_train, y_train)
    return lin


def evaluate_model(X_test, y_test, lin):
    lin.score(X_test, y_test)
    print(classification_report(y_test, lin.predict(X_test)))

def optimize_model(X_train, y_train, lin):
    param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
    }

    grid_search = GridSearchCV(lin, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_
