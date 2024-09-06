#may moktar algallai 
#hadil ali aljadid
from random import randint, uniform
import pandas as pd
from data_loader import DataLoader
from data_preprocessor import DataPreprocessor
from model_trainer import ModelTrainer
from model_trainer import ModelTrainer

def main():
    # Model and vectorizer settings
    model_name = 'svc'#  decision_tree, logistic_regression, naive_bayes, svc
    vectorizer_type = 'tfidf'  # or 'bow'
    use_hyperparameters = True
    use_smote = True  # New flag for applying SMOTE

    # Load data
    data_loader = DataLoader('cleaneddata.csv')
    X, y = data_loader.load_data()

    # Preprocess data
    preprocessor = DataPreprocessor(pd.DataFrame({'Masseges': X, 'Category': y}))
    X_cleaned, y_cleaned = preprocessor.preprocess()

    # Hyperparameters dictionary for models
    hyperparameters = {
        'decision_tree': {
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random'],
            'max_depth': [None, 10, 20, 30, 40, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': [None, 'sqrt', 'log2']
        },
        'logistic_regression': {
            'C': [uniform(0.1, 100)],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],
            'tol': [1e-4, 1e-3, 1e-2]
        },
        'svc': {
            'C': [uniform(0.1, 100)],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'degree': [randint(2, 5)],
            'coef0': [uniform(0, 1)]
        },
        'naive_bayes': {
            'alpha': [uniform(0.01, 10)],
            'fit_prior': [True, False]
        }
    }

    # Initialize and run ModelTrainer
    model_trainer = ModelTrainer(
        X_cleaned,
        y_cleaned,
        model_name=model_name,  # Pass model_name here
        vectorizer_type=vectorizer_type,
        hyperparameters=hyperparameters.get(model_name) if use_hyperparameters else None,#pass the  Hyperparameters
        use_smote=use_smote  # Pass the SMOTE flag to ModelTrainer
    )
    model_trainer.train_and_evaluate()

if __name__ == '__main__':
    main()
