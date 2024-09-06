import importlib
import joblib
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE  # Import SMOTE
import itertools
import json
import random
import numpy as np
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE  # Import SMOTE

# Set the random seeds for reproducibility
random.seed(42)
np.random.seed(42)

class ModelTrainer:
    def __init__(self, X, y, model_name, vectorizer_type, use_smote, hyperparameters=None):
        self.X = X
        self.y = y
        self.model_name = model_name
        self.vectorizer_type = vectorizer_type
        self.use_smote = use_smote
        self.hyperparameters = hyperparameters
        self.accuracies = []
        self.precisions = []
        self.recalls = []
        self.f1_scores = []
        self.best_f1_score = 0
        self.best_model = None
        self.best_fold = None
        self.skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # Set seed for StratifiedKFold

    def get_model_class(self):
        module = importlib.import_module(f'{self.model_name}_model')
        class_name = ''.join([part.capitalize() for part in self.model_name.split('_')])
        model_class = getattr(module, f'{class_name}Model')
        return model_class

    def train_and_evaluate(self):
        model_class = self.get_model_class()
        fold = 1

        for train_index, test_index in self.skf.split(self.X, self.y):
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]

            # Ensure all data is in string format
            X_train = X_train.astype(str)
            X_test = X_test.astype(str)

            print(f"Fold {fold}: Training data size: {X_train.shape[0]}, Test data size: {X_test.shape[0]}")

            # Initialize model with vectorizer_type
            model = model_class(vectorizer_type=self.vectorizer_type)

            # Transform data
            X_train_transformed = model.get_vectorizer().fit_transform(X_train)
            X_test_transformed = model.get_vectorizer().transform(X_test)

            # Apply SMOTE if enabled
            if self.use_smote:
                smote = SMOTE(random_state=42)  # Set seed for SMOTE
                X_train_transformed, y_train = smote.fit_resample(X_train_transformed, y_train)
                print(f"SMOTE applied. New training data size: {X_train_transformed.shape[0]}")

            # Use RandomizedSearchCV for hyperparameter tuning
            if self.hyperparameters:
                search = RandomizedSearchCV(
                    model.model, 
                    self.hyperparameters, 
                    cv=3, 
                    n_jobs=-1, 
                    scoring='f1_weighted',
                    n_iter=24,  # Number of iterations for random search
                    random_state=42  # Seed for reproducibility
                )
                search.fit(X_train_transformed, y_train)
                model.model = search.best_estimator_
                best_params = search.best_params_
            else:
                model.train(X_train_transformed, y_train)
                best_params = model.get_params()

            y_pred = model.predict(X_test_transformed)

            accuracy = accuracy_score(y_test, y_pred)
            self.accuracies.append(accuracy)

            precision = precision_score(y_test, y_pred, average='weighted')
            self.precisions.append(precision)

            recall = recall_score(y_test, y_pred, average='weighted')
            self.recalls.append(recall)

            f1 = f1_score(y_test, y_pred, average='weighted')
            self.f1_scores.append(f1)

            if f1 > self.best_f1_score:
                self.best_f1_score = f1
                self.best_model = model
                self.best_fold = fold

            cm = confusion_matrix(y_test, y_pred)
            cr = classification_report(y_test, y_pred)

            # Save classification report and confusion matrix
            report_file_path = 'classification_report.txt'
            with open(report_file_path, 'a', encoding='utf-8') as report_file:
                report_file.write(f"Fold {fold}:\n")
                report_file.write(f"Accuracy: {accuracy:.4f}\n")
                report_file.write(f"Precision: {precision:.4f}\n")
                report_file.write(f"Recall: {recall:.4f}\n")
                report_file.write(f"F1-Score: {f1:.4f}\n")
                report_file.write(f"Best Hyperparameters: {best_params}\n")
                report_file.write("Confusion Matrix:\n")
                report_file.write(f"{cm.tolist()}\n")
                report_file.write("Classification Report:\n")
                report_file.write(f"{cr}\n")
                report_file.write("\n" + "="*50 + "\n\n")

            
            plt.figure(figsize=(8, 6))
            classes = np.unique(self.y)
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title(f"Confusion Matrix - Fold {fold}")
            plt.colorbar()
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)

            fmt = 'd'
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, format(cm[i, j], fmt),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.tight_layout()
            plt.savefig(f'confusion_matrix_fold_{fold}.png')
            plt.close()

            fold += 1

        # Calculate average metrics
        avg_accuracy = np.mean(self.accuracies)
        avg_precision = np.mean(self.precisions)
        avg_recall = np.mean(self.recalls)
        avg_f1 = np.mean(self.f1_scores)

        # Save accuracies and F1-scores
        with open('results.json', 'w') as file:
            json.dump({
                'accuracies': self.accuracies,
                'f1_scores': self.f1_scores,
                'best_f1_score': self.best_f1_score,
                'best_fold': self.best_fold,
                'average_accuracy': avg_accuracy,
                'average_precision': avg_precision,
                'average_recall': avg_recall,
                'average_f1_score': avg_f1
            }, file, indent=4)

        # Print the best fold details and average metrics
        print(f"Best fold: {self.best_fold} with F1-Score: {self.best_f1_score:.4f}")
        print(f"Average Accuracy: {avg_accuracy:.4f}")
        print(f"Average Precision: {avg_precision:.4f}")
        print(f"Average Recall: {avg_recall:.4f}")
        print(f"Average F1-Score: {avg_f1:.4f}")

        # Save the best model
        if self.best_model:
            joblib.dump(self.best_model, 'best_model.pkl')
