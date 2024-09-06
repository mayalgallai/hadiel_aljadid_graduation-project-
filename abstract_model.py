from abc import ABC, abstractmethod

class AbstractModel(ABC):
    def __init__(self, vectorizer_type='tfidf'):
        if vectorizer_type not in ['tfidf', 'bow']:
            raise ValueError("Unsupported vectorizer type. Use 'tfidf' or 'bow'.")
        self.vectorizer_type = vectorizer_type

    @abstractmethod
    def train(self, X_train_transformed, y_train):
        pass

    @abstractmethod
    def predict(self, X_test_transformed):
        pass

    @abstractmethod
    def get_vectorizer(self):
        pass

    @abstractmethod
    def set_params(self, **kwargs):
        pass

    @abstractmethod
    def get_params(self):
        pass
