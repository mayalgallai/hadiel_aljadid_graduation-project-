from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import SVC
from abstract_model import AbstractModel

class SvcModel(AbstractModel):
    def __init__(self, vectorizer_type='tfidf', **kwargs):
        super().__init__(vectorizer_type)
        self.vectorizer = TfidfVectorizer() if vectorizer_type == 'tfidf' else CountVectorizer()
        self.model = SVC(**kwargs)

    def train(self, X_train_transformed, y_train):
        self.model.fit(X_train_transformed, y_train)

    def predict(self, X_test_transformed):
        return self.model.predict(X_test_transformed)

    def get_vectorizer(self):
        return self.vectorizer

    def set_params(self, **kwargs):
        self.model.set_params(**kwargs)

    def get_params(self):
        return self.model.get_params()
