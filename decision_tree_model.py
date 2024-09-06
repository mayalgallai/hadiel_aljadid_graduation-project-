from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from abstract_model import AbstractModel

class DecisionTreeModel(AbstractModel):
    def __init__(self, vectorizer_type='tfidf', **kwargs):
        super().__init__(vectorizer_type)
        self.vectorizer = None  # Initialize as None
        if vectorizer_type == 'tfidf':
            self.vectorizer = TfidfVectorizer()
        elif vectorizer_type == 'bow':
            self.vectorizer = CountVectorizer()
        # If 'pretransformed', do not initialize a vectorizer
        self.model = DecisionTreeClassifier(**kwargs)

    def train(self, X_train_transformed, y_train):
        self.model.fit(X_train_transformed, y_train)

    def predict(self, X_test_transformed):
        return self.model.predict(X_test_transformed)

    def get_vectorizer(self):
        if self.vectorizer_type == 'pretransformed':
            raise ValueError("No vectorizer available for pretransformed data.")
        return self.vectorizer

    def set_params(self, **kwargs):
        self.model.set_params(**kwargs)

    def get_params(self):
        return self.model.get_params()
