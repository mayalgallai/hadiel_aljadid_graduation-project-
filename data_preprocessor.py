import pandas as pd

class DataPreprocessor:
    def __init__(self, df):
        self.df = df

    def preprocess(self):
        # حذف أي قيم مفقودة في الأعمدة المستهدفة
        self.df = self.df.dropna(subset=['Masseges', 'Category'])
        
        # تحويل جميع القيم في العمود 'Masseges' إلى نصوص
        self.df['Masseges'] = self.df['Masseges'].astype(str)
        
        # تعيين X و y
        X = self.df['Masseges']
        y = self.df['Category']
        
        # طباعة معلومات عن القيم المفقودة
        print("\nMissing values in X:", X.isnull().sum())
        print("Missing values in y:", y.isnull().sum())
        
        return X, y
