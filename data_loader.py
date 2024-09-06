import pandas as pd

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    def load_data(self):
        self.df = pd.read_csv(self.file_path)
        print("First five rows of the dataset:")
        print(self.df.head())
        print("\nDataset information:")
        print(self.df.info())
        
        # إعادة X و y
        X = self.df['Masseges']
        y = self.df['Category']
        return X, y
