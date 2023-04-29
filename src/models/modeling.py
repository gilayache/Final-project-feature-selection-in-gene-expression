from sklearn.linear_model import LogisticRegression, LinearRegression
import pandas as pd

class Model():

    def __init__(self, model_name: str):
        """

        """
        # self.model_name = model_name
        self.model = self.InitModel(model_name)

    def fit(self, X, y=None):
        """

        """
        if type(y) == pd.Series:
            y = y.values.reshape(-1, 1)
        self.model.fit(X, y)
        return self

    def transform(self, X):
        """

        """
        X_transformed = self.model.transform(X)
        return X_transformed

    def InitModel(self, model_name):
        """

        """
        if model_name == 'LogisticRegression':
            model = LogisticRegression()

        elif model_name == 'LinearRegression':
            model = LinearRegression()

        return model