class Model:

    def __init__(self, model_name: str):
        """
        """
        self.model = self.InitModel(model_name)

    def fit(self, X, y=None):
        """
        """
        import pandas as pd
        if type(y) != pd.Series: # todo: change it to == np.array/np.ndarray
            y = y.reshape(-1, 1)

        elif type(y) == pd.Series:
            y = y.values.reshape(-1, 1)

        self.model.fit(X, y)

        return self

    def transform(self, X):
        """
        """
        X_transformed = self.model.transform(X)

        return X_transformed

    def predict(self, X):
        """
        """
        y_pred = self.model.predict(X)

        return y_pred

    def InitModel(self, model_name):
        """
        """
        from sklearn.linear_model import LogisticRegression, LinearRegression

        if model_name == 'LogisticRegression':
            model = LogisticRegression()

        elif model_name == 'LinearRegression':
            model = LinearRegression()

        return model

