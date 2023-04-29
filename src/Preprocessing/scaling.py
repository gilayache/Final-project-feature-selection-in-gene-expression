from sklearn.base import BaseEstimator, TransformerMixin


class Scaler(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        self.features = features


    def fit(self, X, y=None):
        self.min = X[self.features].min()
        self.range = X[self.features].max() - self.min
        self.range[self.range == 0] = 1  # Replace 0 with 1 to avoid division by zero
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed[self.features] = (X[self.features] - self.min) / self.range
        return X_transformed
