from sklearn.base import BaseEstimator, TransformerMixin
import time
start_time = time.time()



class Scaler(BaseEstimator, TransformerMixin):
    def __init__(self, scaler_name = None):
        """

        """
        self.scaler_name = scaler_name
        # self.encoded_features = encoded_features

    def fit(self, X, y=None):
        """

        """
        self.min = X.min()
        self.range = X.max() - self.min
        self.range[self.range == 0] = 1  # Replace 0 with 1 to avoid division by zero

        return self

    def transform(self, X):
        """

        """
        X_transformed = X.copy()
        X_transformed = (X - self.min) / self.range

        end_time = time.time()
        total_time = end_time - start_time
        print(f"Scaling was done successfully in {total_time:.2f} seconds")
        # print(X_transformed)

        return X_transformed
