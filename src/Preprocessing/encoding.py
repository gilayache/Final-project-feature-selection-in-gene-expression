import pandas as pd
from category_encoders import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import time
start_time = time.time()


class Encoder(BaseEstimator, TransformerMixin):
    def __init__(self, encoder_name, features,  drop="first"):
        """

        """
        self.features = features
        self.drop = drop
        self.encoder_name = encoder_name
        self.encoder = self.InitEncoder(encoder_name)


    def fit(self, X, y=None):
        """

        """
        self.encoder.fit(X)

        return self

    def transform(self, X):
        """

        """
        X_transformed = pd.DataFrame(self.encoder.transform(X[self.features]))

        self.encoded_features = X_transformed.columns.to_list()
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Encoding was done successfully in {elapsed_time:.2f} seconds")
        return X_transformed

    def InitEncoder(self, encoder_name):
        """

        """
        if encoder_name == 'OneHotEncoder':
            encoder = OneHotEncoder()
        #
        # elif encoder_name == 'OrdinalEncoder':
        #     encoder = OrdinalEncoder()

        return encoder