import pandas as pd
from category_encoders import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin


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
        # self.encoder = OneHotEncoder()
        self.encoder.fit(X)
        return self

    def transform(self, X):
        """

        """
        X_transformed = pd.concat(
            [
                X.drop(columns=self.features).reset_index(drop=True),
                pd.DataFrame(
                    self.encoder.transform(X[self.features]),
                    columns=self.encoder.get_feature_names_out(),
                ),
            ],
            axis=1,
        )
        return X_transformed

    def InitEncoder(self, encoder_name):
        """

        """
        if encoder_name == 'OneHotEncoder':
            self.encoder = OneHotEncoder()
        #
        # elif encoder_name == 'OrdinalEncoder':
        #     self.encoder = OrdinalEncoder()