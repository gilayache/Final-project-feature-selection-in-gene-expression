from sklearn.base import BaseEstimator, TransformerMixin


class Imputer(BaseEstimator, TransformerMixin):
    def __init__(self, numerical_features, categorical_features, num_method="constant", cat_method="most_frequent", value="missing"):
        """

        """
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.num_method = num_method
        self.cat_method = cat_method
        self.value = value

    def fit(self, X, y=None):
        """

        """
        if self.num_method == "mean":
            self.num_value = X[self.numric_features].mean()

        elif self.num_method == "most_frequent":
            self.num_value = X[self.numeric_features].mode().iloc[0]

        elif self.num_method == "constant":
            self.num_value = self.value

        if self.cat_method == "most_frequent":
            self.cat_value = X[self.categorical_features].mode().iloc[0]

        return self

    def transform(self, X):
        """

        """
        X_transformed = X.copy()
        X_transformed[self.numerical_features] = X[self.numerical_features].fillna(self.num_value)
        X_transformed[self.categorical_features] = X[self.categorical_features].fillna(self.cat_value)
        return X_transformed
