from sklearn.base import BaseEstimator, TransformerMixin
import time
start_time = time.time()

class Imputer(BaseEstimator, TransformerMixin):
    def __init__(self, numerical_features, categorical_features, num_method="constant", cat_method="most_frequent", value="missing"):
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.num_method = num_method
        self.cat_method = cat_method
        self.value = value

    def fit(self, X, y=None):
        if self.num_method == "mean":
            self.num_value = {col: X[col].mean() for col in self.numerical_features}
        elif self.num_method == "most_frequent":
            self.num_value = {col: X[col].mode().iloc[0] for col in self.numerical_features}
        elif self.num_method == "constant":
            self.num_value = {col: self.value for col in self.numerical_features}

        if self.cat_method == "most_frequent":
            self.cat_value = {col: X[col].mode().iloc[0] for col in self.categorical_features}
        else:
            self.cat_value = {col: self.value for col in self.categorical_features}

        return self

    def transform(self, X):

        X_transformed = X.copy()
        # fill missing values in one step.
        X_transformed.fillna(value={**self.num_value, **self.cat_value}, inplace=True)

        end_time = time.time()
        total_time = end_time - start_time
        print(f"Imputation was done successfully in {total_time:.2f} seconds")
        return X_transformed
