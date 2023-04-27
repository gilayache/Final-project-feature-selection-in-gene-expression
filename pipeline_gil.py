# Data manipulation
import numpy as np
import pandas as pd

pd.options.display.precision = 4
pd.options.mode.chained_assignment = None
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import set_config
from sklearn.preprocessing import FunctionTransformer

set_config(display="diagram")

class Imputer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        num_features,
        cat_features,
        num_method="constant",
        cat_method="most_frequent",
        value="missing",
    ):
        self.num_features = num_features
        self.cat_features = cat_features
        self.num_method = num_method
        self.cat_method = cat_method
        self.value = value

    def fit(self, X, y=None):
        if self.num_method == "mean":
            self.num_value = X[self.num_features].mean()
        elif self.num_method == "most_frequent":
            self.num_value = X[self.num_features].mode().iloc[0]
        elif self.num_method == "constant":
            self.num_value = self.value

        if self.cat_method == "most_frequent":
            self.cat_value = X[self.cat_features].mode().iloc[0]

        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed[self.num_features] = X[self.num_features].fillna(self.num_value)
        X_transformed[self.cat_features] = X[self.cat_features].fillna(self.cat_value)
        return X_transformed

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

class Encoder(BaseEstimator, TransformerMixin):
    def __init__(self, features, drop="first"):
        self.features = features
        self.drop = drop

    def fit(self, X, y=None):
        self.encoder = OneHotEncoder(sparse_output=False, drop=self.drop)
        self.encoder.fit(X[self.features])
        return self

    def transform(self, X):
        X_transformed = pd.concat(
            [
                X.drop(columns=self.features).reset_index(drop=True),
                pd.DataFrame(
                    self.encoder.transform(X[self.features]),
                    columns=self.encoder.get_feature_names_out(self.features),
                ),
            ],
            axis=1,
        )
        return X_transformed

def load_data(data_path):
    return pd.read_csv(data_path)

class RunPipeline:
    def __init__(self, run_type: str, target_col: str, input_path: str, output_path: str, model_type: str, test_size: float):
        self.run_type = run_type ### why do we need this ???? ###
        self.model_type = model_type
        self.target_col = target_col
        self.input_path = input_path or 'data/processed/merged_dataset.csv'
        self.output_path = output_path
        self.df = pd.DataFrame()
        self.test_size = test_size

        # Define the additional parameters
        self.SEED = 42
        # Choose the model based on the model_type parameter columns_to_remove changed
        if self.model_type == 'regression':
            self.model = LinearRegression()
            self.columns_to_remove = ["Lympho","samplename"]
        elif self.model_type == 'classification':
            self.model = LogisticRegression()
            self.columns_to_remove = ["ER", "samplename"]
        else:
            raise ValueError(f"Invalid model_type: {self.model_type}. Choose either 'regression' or 'classification'.")

    def run(self):
        df = load_data(self.input_path)
        FEATURES = df.columns.drop(self.columns_to_remove)

        NUMERICAL = df[FEATURES].select_dtypes("number").columns
        CATEGORICAL = pd.Index(np.setdiff1d(FEATURES, NUMERICAL))

        X = df[FEATURES]
        y = df[self.target_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.SEED)

        pipe = Pipeline([
            ('column_dropper',
             FunctionTransformer(lambda X: X.drop(columns=[col for col in self.columns_to_remove if col in X.columns]))),
            ('preprocessor', ColumnTransformer(transformers=[
                ('num', Pipeline([
                    ('num_imputer', SimpleImputer(strategy='mean')),
                    ('scaler', MinMaxScaler())
                ]), NUMERICAL),
                ('cat', Pipeline([
                    ('cat_imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OneHotEncoder(drop='first'))
                ]), CATEGORICAL)
            ])),
            ('model', self.model)
        ])

        pipe.fit(X_train, y_train)

        y_test_pred = pipe.predict(X_test)

        mse = mean_squared_error(y_test, y_test_pred)
        print("Mean Squared Error:", mse)

if __name__ == '__main__':
    run_pipeline = RunPipeline(model_type='regression', target_col='Lympho',
                               input_path='data/processed/merged_dataset.csv', output_path=None,
                               run_type='train',
                               test_size=0.2)
    run_pipeline.run()


