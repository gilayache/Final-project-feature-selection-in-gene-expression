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


# Running the pipeline
data_path = "data/processed/merged_dataset.csv"
model = LinearRegression()
df = load_data(data_path)
# Load data
SEED = 42
columns_to_remove = ["ER", "Lympho", "samplename"]
# Drop the columns
FEATURES = df.columns.drop(columns_to_remove)

NUMERICAL = df[FEATURES].select_dtypes("number").columns
CATEGORICAL = pd.Index(np.setdiff1d(FEATURES, NUMERICAL))

# Prepare the data
X = df[FEATURES]
y = df["Lympho"]

# Perform the train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

# Create the main pipeline (including the model)
pipe = Pipeline([
    ('column_dropper', FunctionTransformer(lambda X: X.drop(columns=[col for col in columns_to_remove if col in X.columns]))),
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
    ('model', model)
])

# Fit the pipeline using the training data
pipe.fit(X_train, y_train)

# Predict using the pipeline on the test data
y_test_pred = pipe.predict(X_test)

# Evaluate the pipeline on the test data
mse = mean_squared_error(y_test, y_test_pred)
print("Mean Squared Error:", mse)



