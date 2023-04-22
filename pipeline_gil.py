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
from sklearn.metrics import roc_auc_score
from sklearn import set_config
set_config(display="diagram")

# Load data
df = pd.read_csv("data/processed/merged_dataset.csv")

SEED = 42
TARGET = ["ER", "Lympho","samplename"]
FEATURES = df.columns.drop(TARGET)

NUMERICAL = df[FEATURES].select_dtypes('number').columns
print(f"Numerical features: {', '.join(NUMERICAL)}")

CATEGORICAL = pd.Index(np.setdiff1d(FEATURES, NUMERICAL))
print(f"Categorical features: {', '.join(CATEGORICAL)}")


class Imputer(BaseEstimator, TransformerMixin):
    def __init__(self, features, method='constant', value='missing'):
        self.features = features
        self.method = method
        self.value = value

    def fit(self, X, y=None):
        if self.method == 'mean':
            self.value = X[self.features].mean()
        elif self.method == 'most_frequent':
            self.value = X[self.features].mode().iloc[0]
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed[self.features] = X[self.features].fillna(self.value)
        return X_transformed


class Scaler(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        self.min = X[self.features].min()
        self.range = X[self.features].max() - self.min
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed[self.features] = (X[self.features] - self.min) / self.range
        return X_transformed


class Encoder(BaseEstimator, TransformerMixin):
    def __init__(self, features, drop='first'):
        self.features = features
        self.drop = drop

    def fit(self, X, y=None):
        self.encoder = OneHotEncoder(sparse_output=False, drop=self.drop)
        self.encoder.fit(X[self.features])
        return self

    def transform(self, X):
        X_transformed = pd.concat([X.drop(columns=self.features).reset_index(drop=True),
                                   pd.DataFrame(self.encoder.transform(X[self.features]),
                                                columns=self.encoder.get_feature_names_out(self.features))],
                                  axis=1)
        return X_transformed

# Create the pipeline (including the model)
pipe = Pipeline([
    ('num_imputer', Imputer(NUMERICAL, method='mean')),
    ('scaler', Scaler(NUMERICAL)),
    ('cat_imputer', Imputer(CATEGORICAL, method='most_frequent')),
    ('encoder', Encoder(CATEGORICAL)),
    ('model', LinearRegression())
])

# Prepare the data
X = df[FEATURES]
y = df["Lympho"]

# Perform the train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

# Fit the pipeline (including the model) using the training data
pipe.fit(X_train, y_train)

# Evaluate the pipeline on the test data
y_pred = pipe.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)