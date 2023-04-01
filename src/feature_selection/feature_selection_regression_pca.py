import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin

# Custom transformer
class DimensionalityReductionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, method='PCA', n_components=None, explained_variance=None):
        self.method = method
        self.n_components = n_components
        self.explained_variance = explained_variance

    def _get_num_components(self, X):
        pca = PCA().fit(X)
        cum_explained_variance = np.cumsum(pca.explained_variance_ratio_)
        num_components = np.searchsorted(cum_explained_variance, self.explained_variance) + 1
        return num_components

    def fit(self, X, y=None):
        if self.method == 'PCA':
            if self.explained_variance is not None:
                self.n_components = self._get_num_components(X)
            self.dim_reduction_ = PCA(n_components=self.n_components)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        self.dim_reduction_.fit(X)
        return self

    def transform(self, X):
        return self.dim_reduction_.transform(X)

    def set_params(self, **params):
        self.__dict__.update(params)
        return self


# Create a pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('dim_reduction', DimensionalityReductionTransformer(method='PCA')),
    ('regressor', LinearRegression())
])

# Load the dataset
data = pd.read_csv("../../data/processed/merged_dataset.csv")
# Dropping irrelevant columns (keeping only the gene expression data)
X = data.iloc[:, 1:-6]

# Dropping columns with all missing values
cols_with_nan = X.columns[X.isna().any()].tolist()
[
    X.drop(columns=col, inplace=True)
    for col in cols_with_nan
    if X[col].isna().sum() / X.shape[0] == 1
]

# Filling missing values with 0
X = X.fillna(0)

# The target variable
y = data["Lympho"]

# Splitting the data into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)

# Initialize the Linear Regression model
lr = LinearRegression()

def apply_pca(X, explained_variance, pca=None, plot_scree=True):
    if pca is None:
        pca = PCA(n_components=explained_variance)
        X_pca = pca.fit_transform(X)

        if plot_scree:
            plt.plot(np.arange(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_))
            plt.xlabel("Number of Components")
            plt.ylabel("Cumulative Explained Variance")
            plt.title("Scree Plot")
            plt.show()
    else:
        X_pca = pca.transform(X)

    return X_pca, pca


# Apply PCA with specified explained variance
X_train_pca, pca = apply_pca(X_train, explained_variance=0.99) # pca parameter

# Fit the Linear Regression model on the PCA-transformed training dataset
lr.fit(X_train_pca, y_train)

# Transform the test dataset using the previously fitted PCA object
X_test_pca, _ = apply_pca(X_test, explained_variance=0.99, pca=pca, plot_scree=False) # pca parameter

# Make predictions using the test dataset
y_pred_pca = lr.predict(X_test_pca)

# Calculate the mean squared error after PCA
mse_pca = mean_squared_error(y_test, y_pred_pca)

# Fit the Linear Regression model directly on the training dataset without PCA
lr.fit(X_train, y_train)

# Make predictions using the test dataset without PCA
y_pred_no_transform = lr.predict(X_test)

# Calculate the mean squared error before applying PCA using the Linear Regression model
mse_no_transform = mean_squared_error(y_test, y_pred_no_transform)

# Number of features before and after PCA
num_features_before = X_train.shape[1]
num_features_after_pca = X_train_pca.shape[1]

# Print the number of features before and after PCA
print("Number of features before PCA: ", num_features_before)
print("Number of features after PCA: ", num_features_after_pca)

# Print the MSE before and after applying PCA
print("MSE before applying PCA: ", round(mse_no_transform,4))
print("MSE after applying PCA: ", round(mse_pca,4))
