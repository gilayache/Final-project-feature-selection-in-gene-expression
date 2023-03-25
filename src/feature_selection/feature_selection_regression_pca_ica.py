import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA, FastICA

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

def apply_ica(X, n_components, ica=None):
    if ica is None:
        ica = FastICA(n_components=n_components)
        X_ica = ica.fit_transform(X)
    else:
        X_ica = ica.transform(X)

    return X_ica, ica

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

# Calculate the mean squared error before PCA using the naive model
y_pred_naive = np.mean(y_train)
mse_naive = mean_squared_error(y_test, np.full_like(y_test, y_pred_naive))

# Apply ICA with specified number of components
n_components = 500 # ica parameter
X_train_ica, ica = apply_ica(X_train, n_components=n_components)

# Fit the Linear Regression model on the ICA-transformed training dataset
lr.fit(X_train_ica, y_train)

# Transform the test dataset using the previously fitted ICA object
X_test_ica, _ = apply_ica(X_test, n_components=n_components, ica=ica)

y_pred_ica = lr.predict(X_test_ica)

# Calculate the mean squared error after ICA
mse_ica = mean_squared_error(y_test, y_pred_ica)

# Print the MSE before and after applying PCA and ICA
print("MSE before applying PCA or ICA (naive model): ", mse_naive)
print("MSE after applying PCA: ", mse_pca)
print("MSE after applying ICA: ", mse_ica)

