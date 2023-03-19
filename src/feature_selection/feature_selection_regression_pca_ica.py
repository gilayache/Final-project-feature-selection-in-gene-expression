import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FastICA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

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

# Get the initial number of features
initial_num_features = X.shape[1]

# Splitting the data into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)

def apply_pca(X, pca=None, plot_scree=True):
    if pca is None:
        pca = PCA()
        X_pca = pca.fit_transform(X)

        if plot_scree:
            plt.plot(np.arange(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_))
            plt.xlabel("Number of Components")
            plt.ylabel("Cumulative Explained Variance")
            plt.title("Scree Plot")
            plt.show()
    else:
        X_pca = pca.transform(X)

    return X_pca

def apply_ica(X, ica=None):
    if ica is None:
        ica = FastICA()
        X_ica = ica.fit_transform(X)
    else:
        X_ica = ica.transform(X)
    return X_ica



def apply_ica_pca(X, ica=None, pca=None):
    X_ica, ica = apply_ica(X, ica=ica)
    X_ica_pca, pca = apply_pca(X_ica, pca=pca, plot_scree=False)
    return X_ica_pca, ica, pca

def apply_pca_ica(X, pca=None, ica=None):
    X_pca, pca = apply_pca(X, pca=pca, plot_scree=False)
    X_pca_ica, ica = apply_ica(X_pca, ica=ica)
    return X_pca_ica, pca, ica


# Apply PCA only
X_pca, pca = apply_pca(X_train)
# Apply ICA only
X_ica, ica = apply_ica(X_train)
# Apply ICA followed by PCA
X_ica_pca, ica_pca_ica, pca_ica_pca = apply_ica_pca(X_train)
# Apply PCA followed by ICA
X_pca_ica, pca_pca_ica, ica_pca_ica = apply_pca_ica(X_train)


# Initialize the Linear Regression model
lr = LinearRegression()

X_test_pca = apply_pca(X_test, pca=pca, plot_scree=False)
y_pred_pca = lr.predict(X_test_pca)
mse_pca = mean_squared_error(y_test, y_pred_pca)

lr.fit(X_ica, y_train)
X_test_ica = apply_ica(X_test, ica=ica)
y_pred_ica = lr.predict(X_test_ica)
mse_ica = mean_squared_error(y_test, y_pred_ica)

# Fit and predict for ICA followed by PCA
lr.fit(X_ica_pca, y_train)
X_test_ica_pca, _ = apply_ica_pca(X_test, ica=ica_pca_ica, pca=pca_ica_pca)
y_pred_ica_pca = lr.predict(X_test_ica_pca)
mse_ica_pca = mean_squared_error(y_test, y_pred_ica_pca)

# Fit and predict for PCA followed by ICA
lr.fit(X_pca_ica, y_train)
X_test_pca_ica, _ = apply_pca_ica




# Plot the number of features before and after reduction
methods = ["Initial", "PCA", "ICA", "ICA-PCA", "PCA-ICA"]
num_features = [
    initial_num_features,
    X_pca.shape[1],
    X_ica.shape[1],
    X_ica_pca.shape[1],
    X_pca_ica.shape[1],
]

plt.bar(methods, num_features)
plt.xlabel("Method")
plt.ylabel("Number of Features")
plt.title("Number of Features Before and After Reduction")
plt.show()

# Print the MSE for each method
print("MSE for PCA only: ", mse_pca)
print("MSE for ICA only: ", mse_ica)
print("MSE for ICA followed by PCA: ", mse_ica_pca)
print("MSE for PCA followed by ICA: ", mse_pca_ica)

