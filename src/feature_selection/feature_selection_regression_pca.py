import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the dataset
data = pd.read_csv("../../data/processed/merged_dataset.csv")
X = data.iloc[:, 1:-6]
cols_with_nan = X.columns[X.isna().any()].tolist()
[
    X.drop(columns=col, inplace=True)
    for col in cols_with_nan
    if X[col].isna().sum() / X.shape[0] == 1
]
X = X.fillna(0)
y = data["Lympho"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)

# Initialize the Linear Regression model
lr = LinearRegression()

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply PCA with specified explained variance
pca = PCA(n_components=0.99)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Plot the scree plot
plt.plot(np.arange(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Scree Plot")
plt.show()

# Fit the Linear Regression model on the PCA-transformed training dataset
lr.fit(X_train_pca, y_train)

# Make predictions using the test dataset
y_pred_pca = lr.predict(X_test_pca)

# Calculate the mean squared error after PCA
mse_pca = mean_squared_error(y_test, y_pred_pca)

# Fit the Linear Regression model directly on the training dataset without PCA
lr.fit(X_train_scaled, y_train)

# Make predictions using the test dataset without PCA
y_pred_no_transform = lr.predict(X_test_scaled)

# Calculate the mean squared error before applying PCA using the Linear Regression model
mse_no_transform = mean_squared_error(y_test, y_pred_no_transform)

# Number of features before and after PCA
num_features_before = X_train.shape[1]
num_features_after_pca = X_train_pca.shape[1]

# Print the number of features before and after PCA
print("Number of features before PCA: ", num_features_before)
print("Number of features after PCA: ", num_features_after_pca)

# Print the MSE before and after applying PCA
print("MSE before applying PCA: ", round(mse_no_transform, 4))
print("MSE after applying PCA: ", round(mse_pca, 4))
