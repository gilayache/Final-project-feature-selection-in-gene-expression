import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('../../data/processed/merged_dataset.csv')

# Preprocessing the data
X = data.iloc[:, 1:-6] # Dropping irrelevant columns (keeping only the gene expression data)
X = X.fillna(0)

y = data['Lympho'] # The target variable

# Split the dataset into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Fit the Elastic Net model with different alpha values
alphas = np.logspace(-3, 3, num=7)
num_features = []
mse_scores = []

for alpha in alphas:
    enet = ElasticNet(alpha=alpha, l1_ratio=0.5, random_state=123)
    enet.fit(X_train, y_train)
    num_nonzero = np.sum(enet.coef_ != 0)
    num_features.append(num_nonzero)
    y_pred = enet.predict(X_test)
    mse_scores.append(mean_squared_error(y_test, y_pred))

# Plot the number of features and MSE scores for different alpha values
fig, ax1 = plt.subplots()
ax1.plot(alphas, num_features, 'b-')
ax1.set_xlabel('alpha')
ax1.set_ylabel('Number of features', color='b')
ax1.tick_params('y', colors='b')
ax1.set_title('Elastic Net - Number of Features and MSE vs Alpha')

ax2 = ax1.twinx()
ax2.plot(alphas, mse_scores, 'r-')
ax2.set_ylabel('MSE', color='r')
ax2.tick_params('y', colors='r')

# Annotate the number of features for each point
for i, num_feat in enumerate(num_features):
    ax1.annotate(f"{num_feat}", xy=(alphas[i], num_feat), xytext=(alphas[i], num_feat+100), ha='center', va='bottom', fontsize=8)

fig.tight_layout()
plt.xscale('log')
plt.show()
