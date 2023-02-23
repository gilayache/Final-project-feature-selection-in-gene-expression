import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('../../../data/processed/merged_dataset.csv')

# Preprocessing the data
X = data.iloc[:, 1:-6] # Dropping irrelevant columns (keeping only the gene expression data)
X = X.fillna(0)

y = data['Lympho'] # The target variable

# Split the dataset into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Train the linear regression model
reg = LinearRegression()
reg.fit(X_train, y_train)

# Make predictions on the testing dataset
y_pred = reg.predict(X_test)

# Calculate the root mean squared error
rmse = mean_squared_error(y_test, y_pred, squared=False)
print('RMSE:', rmse)
