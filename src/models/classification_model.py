import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

#Load the dataset
data = pd.read_csv('../../data/processed/merged_dataset.csv')
# Preprocessing the data
# need to change this one.#
X = data.iloc[:, 1:-6]# Dropping irrelevant columns (keeping only the gene expression data)
# X = X.fillna(X.mean()) # Handling missing values
# X = pd.get_dummies(X) # Encoding categorical variables
# need to change this
X = X.fillna(0)

y = data['ER'] # The target variable

# Split the dataset into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Define the logistic regression model and fit it on the training dataset
lr = LogisticRegression(max_iter=5000)
lr.fit(X_train, y_train)

# Use the trained model to predict the labels of the testing dataset
y_pred = lr.predict(X_test)

# Evaluate the performance of the model using various metrics
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:\n", cm)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))
