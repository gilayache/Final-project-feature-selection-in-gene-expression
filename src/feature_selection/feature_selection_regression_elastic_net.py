import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from pandas import DataFrame
from sklearn.linear_model import LinearRegression


def elastic_net_regression(data: DataFrame, viz: bool):
    """
    The function
    :param data:
    :param viz:
    :return:
    """
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

        # Stop when the number of features becomes 0
        if num_nonzero == 0:
            break

    # If viz=True, plot the number of features and MSE scores for different alpha values
    if viz:
        fig, ax1 = plt.subplots()
        ax1.plot(alphas[: len(num_features)], num_features, "b-")
        ax1.set_xlabel("alpha")
        ax1.set_ylabel("Number of features", color="b")
        ax1.tick_params("y", colors="b")
        ax1.set_title("Elastic Net - Number of Features and MSE vs Alpha")

        ax2 = ax1.twinx()
        ax2.plot(alphas[: len(mse_scores)], mse_scores, "r-")
        ax2.set_ylabel("MSE", color="r")
        ax2.tick_params("y", colors="r")

        # Annotate the number of features for each point
        for i, num_feat in enumerate(num_features):
            ax1.annotate(
                f"{num_feat}",
                xy=(alphas[i], num_feat),
                xytext=(alphas[i], num_feat + 100),
                ha="center",
                va="bottom",
                fontsize=8,
            )

        fig.tight_layout()
        plt.xscale("log")
        plt.show()

    # Find the alpha value that optimizes the MSE
    best_alpha = alphas[np.argmin(mse_scores)]

    # Fit the Elastic Net model with the best alpha value
    enet = ElasticNet(alpha=best_alpha, l1_ratio=0.5, random_state=123)
    enet.fit(X_train, y_train)

    # Find the number of non-zero coefficients in the model
    num_nonzero = np.sum(enet.coef_ != 0)

    # Calculate the mean squared error (MSE) of the model
    mse = mean_squared_error(y_test, enet.predict(X_test))

    # Get the indices of the best features
    best_feature_indices = np.where(enet.coef_ != 0)[0]

    # Get the names of the best features
    best_feature_names = X.columns[best_feature_indices]

    # Print the starting number of features
    print(f"Number of features before Elastic Net: {X.shape[1]}")

    # Print the number of features after Elastic Net
    print(f"Number of features after Elastic Net: {len(best_feature_names)}")

    # Fit the Linear Regression model (without regularization)
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # Calculate the mean squared error (MSE) of the Linear Regression model
    mse_initial = mean_squared_error(y_test, lr.predict(X_test))

    print(f"MSE before applying Elastic Net : {round(mse_initial,4)}")
    print(f"MSE after applying Elastic Net: {round(mse,4)}")

    # Return the plot (if viz=True), the number of features, and the input data
    if viz:
        return (fig, num_nonzero, X[best_feature_names])
    else:
        return (num_nonzero, X[best_feature_names])


# Load the dataset
data = pd.read_csv("../../data/processed/merged_dataset.csv")
viz = False

# apply the function
if viz == True:
    # Call the function with viz=True and assign the result to a variable
    fig, num_nonzero, X = elastic_net_regression(data, viz=True)
else:
    # Call the function with viz=False and assign the result to a variable
    num_nonzero, X = elastic_net_regression(data, viz=False)

