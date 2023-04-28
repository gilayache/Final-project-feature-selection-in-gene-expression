import numpy as np
import pandas as pd
from mrmr import mrmr_classif, mrmr_regression
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error


class FeatureSelection:
    """
    This class is responsible for feature selection. Each method returns the selected features
    """

    def __init__(self):
        pass

    def mrmr(self, X: pd.DataFrame, y: pd.Series, model_type: str, K: int = 10):
        """
        MRMR (Maximum Relevance Minimum Redundancy) selects informative and non-redundant features by ranking them according
         to their relevance to the target variable while minimizing redundancy between the selected features.
        :param X: X - should be completely numerical
        :param y: the target col
        :param model_type:
        :param K: num of cols to choose
        :return: selected features (the top K features)
        """

        if model_type == 'classification':
            selected_features = mrmr_classif(X=X, y=y, K=K)

        elif model_type == 'regression':
            selected_features = mrmr_regression(X=X, y=y, K=K)

        return selected_features


    def elastic_net(self, X_train, X_test, y_train, y_test):
        """
        # todo: make sure this is working and update the documentation
        Elastic Net is a combination of L1 and L2 regularization.
        """
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
        best_feature_names = X_train.columns[best_feature_indices]

        return best_feature_names
