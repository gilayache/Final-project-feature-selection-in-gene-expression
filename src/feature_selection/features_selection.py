import numpy as np
import pandas as pd
from mrmr import mrmr_classif, mrmr_regression
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LogisticRegression
from typing import List
import time
from tqdm import tqdm

from sklearn.feature_selection import RFE
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


start_time = time.time()
class FeaturesSelection:
    """
    This class is responsible for feature selection. Each method returns the selected features
    """


    def __init__(self, fs_method: str, model_type: str, K: int = 5, random_state: int = 42, alpha: float = 0.01,
                 l1_ratio: float = 0.5, C: float = 0.01, n_features_to_select: int = 10):
        """
        :param fs_method: the method of the feature selection (elastic_net or mrr)
        :param model_type: `regression` or `classification`
        :param K: num of cols to choose for mrmr.
        :param random_state.
        :param alpha: is a regularization parameter that controls the strength of the penalty applied to the coefficients for elastic net.
        :param l1_ratio: controls the mix of L1 and L2 penalties in the Elastic Net regularization.
        :param C: Is the inverse of regularization strength for the classification problem for elastic net.
        :param List of the names of the selected features
        :param n_features_to_select: Number of features to select (for rfe).

        """
        self.fs_method = fs_method
        self.model_type = model_type
        self.K = K
        self.random_state = random_state
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.C = C
        self.n_features_to_select = n_features_to_select
        self.selected_features = []

    def mrmr(self, X: pd.DataFrame, y: pd.Series) -> List:
        """
        MRMR (Maximum Relevance Minimum Redundancy) selects informative and non-redundant features by ranking them according
         to their relevance to the target variable while minimizing redundancy between the selected features.
        :param X: X - should be completely numerical
        :param y: the target col
        :return: List of selected features (the top K features)
        """

        if self.model_type == "classification":
            selected_features = mrmr_classif(X=X, y=y, K=self.K)

        elif self.model_type == "regression":
            selected_features = mrmr_regression(X=X, y=y, K=self.K)

        return selected_features

    def elastic_net(self, X: pd.DataFrame, y: pd.Series) -> List:
        """
        Elastic Net is a linear regression model with both L1 and L2 regularization,
        which combines the strengths of Ridge and Lasso regularization to balance feature selection and model complexity.
        :param X: X - should be completely numerical
        :param y: the target col
        :return: List of selected features
        """

        # Instantiate Elastic Net model based on the model_type
        if self.model_type == "regression":
            model = ElasticNet(
                alpha=self.alpha, l1_ratio=self.l1_ratio, random_state=self.random_state
            )
        elif self.model_type == "classification":
            # The 'saga' solver is required for Elastic Net regularization in logistic regression
            model = LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                l1_ratio=self.l1_ratio,
                C=self.C,
                random_state=self.random_state,
            )

        # Fit the model
        model.fit(X, y)

        # Get the indices of the best features
        # because they have impact on the model's prediction
        best_feature_indices = np.where(model.coef_ != 0)[0]

        # Get the names of the best features
        selected_features = X.columns[best_feature_indices]
        return selected_features

    from tqdm import tqdm

    from tqdm import tqdm

    def rfe(self, X: pd.DataFrame, y: pd.Series) -> List:
        """
        Recursive Feature Elimination (RFE) selects features by recursively eliminating features based on their importance or coefficients.
        :param X: X - should be completely numerical
        :param y: the target col
        :return: List of selected features
        """

        if self.model_type == "regression":
            estimator = RandomForestRegressor(random_state=self.random_state)
        elif self.model_type == "classification":
            estimator = RandomForestClassifier(random_state=self.random_state)

        selector = RFE(estimator=estimator, n_features_to_select=self.n_features_to_select)

        selected_features = []
        n_iterations = X.shape[1] - self.n_features_to_select + 1

        # Iterate over the RFE steps using tqdm
        with tqdm(total=n_iterations, desc="RFE Progress") as pbar:
            for _ in range(n_iterations):
                selector.fit(X, y)
                selected_features.append(X.columns[selector.support_])
                X = X.loc[:, selector.support_]
                pbar.update()

        return selected_features

    def fit(self, X:pd.DataFrame, y:pd.Series):
        """
        Applying the feature selection method and return the all the feature selection parameters
        including the features names.
        """
        if self.fs_method == "mrmr":
            self.selected_features = self.mrmr(X, y=y)

        elif self.fs_method == "elastic_net":
            self.selected_features = self.elastic_net(X, y=y)

        elif self.fs_method == "rfe":
            self.selected_features = self.rfe(X, y=y)

        return self

    def transform(self, X:pd.DataFrame):
        """
        Return the filtered dataframe after the feature selection applied
        """

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"The feature selection was done successfully in {elapsed_time:.2f} seconds")
        # print(X[self.selected_features])
        return X[self.selected_features]
