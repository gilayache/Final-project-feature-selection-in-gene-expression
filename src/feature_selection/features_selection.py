import numpy as np
import pandas as pd
from mrmr import mrmr_classif, mrmr_regression
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LogisticRegression
from typing import List
import time
from tqdm import tqdm

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import lightgbm as lgb

start_time = time.time()
class FeaturesSelection:
    """
    This class is responsible for feature selection. Each method returns the selected features
    """


    def __init__(self, fs_method_1: str,fs_method_2:str, model_type: str, K: int = 5, random_state: int = 42, alpha: float = 0.01,
                 l1_ratio: float = 0.5, C: float = 0.01, n_features_to_select: int = 10):
        """
        :param fs_method_1: the first method of the feature selection
        :param fs_method_1: the second method of the feature selection
        :param model_type: `regression` or `classification`
        :param K: num of cols to choose for mrmr.
        :param random_state.
        :param alpha: is a regularization parameter that controls the strength of the penalty applied to the coefficients for elastic net.
        :param l1_ratio: controls the mix of L1 and L2 penalties in the Elastic Net regularization.
        :param C: Is the inverse of regularization strength for the classification problem for elastic net.
        :param List of the names of the selected features
        :param n_features_to_select: Number of features to select (for rfe).

        """
        self.fs_method_1 = fs_method_1
        self.fs_method_2 = fs_method_2
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
            print("Running the mrr: ")
            selected_features = mrmr_classif(X=X, y=y, K=self.K)

        elif self.model_type == "regression":
            print("Running the mrr: ")
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

    def rfe(self, X: pd.DataFrame, y: pd.Series) -> List:
        """
        Recursive Feature Elimination (RFE) selects features by recursively eliminating features based on their importance or coefficients.
        :param X: X - should be completely numerical
        :param y: the target col
        :return: List of selected features
        """

        if self.model_type == "regression":
            estimator = lgb.LGBMRegressor(random_state=self.random_state)
        elif self.model_type == "classification":
            estimator = lgb.LGBMClassifier(random_state=self.random_state)

        selector = RFE(estimator=estimator, n_features_to_select=self.n_features_to_select)

        selected_features = []

        # Iterate over the RFE steps using tqdm
        with tqdm(total=X.shape[1] - self.n_features_to_select, desc="RFE Progress") as pbar:
            for _ in range(X.shape[1] - self.n_features_to_select):
                selector.fit(X, y)
                feature_ranks = selector.ranking_
                worst_feature = np.argmax(feature_ranks)
                selected_features.append(X.columns[worst_feature])
                X = X.drop(columns=X.columns[worst_feature])
                pbar.update()

        selected_features.reverse()  # Reverse the list to get the best features

        print(f"The final number of features is: {len(selected_features)}")

        return selected_features

    def fit(self, X:pd.DataFrame, y:pd.Series):
        """
        Applying the feature selection method and return the all the feature selection parameters
        including the features names.
        """
        if self.fs_method_1 == "mrmr" or self.fs_method_2 == "mrmr":
            self.selected_features = self.mrmr(X, y=y)

        elif self.fs_method_1 == "elastic_net" or self.fs_method_2 == "elastic_net":
            self.selected_features = self.elastic_net(X, y=y)

        elif self.fs_method_1 == "rfe" or self.fs_method_2 == "rfe":
            self.selected_features = self.rfe(X, y=y)

        else:
            print("please provide a valid feature selection method")

        return self

    def transform(self, X:pd.DataFrame):
        """
        Return the filtered dataframe after the feature selection applied
        """

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"The feature selection was done successfully in {elapsed_time:.2f} seconds")
        return X[self.selected_features]
