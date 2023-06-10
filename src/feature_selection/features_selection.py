import numpy as np
import pandas as pd
from mrmr import mrmr_classif, mrmr_regression
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LogisticRegression
from typing import List
import time
from tqdm import tqdm
from sklearn.metrics import f1_score, mean_squared_error

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import lightgbm as lgb

start_time = time.time()

class FeaturesSelection:
    """
    This class is responsible for feature selection. Each method returns the selected features
    """

    def __init__(self, fs_method_1: str,fs_method_2:str, model_type: str, K: int, random_state: int, alpha: float,
                 l1_ratio: float , C: float , n_features_to_select:int):
        """
        :param fs_method_1: the first method of the feature selection
        :param fs_method_2: the second method of the feature selection
        :param model_type: `regression` or `classification`
        :param K: num of cols to choose for mrmr.
        :param random_state.
        :param alpha: is a regularization parameter that controls the strength of the penalty applied to the coefficients for elastic net.
        :param l1_ratio: controls the mix of L1 and L2 penalties in the Elastic Net regularization.
        :param C: Is the inverse of regularization strength for the classification problem for elastic net.
        :param List of the names of the selected features
        :param n_features_to_select: Number of features to select (for rfe, forward or backward).

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

        selected_features_to_remove = []

        # Iterate over the RFE steps using tqdm
        with tqdm(total=X.shape[1] - self.n_features_to_select, desc="RFE Progress") as pbar:
            for _ in range(X.shape[1] - self.n_features_to_select):
                selector.fit(X, y)
                feature_ranks = selector.ranking_
                worst_feature = np.argmax(feature_ranks)
                selected_features_to_remove.append(X.columns[worst_feature])
                X = X.drop(columns=X.columns[worst_feature])
                pbar.update()

        selected_features = X.columns[~X.columns.isin(selected_features_to_remove)].tolist()

        return selected_features

    def forward_selection(self, X: pd.DataFrame, y: pd.Series) -> List:
        """
        Forward feature selection selects features by iteratively adding the most relevant feature to the feature set.
        :param X: X - should be completely numerical
        :param y: the target col
        :param n_features_to_select: the maximum number of features to select
        :return: List of selected features
        """

        selected_features = []
        remaining_features = set(X.columns)

        # Add tqdm progress bar for feature selection
        with tqdm(total=min(X.shape[1], self.n_features_to_select), desc="Forward Selection Progress") as pbar:
            # Iterate until all features are selected or we've reached the n_features_to_select limit
            while remaining_features and len(selected_features) < self.n_features_to_select:
                best_feature = None
                best_score = np.inf


                # Iterate over remaining features and select the one with the best score
                for feature in remaining_features:
                    feature_set = selected_features + [feature]
                    X_subset = X[feature_set]

                    if self.model_type == "regression":
                        model = lgb.LGBMRegressor(random_state=self.random_state)
                        model.fit(X_subset, y)
                        score = mean_squared_error(y, model.predict(X_subset))
                    elif self.model_type == "classification":
                        model = lgb.LGBMClassifier(random_state=self.random_state)
                        model.fit(X_subset, y)
                        score = f1_score(y, model.predict(X_subset))

                    if score < best_score:
                        best_score = score
                        best_feature = feature

                selected_features.append(best_feature)
                remaining_features.remove(best_feature)
                pbar.update(1)  # Update progress bar

        return selected_features

    def backward_selection(self, X: pd.DataFrame, y: pd.Series) -> List:
        """
        Backward feature selection selects features by iteratively removing the least relevant feature from the feature set.
        :param X: X - should be completely numerical
        :param y: the target col
        :param n_features_to_select: the maximum number of features to keep
        :return: List of selected features
        """
        selected_features = list(X.columns)

        # Add tqdm progress bar for feature selection
        with tqdm(total=max(0, X.shape[1] - self.n_features_to_select), desc="Backward Selection Progress") as pbar:
            # Iterate until we reach the n_features_to_select limit
            while len(selected_features) > self.n_features_to_select:
                scores = []

                for feature in selected_features:
                    feature_set = selected_features.copy()
                    feature_set.remove(feature)
                    X_subset = X[feature_set]

                    if self.model_type == "regression":
                        model = lgb.LGBMRegressor(random_state=self.random_state)
                        model.fit(X_subset, y)
                        score = mean_squared_error(y, model.predict(X_subset))
                    elif self.model_type == "classification":
                        model = lgb.LGBMClassifier(random_state=self.random_state)
                        model.fit(X_subset, y)
                        score = f1_score(y, model.predict(X_subset))

                    scores.append(score)

                worst_feature = selected_features[np.argmax(scores)]

                selected_features.remove(worst_feature)
                pbar.update(1)  # Update progress bar

        return selected_features

    def _apply_fs_method(self, X, y, method):
        if method == "mrmr":
            selected_features = self.mrmr(X, y=y)

        elif method == "elastic_net":
            selected_features = self.elastic_net(X, y=y)

        elif method == "rfe":
            selected_features = self.rfe(X, y=y)

        elif method == "forward_selection":
            selected_features = self.forward_selection(X, y=y)

        elif method == "backward_selection":
            selected_features = self.backward_selection(X, y=y)

        else:
            print("please provide a valid feature selection method")
            return []

        return selected_features

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Applying the feature selection method and return all the feature selection parameters
        including the features names.
        """
        self.selected_features_1 = None
        self.selected_features_2 = None

        print(f"Size of X before feature selection: {X.shape}")

        if self.fs_method_1:
            self.selected_features_1 = self._apply_fs_method(X, y, self.fs_method_1)
            print(f'Number of features after {self.fs_method_1}: {len(self.selected_features_1)}')
            X = X[self.selected_features_1]  # filter the data with the selected features
            print(f'Shape of X after {self.fs_method_1}: {X.shape}')

        if self.fs_method_2:
            self.selected_features_2 = self._apply_fs_method(X, y, self.fs_method_2)
            print(f'Number of features after {self.fs_method_2}: {len(self.selected_features_2)}')
            X = X[self.selected_features_2]  # filter the data with the selected features
            print(f'Shape of X after {self.fs_method_2}: {X.shape}')

        self.final_selected_features = X.columns.tolist()

        return self

    def transform(self, X: pd.DataFrame):
        """
        Return the filtered dataframe after the feature selection applied
        """
        # Filter the DataFrame based on the selected features from each method, if they are defined
        if self.selected_features_1 is not None:
            X = X[self.selected_features_1]

        if self.selected_features_2 is not None:
            X = X[self.selected_features_2]

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"The feature selection was done successfully in {elapsed_time:.2f} seconds")

        return X





