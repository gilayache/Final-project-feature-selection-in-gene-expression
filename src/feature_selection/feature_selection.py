import numpy as np
import pandas as pd
from mrmr import mrmr_classif, mrmr_regression
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LogisticRegression
from typing import List

class FeaturesSelection:
    """
    This class is responsible for feature selection. Each method returns the selected features
    """

    def __init__(self, fs_method: str, model_type: str, K: int = 5, random_state: int = 42, alpha: float = 0.01,
                 l1_ratio: float = 0.5, C: float = 0.01):
        """
        :param fs_method: the method of the feature selection (elastic_net or mrr)
        :param model_type: `regression` or `classification`
        :param K: num of cols to choose for mrmr.
        :param random_state.
        :param alpha: is a regularization parameter that controls the strength of the penalty applied to the coefficients for elastic net.
        :param l1_ratio: controls the mix of L1 and L2 penalties in the Elastic Net regularization.
        :param C: Is the inverse of regularization strength for the classification problem for elastic net.
        :param List of the names of the selected features
        """
        self.fs_method = fs_method
        self.model_type = model_type
        self.K = K
        self.random_state = random_state
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.C = C
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

    def fit(self, X:pd.DataFrame, y:pd.Series):
        """
        Applying the feature selection method and return the all the feature selection parameters
        including the features names.
        """
        if self.fs_method == "mrmr":
            self.selected_features = self.mrmr(X, y=y)

        elif self.fs_method == "elastic_net":
            self.selected_features = self.elastic_net(X, y=y)
        return self

    def transform(self, X:pd.DataFrame):
        """
        Return the filtered dataframe after the feature selection applied
        """
        return X[self.selected_features]


########## for debugging only #########


# df = pd.read_csv(
#     "/Users/gilayache/PycharmProjects/Final-project-feature-selection-in-gene-expression/data/processed/merged_dataset.csv"
# )
# columns_to_remove = ["ER", "samplename"]
# FEATURES = df.columns.drop(columns_to_remove)
# #
# from sklearn.preprocessing import LabelEncoder
#
# # encode the categorical features
# cat_columns = df.select_dtypes(include=["object"]).columns
# for column in cat_columns:
#     label_encoder = LabelEncoder()
#     df[column] = label_encoder.fit_transform(df[column])
# #
# #
# X = df[FEATURES]
# y = df["ER"]
#
# cols_with_nan = X.columns[X.isna().any()].tolist()
# [
#     X.drop(columns=col, inplace=True)
#     for col in cols_with_nan
#     if X[col].isna().sum() / X.shape[0] == 1
# ]
#
# X = X.fillna(0)
#
# # mrmr
# fs = FeaturesSelection(
#     fs_method="elastic_net", model_type="classification", K=10, random_state=42, alpha=0.01, l1_ratio=0.5, C=0.01
# )
#
# # Fit the FeaturesSelection object on the data
# fs.fit(X, y)
#
# # Transform the input data (X) using the selected features
# transformed_X = fs.transform(X)
#
# # Print the transformed data
# print(transformed_X)
# print(len(transformed_X))




# Elastic net
# Filling missing values with 0
# Dropping columns with all missing values
# cols_with_nan = X.columns[X.isna().any()].tolist()
# [
#     X.drop(columns=col, inplace=True)
#     for col in cols_with_nan
#     if X[col].isna().sum() / X.shape[0] == 1
# ]
#
# X = X.fillna(0)
#
# fs = FeaturesSelection(
#     fs_method="elastic_net", model_type="classification", K=10, random_state=42, alpha=0.01, l1_ratio=0.5, C=0.01
# )
# # Fit the FeaturesSelection object on the data
# fs.fit(X, y)
#
# # Transform the input data (X) using the selected features
# transformed_X = fs.transform(X)
#
# # Print the transformed data
# print(transformed_X)
# print(len(transformed_X))
