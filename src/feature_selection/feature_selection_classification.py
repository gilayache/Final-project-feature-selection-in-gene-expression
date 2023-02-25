import pandas as pd
from mrmr import mrmr_classif, mrmr_regression


def mrmr(X: pd.DataFrame,y: pd.Series, model_type: str, K: int = 10):
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

