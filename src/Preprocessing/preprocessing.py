import pandas as pd

def remove_constant_columns(X):
    """
    remove constant columns from the given X
    """

    constant_cols = X.loc[:,X.apply(pd.Series.nunique) == 1].columns.to_list()

    X.drop(columns=constant_cols, inplace=True)

    return X

def remove_nan_columns(X):
    """
    remove columns that contain only nan values from the given df
    """

    cols_with_nan = [col for col in X.columns if X[col].isna().any() > 0]

    for col in cols_with_nan:
        if X[col].isna().sum() / X.shape[0] == 1:
            X.drop(columns=col, inplace=True)

    # todo: ask Gil if he likes it more like the above or below
    # or in one line:
    # [X.drop(columns=col, inplace=True) for col in cols_with_nan if X[col].isna().sum() / X.shape[0] == 1]

    return X