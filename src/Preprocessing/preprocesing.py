import pandas as pd

class Preprocessing:
    """
    class for preprocessing methods
    """
    def __init__(self, run_type: str, preprocessing_operations: list, df: pd.DataFrame):
        self.run_type = run_type
        self.list_of_methods = ['remove_constant_columns', 'remove_nan_columns', 'create_x_y']
        self.constant_cols = []
        self.cols_with_nan = []
        self.features = []
        self.numeric_features = []
        self.categorical_features = []
        self.preprocessing_operations = preprocessing_operations
        self.df = df

    def remove_constant_columns(self, X):
        """
        remove constant columns from the given X
        """
        if self.run_type == 'train':
            self.constant_cols = X.loc[:, X.apply(pd.Series.nunique) == 1].columns.to_list()

        X.drop(columns=self.constant_cols, inplace=True)

        return X


    def remove_nan_columns(self, X):
        """
        remove columns that contain only nan values from the given df
        """
        if self.run_type == 'train':
            self.cols_with_nan = [col for col in X.columns if X[col].isna().any() > 0]

        for col in self.cols_with_nan:
            if X[col].isna().sum() / X.shape[0] == 1:
                X.drop(columns=col, inplace=True)

        return X

    def create_x_y(self):
        """

        """
        # todo: we failed because 'LumA_target' was not in self.df --> i removed it from the params - ask gil about it
        _orig_features = self.df.columns.drop(self.columns_to_remove).to_list()

        X = self.df[_orig_features]
        y = self.df[self.target_col]

        return X, y