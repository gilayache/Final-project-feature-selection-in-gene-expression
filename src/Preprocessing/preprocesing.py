import pandas as pd

class Preprocessing:
    """
    class for preprocessing methods
    """
    def __init__(self, df: pd.DataFrame, target_col: str):
        self.df = df
        self.target_col = target_col

    def remove_constant_columns(self):
        """
        remove constant columns from the given X
        """

        constant_cols = self.df.loc[:, self.df.apply(pd.Series.nunique) == 1].columns.to_list()

        self.df.drop(columns=constant_cols, inplace=True)

        return self.df

    def remove_nan_columns(self):
        """
        remove columns that contain only nan values from the given df
        """

        cols_with_nan = [col for col in self.df.columns if self.df[col].isna().any() > 0]

        for col in cols_with_nan:
            if self.df[col].isna().sum() / self.df.shape[0] == 1:
                self.df.drop(columns=col, inplace=True)

    def split_x_y(self, df: pd.DataFrame):
        """
        get a df and a target col and return X,y
        from X we drop both the target col of regression & classification
        :return: X, y
        """
        y = df[self.target_col]
        X = df.drop(columns=['Lympho', 'ER'])

        return X, y
