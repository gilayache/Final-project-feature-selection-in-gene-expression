import pandas as pd

class Preprocessing:
    """
    class for preprocessing methods
    """
    def __init__(self, target_col: str):
        self.target_col = target_col

    def remove_constant_columns(df):
        """
        remove constant columns from the given X
        """

        constant_cols = df.loc[:, df.apply(pd.Series.nunique) == 1].columns.to_list()

        df.drop(columns=constant_cols, inplace=True)

        return df

    def remove_nan_columns(df):
        """
        remove columns that contain only nan values from the given df
        """

        cols_with_nan = [col for col in df.columns if df[col].isna().any() > 0]

        for col in cols_with_nan:
            if df[col].isna().sum() / df.shape[0] == 1:
                df.drop(columns=col, inplace=True)


        return df

    def split_x_y(self, df: pd.DataFrame):
        """
        get a df and a target col and return X,y
        from X we drop both the target col of regression & classification
        :return: X, y
        """
        y = df[self.target_col]
        X = df.drop(columns=['Lympho', 'ER'])

        return X, y
