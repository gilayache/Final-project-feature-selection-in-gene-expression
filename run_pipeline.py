import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.Preprocessing import preprocesing
from src.feature_selection import feature_selection
from sklearn.pipeline import FeatureUnion, Pipeline

class run_pipeline:
    """

    """
    def __init__(self, run_type: str, target_col: str, input_path: str, output_path: str,
                 model_type: str):
        """
        init the class
        :param run_type: 'train' or 'inference'
        :param model_type: 'classification' or 'regression'
        :param target_col: the target col name (Lympho or ER)
        :param input_path: the path to the input data
        :param output_path: the path to the output data & results
        """
        self.run_type = run_type
        self.model_type = model_type
        self.target_col = target_col
        self.input_path = input_path or 'data/processed/merged_dataset.csv'
        self.output_path = output_path
        self.df = pd.DataFrame()

    def run(self):
        """

        """

        data = self.load_data()
        X, y = preprocesing.Preprocessing.split_x_y(self, data)

        X = self.run_preprocessing_steps(X,y)

        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if self.run_type == 'train':
            self.run_train()

        elif self.run_type == 'inference':
            self.run_inference(

            )
        # todo: make the below work. if not working than change to run_preprocessing_steps and run_train & run_inference?
        # in order the below work we need to to call only classes with fit & transform implemented inside
        # pipeline = Pipeline(steps=[
        #         ('remove_constant_columns', preprocesing.Preprocessing.remove_constant_columns(self)),
        #         ('remove_nan_columns', preprocesing.Preprocessing.remove_nan_columns(self)),
        #         ('select_features', feature_selection.FeatureSelection.mrmr(self))])
        #
        # pipeline.fit(X_train, y_train)


    def load_data(self):
        """
        load the data
        """
        data = pd.read_csv(self.input_path)

        return data

    def run_preprocessing_steps(self, X: pd.DataFrame, y: pd.Series):
        """
        run the preprocessing steps
        """

        _X = preprocesing.Preprocessing.remove_constant_columns(df=X)
        _X = preprocesing.Preprocessing.remove_nan_columns(_X)

        return _X


    def run_train(self):
        """
        run the train steps
        """
        pass

    def run_inference(self):
        """
        run the inference steps
        """
        pass

if __name__ == '__main__':

    run_pipeline = run_pipeline(model_type='classification', target_col='Lympho',
                                input_path='data/processed/merged_dataset.csv', output_path=None,
                                run_type='train')
    run_pipeline.run()

# data = pd.read_csv('data/processed/merged_dataset.csv')
# # split x_y
# X, y = preprocesing.split_x_y(data, 'Lympho')
# X_train, X_test, y_testtrain, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# todo: create eval_results methods

# todo: we can also consider use Dimensional Reduction methods and then use regular feature selection and check which
#  method gives us better results (and anyway that might be interesting to see)

# pipeline = Pipeline(steps=[
#         ('remove_constant_columns', preprocessing.remove_constant_columns()),
#         ('remove_nan_columns', preprocessing.remove_nan_columns())
#         ('select_features', feature_selection_classification.mrmr())])

# pipeline.fit(X_train, y_train)

