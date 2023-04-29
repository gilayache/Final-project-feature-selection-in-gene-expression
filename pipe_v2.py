import pandas as pd
from sklearn import set_config
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

import imputation, preprocesing, scaling, encoding, modeling, features_selection
from src import Utils
from sklearn.model_selection import train_test_split

pd.options.display.precision = 4
pd.options.mode.chained_assignment = None
set_config(display="diagram")

class RunPipeline:

    def __init__(self, input_data_path, params_path):
        self.input_path = input_data_path or 'data/processed/merged_dataset.csv'
        self.df = pd.DataFrame()
        self.params_path = params_path


    def run(self):
        """
        Runs the pipeline
        """
        self.df, params = self.load_data_and_params()
        X, y = preprocesing.Preprocessing.create_x_y(self)
        X = preprocesing.Preprocessing.remove_nan_columns(self, X)
        # todo the below as well ?
        # preprocesing.Preprocessing.remove_low_variance_columns(self)
        X = preprocesing.Preprocessing.remove_constant_columns(self, X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.seed)

        self.features = X_train.columns.to_list()
        self.numerical_features = X_train.select_dtypes("number").columns
        self.categorical_features = X_train.select_dtypes("object").columns

        pipe = Pipeline(steps=[
                    ('Imputation', imputation.Imputer(categorical_features=self.categorical_features,
                                                numerical_features=self.numerical_features)),
                    ('Encoding', encoding.Encoder(encoder_name=self.encoder_name, features=self.features)),
                    ('Scaling', scaling.Scaler(scaler_name=self.scaler_name)),
                    ('Features Selection', features_selection.FeaturesSelection(fs_method=self.fs_method,
                                        model_type=self.model_type, K=self.K, random_state=self.seed)),
                    ('Modeling', modeling.Model(self.model_name))])

        # todo: create class for the below
        # Evaluation
        # Save the model

        pipe.fit(X_train, y_train)

        y_test_pred = pipe.predict(X_test)

        mse = mean_squared_error(y_test, y_test_pred)
        print("Mean Squared Error:", mse)

    def load_data_and_params(self):

        df = Utils.load_data(self.input_path)
        params = Utils.load_params(self.params_path)

        for key, value in params.items():
            setattr(self, key, value)

        return df, params



if __name__ == '__main__':
    run_pipeline = RunPipeline(params_path='src/data/params.yaml',
                               input_data_path='data/processed/merged_dataset.csv'
    )
    run_pipeline.run()