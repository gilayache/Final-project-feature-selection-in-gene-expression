from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, classification_report
import pandas as pd

import src.Preprocessing.imputation as imputation
import src.Preprocessing.preprocesing as preprocesing
import src.Preprocessing.scaling as scaling
import src.Preprocessing.encoding as encoding
import src.models.modeling as modeling
import src.feature_selection.features_selection as features_selection
from src.evaluation import evaluate

import src.evaluation.evaluate as evaluation

from src import Utils
from sklearn.model_selection import train_test_split

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

        X = preprocesing.Preprocessing.remove_constant_columns(self, X)

        # Split the data into train+validation and test sets
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=self.test_size,
                                                                    random_state=self.seed)

        # Further split train+validation into separate train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=self.val_size,
                                                          random_state=self.seed)

        y_train, y_val, y_test = y_train.values.reshape(-1, 1), y_val.values.reshape(-1, 1), y_test.values.reshape(-1,
                                                                                                                   1)

        self.features = X_train.columns.to_list()
        self.numerical_features = X_train.select_dtypes("number").columns
        self.categorical_features = X_train.select_dtypes("object").columns

        pipe = Pipeline(steps=[
                    ('Imputation', imputation.Imputer(categorical_features=self.categorical_features,
                                                numerical_features=self.numerical_features)),
                    ('Encoding', encoding.Encoder(encoder_name=self.encoder_name, features=self.features)),
                    ('Scaling', scaling.Scaler(scaler_name=self.scaler_name)),
                    ('Features Selection 1', features_selection.FeaturesSelection(**self.fs_params)),
                    ('Features Selection 2', features_selection.FeaturesSelection(**self.fs_params)),
                    ('Modeling', modeling.Model(model_name=self.model_name, val_size=self.val_size, seed=self.seed,
                                                hyper_params_dict=self.hyper_params_dict,
                                                n_jobs=self.n_jobs,
                                                fit_intercept=self.fit_intercept
                                                ))])

        # since current we are not using the validation set
        pipe.fit(X_train, y_train)
        y_test_pred = pipe.predict(X_test)
        # evaluate and write the results to csv
        evaluator = evaluation.Evaluation(self.model_type, y_test, y_test_pred, params)
        evaluator.evaluate()

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
