from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, classification_report
import pandas as pd

import src.Preprocessing.imputation as imputation
import src.Preprocessing.preprocesing as preprocesing
import src.Preprocessing.scaling as scaling
import src.Preprocessing.encoding as encoding
import src.models.modeling as modeling
import src.feature_selection.features_selection as features_selection


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
        # todo the below as well ?
        # preprocesing.Preprocessing.remove_low_variance_columns(self)
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
                    ('Features Selection 1', features_selection.FeaturesSelection(fs_method_1=self.fs_method_1,
                                                                                  fs_method_2=None,
                                                                                  model_type=self.model_type,
                                                                                  K=self.K,
                                                                                  random_state=self.seed)),
                    ('Features Selection 2', features_selection.FeaturesSelection(fs_method_2=self.fs_method_2,
                                                                                  fs_method_1= None,
                                                                                  model_type=self.model_type,
                                                                                  random_state=self.seed,
                                                                                  n_features_to_select=self.n_features_to_select)),
                    ('Modeling', modeling.Model(model_name=self.model_name))])

        # todo: Save the model
        # since current we are not using the validation set
        pipe.fit(X_train_val, y_train_val)
        y_test_pred = pipe.predict(X_test)

        # evaluation
        if self.model_type == 'regression':
            mse_test = mean_squared_error(y_test, y_test_pred)
            print("Test Mean Squared Error:", mse_test)
        elif self.model_type == 'classification':
            classification_rep_test = classification_report(y_test, y_test_pred)
            print("Test Classification Report:\n", classification_rep_test)
        else:
            print("Please make sure that the model_type is regression or classification")


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
