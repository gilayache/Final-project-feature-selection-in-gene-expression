from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
import src.Preprocessing.preprocesing as preprocesing

from sklearn.metrics import mean_squared_error,f1_score
from tqdm import tqdm
from src import Utils

class RunPipeline:

    def __init__(self, input_data_path, params_path):
        self.input_path = input_data_path or '../../data/processed/merged_dataset.csv'
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

        K_value = params['fs_params']['K']
        max_features_value = params['fs_params']['max_features']
        fs_method_1_value = params['fs_params']['fs_method_1']
        model_type = params['model_type']

        if fs_method_1_value == 'mrmr':
            n = K_value
        elif fs_method_1_value == 'genetic_selection':
            n = max_features_value

        results = []
        for i in tqdm(range(1, n + 1)):
            current_features = self.features[:i]
            df_sub = self.df[current_features + [params['target_col']]]

            X_temp, X_test, y_temp, y_test = train_test_split(df_sub[current_features], df_sub[params['target_col']],
                                                              test_size=params['test_size'], random_state=params['seed'])

            X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp,
                                                              test_size=params['val_size'], random_state=params['seed'])

            if params['model_type'] == 'classification':
                model = LogisticRegression()
            elif params['model_type'] == 'regression':
                model = LinearRegression()
            model.fit(X_train, y_train)


            preds_train = model.predict(X_train)
            preds_val = model.predict(X_val)
            preds_test = model.predict(X_test)

            if params['model_type'] == 'classification':
                metrics_func = f1_score
            elif params['model_type'] == 'regression':
                metrics_func = mean_squared_error

            for preds, y, data_set in zip([preds_train, preds_val, preds_test], [y_train, y_val, y_test],
                                          ['train', 'val', 'test']):
                metrics = {
                    'score': metrics_func(y, preds),
                    'number_of_features': len(current_features),
                    'data_set': data_set,
                    'model_type': model_type
                }
                results.append(metrics)

            pd.DataFrame(results).to_csv('datasets_analysis_linear_regression.csv', index=False)

    def load_data_and_params(self):

        df = Utils.load_data(self.input_path)
        params = Utils.load_params(self.params_path)

        for key, value in params.items():
            setattr(self, key, value)

        return df, params

if __name__ == '__main__':
    run_pipeline = RunPipeline(params_path='../../src/data/params.yaml',
                               input_data_path='../../data/processed/merged_dataset.csv'
                               )
    run_pipeline.run()