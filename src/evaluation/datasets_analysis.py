from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import pandas as pd

import src.Preprocessing.preprocesing as preprocesing
import src.feature_selection.features_selection as features_selection
import src.models.modeling as modeling

from sklearn.metrics import mean_squared_error, f1_score
from tqdm import tqdm
from src import Utils
import random

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

        y_train, y_val, y_test = y_train.values.reshape(-1, 1), y_val.values.reshape(-1, 1), y_test.values.reshape(-1, 1)

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

        shuffled_features = self.features.copy()
        random.shuffle(shuffled_features)

        results = []

        # n is the numbers of features to select for the feature selection algorithm.
        for i in tqdm(range(1, n + 1)):
            if K_value:
                self.fs_params['K'] = i
            elif max_features_value:
                self.fs_params['max_features'] = i

            current_features = self.features[:i]
            X_train_filtered = X_train[current_features]
            X_val_filtered = X_val[current_features]
            X_test_filtered = X_test[current_features]

            pipe = Pipeline(steps=[
                ('Features Selection', features_selection.FeaturesSelection(**self.fs_params)),
                ('Modeling', modeling.Model(model_name=self.model_name, val_size=self.val_size, seed=self.seed,
                                            hyper_params_dict=self.hyper_params_dict))
            ])

            # fitting the pipeline using the filtered X_train
            pipe.fit(X_train_filtered, y_train)
            preds_train = pipe.predict(X_train_filtered)
            preds_val = pipe.predict(X_val_filtered)
            preds_test = pipe.predict(X_test_filtered)

            # Create lists to collect metrics over 100 iterations
            random_train_metrics = []
            random_val_metrics = []
            random_test_metrics = []

            # Loop for 100 iterations for random selection
            for _ in range(100):
                random.shuffle(shuffled_features)

                # so we won't select a random feature twice in a single iteration.
                random_features = shuffled_features[:i]
                X_train_random = X_train[random_features]
                X_val_random = X_val[random_features]
                X_test_random = X_test[random_features]

                pipe_random = Pipeline(steps=[
                    ('Modeling', modeling.Model(model_name=self.model_name, val_size=self.val_size, seed=self.seed,
                                                hyper_params_dict=self.hyper_params_dict))
                ])
                pipe_random.fit(X_train_random, y_train)
                preds_train_random = pipe_random.predict(X_train_random)
                preds_val_random = pipe_random.predict(X_val_random)
                preds_test_random = pipe_random.predict(X_test_random)

                if params['model_type'] == 'classification':
                    metrics_func = f1_score
                elif params['model_type'] == 'regression':
                    metrics_func = mean_squared_error

                random_train_metrics.append(metrics_func(y_train, preds_train_random))
                random_val_metrics.append(metrics_func(y_val, preds_val_random))
                random_test_metrics.append(metrics_func(y_test, preds_test_random))

            for preds_fs, y, random_metric_list, data_set in zip([preds_train, preds_val, preds_test], [y_train, y_val, y_test], [random_train_metrics, random_val_metrics, random_test_metrics], ['train', 'val', 'test']):
                metrics = {
                    'feature_selection_score': metrics_func(y, preds_fs),
                    'random_selection_score_list': random_metric_list,
                    'number_of_features': len(current_features),
                    'data_set': data_set,
                    'model_type': model_type
                }
                results.append(metrics)

            pd.DataFrame(results).to_csv('datasets_analysis_logistic_regression_mrmr.csv', index=False)

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
