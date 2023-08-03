from sklearn.metrics import f1_score, mean_squared_error
from pipe_v2 import RunPipeline
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import pandas as pd

import src.Preprocessing.encoding as encoding
import src.Preprocessing.imputation as imputation
import src.Preprocessing.scaling as scaling
import src.feature_selection.features_selection as features_selection
import src.models.modeling as modeling
from src.Preprocessing.preprocesing import Preprocessing


def run_pipeline_and_evaluate(params_path, input_data_path):
    run_pipeline = RunPipeline(params_path=params_path, input_data_path=input_data_path)
    df, params = run_pipeline.load_data_and_params()

    features = df.columns.to_list()

    # choose the number of feature from the feature selection method
    K_value = params['fs_params']['K']
    max_features_value = params['fs_params']['max_features']
    fs_method_1_value = params['fs_params']['fs_method_1']
    if fs_method_1_value == 'mrmr':
        n = K_value
    elif fs_method_1_value == 'genetic_selection':
        n = max_features_value

    results = []

    def run_and_return_pipeline(df_sub, run_pipeline, params):
        preprocess = Preprocessing(params['run_type'], params['preprocessing_operations'], df_sub)

        for operation in preprocess.preprocessing_operations:
            if operation in preprocess.list_of_methods:
                if operation == 'remove_constant_columns':
                    df_sub = preprocess.remove_constant_columns(df_sub)
                elif operation == 'remove_nan_columns':
                    df_sub = preprocess.remove_nan_columns(df_sub)
                elif operation == 'create_x_y':
                    X, y = preprocess.create_x_y()
                else:
                    print(f"{operation} is not a defined preprocessing operation.")

        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=params['test_size'],
                                                                    random_state=params['seed'])
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=params['val_size'],
                                                          random_state=params['seed'])

        pipe = Pipeline(steps=[
            ('Imputation', imputation.Imputer(categorical_features=run_pipeline.categorical_features,
                                              numerical_features=run_pipeline.numerical_features)),
            ('Encoding', encoding.Encoder(encoder_name=params['encoder_name'], features=run_pipeline.features)),
            ('Scaling', scaling.Scaler(scaler_name=params['scaler_name'])),
            ('Features Selection', features_selection.FeaturesSelection(**params['fs_params'])),
            ('Modeling',
             modeling.Model(model_name=params['model_name'], val_size=params['val_size'], seed=params['seed'],
                            hyper_params_dict=params['hyper_params_dict']))])

        pipe.fit(X_train_val, y_train_val)
        return pipe

    for i in tqdm(range(1, n + 1)):
        current_features = features[:i]
        df_sub = df[current_features + [params['target_col']]]

        X_temp, X_test, y_temp, y_test = train_test_split(df_sub[current_features], df_sub[params['target_col']],
                                                          test_size=params['test_size'], random_state=params['seed'])

        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp,
                                                          test_size=params['val_size'], random_state=params['seed'])

        # Run the pipeline and get the fitted pipeline
        pipe = run_and_return_pipeline(df_sub, run_pipeline, params)
        preds_train = pipe.predict(X_train)
        preds_val = pipe.predict(X_val)
        preds_test = pipe.predict(X_test)

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
                'feature_selection_method': fs_method_1_value  # Adding the feature selection method to the metrics
            }
            results.append(metrics)

    pd.DataFrame(results).to_csv('/Users/gilayache/PycharmProjects/Final-project-feature-selection-in-gene-expression/src/evaluation/src/datasets_analysis.csv', index=False)
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
        self.columns_to_remove = []  # Define the attribute here
        self.target_col = 'LumA_target'  # Define the attribute here, replace 'LumA_target' with your target column name


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
            self.cols_with_nan = [col for col in X.columns if X[col].isna().all()]

        X.drop(columns=self.cols_with_nan, inplace=True)

        return X

    def create_x_y(self):
        """
        Separate the input dataframe into features (X) and target (y)
        """
        _orig_features = self.df.columns.drop(self.columns_to_remove).to_list()

        X = self.df[_orig_features]
        y = self.df[self.target_col]

        return X, y


if __name__ == '__main__':
    params_path = '../../src/data/params.yaml'
    input_data_path = '../../data/processed/merged_dataset.csv'
    run_pipeline_and_evaluate(params_path=params_path, input_data_path=input_data_path)
