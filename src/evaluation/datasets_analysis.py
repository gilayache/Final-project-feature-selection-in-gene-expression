import pandas as pd
from sklearn.metrics import f1_score, mean_squared_error
from tqdm import tqdm

from sklearn.model_selection import train_test_split
import pandas as pd

import src.Preprocessing.encoding as encoding
import src.Preprocessing.imputation as imputation
import src.Preprocessing.preprocesing as preprocesing
import src.Preprocessing.scaling as scaling
import src.feature_selection.features_selection as features_selection
import src.models.modeling as modeling
from sklearn.pipeline import Pipeline

from pipe_v2 import RunPipeline

# def run_and_return_pipeline(df, run_pipeline, params):
#     run_pipeline.df = df
#     pipe = run_pipeline.run()
#     return pipe

if __name__ == "__main__":
    results = []
    params_path = '../../src/data/params.yaml'
    input_data_path = '../../data/processed/merged_dataset.csv'
    run_pipeline = RunPipeline(params_path=params_path, input_data_path=input_data_path)
    df, params = run_pipeline.load_data_and_params()

    run_type = params['run_type']
    preprocessing_operations = params['preprocessing_operations']

    # Initialize the Preprocessing object
    preprocessing_obj = preprocesing.Preprocessing(run_type, preprocessing_operations, df)

    # Define the columns_to_remove and target_col attributes
    preprocessing_obj.columns_to_remove = params['columns_to_remove']
    preprocessing_obj.target_col = params['target_col']

    # Apply the functions
    X, _ = preprocessing_obj.create_x_y()
    X = preprocessing_obj.remove_nan_columns(X)
    X = preprocessing_obj.remove_constant_columns(X)

    features = X.columns.to_list()

    # choose the number of feature based on the feature selection method
    K_value = params['fs_params']['K']
    max_features_value = params['fs_params']['max_features']
    fs_method_1_value = params['fs_params']['fs_method_1']
    if fs_method_1_value == 'mrmr':
        n = K_value
    elif fs_method_1_value == 'genetic_selection':
        n = max_features_value

    for i in tqdm(range(1, n + 1)):
        current_features = features[:i]
        df_sub = df[current_features + [params['target_col']]]

        X_temp, X_test, y_temp, y_test = train_test_split(df_sub[current_features], df_sub[params['target_col']],
                                                          test_size=params['test_size'], random_state=params['seed'])

        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp,
                                                          test_size=params['val_size'], random_state=params['seed'])

        # Run the pipeline and get the fitted pipeline
        features = X_train.columns.to_list()
        numerical_features = X_train.select_dtypes("number").columns
        categorical_features = X_train.select_dtypes("object").columns

        pipe = Pipeline(steps=[
            ('Imputation', imputation.Imputer(categorical_features=categorical_features,
                                              numerical_features=numerical_features)),
            ('Encoding', encoding.Encoder(encoder_name=encoder_name, features=features)),
            ('Scaling', scaling.Scaler(scaler_name=scaler_name)),
            ('Features Selection', features_selection.FeaturesSelection(**self.fs_params)),
            ('Modeling', modeling.Model(model_name=model_name, val_size=param['val_size'], seed=param['seed'],
                                        hyper_params_dict=param['hyper_params_dict']
                                        ))])
        pipe.fit(X_train, y_train)

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
                'feature_selection_method': fs_method_1_value
            }
            results.append(metrics)

    pd.DataFrame(results).to_csv('src/evaluation/src/datasets_analysis.csv', index=False)