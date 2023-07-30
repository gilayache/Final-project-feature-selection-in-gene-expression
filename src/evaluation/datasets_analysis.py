from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import f1_score, mean_squared_error
from pipe_v2 import RunPipeline
from tqdm import tqdm


def run_pipeline_and_evaluate(params_path, input_data_path):
    run_pipeline = RunPipeline(params_path=params_path, input_data_path=input_data_path)
    df, params = run_pipeline.load_data_and_params()

    features = df.drop(columns=params['columns_to_remove']).columns.to_list()

    results = []
    n = 10

    for i in tqdm(range(1, n + 1)):
        current_features = features[:i]
        df_sub = df[current_features + [params['target_col']]]

        X_temp, X_test, y_temp, y_test = train_test_split(df_sub[current_features], df_sub[params['target_col']],
                                                          test_size=params['test_size'], random_state=params['seed'])

        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp,
                                                          test_size=params['val_size'], random_state=params['seed'])

        # Run the pipeline and get the fitted pipeline
        run_pipeline.run()
        preds_train = run_pipeline.pipe.predict(X_train)
        preds_val = run_pipeline.pipe.predict(X_val)
        preds_test = run_pipeline.pipe.predict(X_test)

        if params['model_type'] == 'classification':
            metrics_func = f1_score
        elif params['model_type'] == 'regression':
            metrics_func = mean_squared_error

        for preds, y, data_set in zip([preds_train, preds_val, preds_test], [y_train, y_val, y_test],
                                      ['train', 'val', 'test']):
            metrics = {'score': metrics_func(y, preds), 'number_of_features': len(current_features),
                       'data_set': data_set}
            results.append(metrics)

    pd.DataFrame(results).to_csv('datasets_analysis.csv', index=False)


if __name__ == '__main__':
    params_path = '../../src/data/params.yaml'
    input_data_path = '../../data/processed/merged_dataset.csv'
    run_pipeline_and_evaluate(params_path=params_path, input_data_path=input_data_path)
