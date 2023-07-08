import os
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_squared_error, classification_report


class Evaluation:

    def __init__(self, model_type, y_true, y_pred, params):
        self.model_type = model_type
        self.y_true = y_true
        self.y_pred = y_pred
        self.params = params

    def evaluate(self):
        evaluation_results = {
            'Time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }

        if self.model_type == 'regression':
            mse = mean_squared_error(self.y_true, self.y_pred)
            # keep 5 decimal places only
            metrics = {'MSE': mse}
        else:  # classification
            class_report = classification_report(self.y_true, self.y_pred, output_dict=True)
            metrics = {**class_report['weighted avg'], **class_report['macro avg']}
            metrics = {key: "{:.5f}".format(value) for key, value in metrics.items()}

        evaluation_results.update({'Metric_' + key: value for key, value in metrics.items()})
        evaluation_results.update({'Param_' + key: value for key, value in self.params.items()})

        self._write_to_csv(evaluation_results)

    def _write_to_csv(self, evaluation_results):
        # Convert all values to string to ensure they can be saved in CSV
        for key in evaluation_results.keys():
            evaluation_results[key] = str(evaluation_results[key])

        # Create results directory if it doesn't exist
        if not os.path.exists('src/evaluation/results'):
            os.makedirs('src/evaluation/results')

        if self.model_type == 'classification':
            path = 'src/evaluation/results/classification_results.csv'
        else:  # regression
            path = 'src/evaluation/results/regression_results.csv'

        # Create dataframe
        df = pd.DataFrame(evaluation_results, index=[0])

        if os.path.isfile(path):
            # Load the existing CSV file
            existing_df = pd.read_csv(path)

            # Adjust 'Time' column to desired format
            try:
                existing_df['Time'] = pd.to_datetime(existing_df['Time']).dt.strftime('%Y-%m-%d %H:%M:%S')
            except ValueError:
                try:
                    existing_df['Time'] = pd.to_datetime(existing_df['Time'], format='%d/%m/%Y %H:%M').dt.strftime(
                        '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    # handle the case where the time data cannot be converted to any known format
                    pass

            # Ensure the DataFrame to be appended has the same column order as the existing CSV
            df = df[existing_df.columns]

            # Check if the file ends with a newline, if not, add it
            with open(path, 'rb+') as f:
                f.seek(-2, os.SEEK_END)
                if f.read(2) != b'\n':
                    f.write(b'\n')

            # Append DataFrame to the CSV file
            df.to_csv(path, mode='a', header=False, index=False)
        else:
            df.to_csv(path, index=False)
import csv
from datetime import datetime
from sklearn.metrics import mean_squared_error, classification_report

# todo: noa - finish -- make debud and make sure everything is working
class Evaluate():
    """

    """
    def __init__(self, model_type, y_pred, y_test):
        """

        """
        self.model_type = model_type
        self.y_test = y_test
        self.y_pred = y_pred

    def run(self):
        """

        """
        result = self.evaluate()
        self.export_results(result)


    def evaluate(self):
        """

        """
        if self.model_type == 'regression':
            mse_test = mean_squared_error(self.y_test, self.y_pred)
            print("Test Mean Squared Error:", mse_test)
            return mse_test
        elif self.model_type == 'classification':
            classification_rep_test = classification_report(self.y_test, self.y_pred)
            print("Test Classification Report:\n", classification_rep_test)
            return classification_rep_test
        else:
            print("Please make sure that the model_type is regression or classification")

    def export_results(self, res):
        """

        """
        res = res
        current_date = datetime.now()
        with open(f'results_{self.model_type}_{current_date}.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(res)
