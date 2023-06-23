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

        def run():
            """

            """
            result = evaluate()
            export_results(result)

        def evaluate(self):
            """

            """
            if self.model_type == 'regression':
                mse_test = mean_squared_error(self.y_test, self.y_pred)
                print("Test Mean Squared Error:", mse_test)
            elif self.model_type == 'classification':
                classification_rep_test = classification_report(self.y_test, self.y_pred)
                print("Test Classification Report:\n", classification_rep_test)
            else:
                print("Please make sure that the model_type is regression or classification")

        def export_results(self, res):
            """

            """
            res = res
            current_date = datetime.datetime.now()
            with open(f'results_{self.model_type}_{current_date}.csv', 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(res)
