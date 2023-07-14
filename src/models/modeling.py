import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from sklearn.model_selection import train_test_split

class Model:

    def __init__(self, model_name: str, val_size, seed, hyper_params_dict):
        """
        """
        self.model = self.InitModel(model_name)
        self.val_size = val_size
        self.seed = seed
        self.hyper_params_dict = hyper_params_dict


    def fit(self, X, y=None):
        """
        """
        if type(y) != pd.Series: # todo: change it to == np.array/np.ndarray
            y = y.reshape(-1, 1)

        elif type(y) == pd.Series:
            y = y.values.reshape(-1, 1)

        # Further split train+validation into separate train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.val_size,
                                                          random_state=self.seed)

        best_params = self.apply_hyper_param(X_val, y_val)
        print('start model fit')
        self.model.fit(X, y, **best_params)

        return self

    def transform(self, X):
        """
        """
        print('start model transform')
        X_transformed = self.model.transform(X)

        return X_transformed

    def predict(self, X):
        """
        """
        y_pred = self.model.predict(X)

        return y_pred

    def InitModel(self, model_name):
        """
        """
        if model_name == 'LogisticRegression':
            model = LogisticRegression()

        elif model_name == 'LinearRegression':
            model = LinearRegression()

        return model

    def apply_hyper_param(self, X, y):
        """

        """
        clf = RandomizedSearchCV(self.model, self.hyper_params_dict, random_state=0)
        print('start hyper params with RandomizedSearchCV')
        search = clf.fit(X, y)
        best_params = search.best_params_

        return best_params