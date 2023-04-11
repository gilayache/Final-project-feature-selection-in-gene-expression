import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.Preprocessing import preprocesing
from src.feature_selection import feature_selection_classification
from sklearn.pipeline import FeatureUnion, Pipeline

data = pd.read_csv('data/processed/merged_dataset.csv')

# split x_y
X, y = preprocesing.split_x_y(data, 'Lympho')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# todo: create eval_results methods

# todo: to use the pipeline below we should use classes. should i build a class for each method?
#  ie we will have a Preprocessing class, Feature Selection class and models class for sure
# todo: we can also consider use Dimensional Reduction methods and then use regular feature selection and check which
#  method gives us better results (and anyway that might be interesting to see)

# pipeline = Pipeline(steps=[
#         ('remove_constant_columns', preprocessing.remove_constant_columns()),
#         ('remove_nan_columns', preprocessing.remove_nan_columns())
#         ('select_features', feature_selection_classification.mrmr())])

# pipeline.fit(X_train, y_train)
