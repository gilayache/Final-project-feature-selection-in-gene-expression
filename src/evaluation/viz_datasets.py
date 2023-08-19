import pandas as pd
import plotly.graph_objs as go
import yaml
import numpy as np
import plotly.io as pio


# Load parameters
with open('../../src/data/params.yaml', 'r') as f:
    params = yaml.safe_load(f)
analysis_data_logistic = 'datasets_analysis_logistic_regression_genetic_selection.csv'
analysis_data_linear = 'datasets_analysis_linear_regression_genetic_selection.csv'


# Check model type and modify title
if params['model_type'] == 'classification':
    title = 'F1 Score vs Number of Features for classification problem,'
    score = "F1"
    df = pd.read_csv(analysis_data_logistic)
elif params['model_type'] == 'regression':
    title = 'MSE vs Number of Features for regression problem,'
    score = "MSE"
    df = pd.read_csv(analysis_data_linear)

# Adding method to title
if params['fs_params']['fs_method_1'] == 'mrmr':
    title += ' Using MRMR:'
elif params['fs_params']['fs_method_1'] == 'genetic_selection':
    title += ' Using Genetic Selection:'

# Filter data
df_train = df[df['data_set'] == 'train']
df_val = df[df['data_set'] == 'val']
df_test = df[df['data_set'] == 'test']

color_map = {
    'train': 'blue',
    'val': 'green',
    'test': 'red'
}

# Create traces for feature_selection_score
trace_train_fs = go.Scatter(
    x=df_train['number_of_features'],
    y=df_train["feature_selection_score"],
    mode='lines',
    line=dict(dash='dash', color=color_map['train']),
    name='Train FS'
)

trace_val_fs = go.Scatter(
    x=df_val['number_of_features'],
    y=df_val["feature_selection_score"],
    mode='lines',
    line=dict(dash='dash', color=color_map['val']),
    name='Val FS'
)

trace_test_fs = go.Scatter(
    x=df_test['number_of_features'],
    y=df_test["feature_selection_score"],
    mode='lines',
    line=dict(dash='dash', color=color_map['test']),
    name='Test FS'
)

# Create traces for random_selection_score
trace_train_random = go.Scatter(
    x=df_train['number_of_features'],
    y=df_train["random_selection_score"],
    mode='lines',
    line=dict(color=color_map['train']),
    name='Train Random'
)

trace_val_random = go.Scatter(
    x=df_val['number_of_features'],
    y=df_val["random_selection_score"],
    mode='lines',
    line=dict(color=color_map['val']),
    name='Val Random'
)

trace_test_random = go.Scatter(
    x=df_test['number_of_features'],
    y=df_test["random_selection_score"],
    mode='lines',
    line=dict(color=color_map['test']),
    name='Test Random'
)

data = [trace_train_fs, trace_val_fs, trace_test_fs, trace_train_random, trace_val_random, trace_test_random]

# Layout
layout = go.Layout(
    title=title,
    xaxis=dict(title='Number of Features'),
    yaxis=dict(
        title=f'{score}'
        # range=[np.percentile(pd.concat([df["feature_selection_score"], df["random_selection_score"]]), 2),
        #        np.percentile(pd.concat([df["feature_selection_score"], df["random_selection_score"]]), 98)]
        )
)

# Create plot
fig = go.Figure(data=data, layout=layout)
# Write the plot to html
pio.write_html(fig, file="linear_regression_genetic_selection.html", auto_open=True)