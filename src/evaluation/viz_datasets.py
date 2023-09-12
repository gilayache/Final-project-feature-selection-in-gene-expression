import pandas as pd
import plotly.graph_objs as go
import yaml
import numpy as np
import plotly.io as pio

# Load parameters
with open('../../src/data/params.yaml', 'r') as f:
    params = yaml.safe_load(f)
analysis_data_logistic = 'datasets_analysis_logistic_regression_mrmr.csv'
analysis_data_linear = 'datasets_analysis_linear_regression_mrmr.csv'

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
df_train = df[df['data_set'] == 'train'].copy()
df_val = df[df['data_set'] == 'val'].copy()
df_test = df[df['data_set'] == 'test'].copy()

color_map = {
    'train': 'blue',
    'val': 'green',
    'test': 'red'
}

# Convert string representation of lists to actual lists (assuming lists are stored as strings)
df_train.loc[:, "random_selection_score_list"] = df_train["random_selection_score_list"].apply(yaml.safe_load)
df_val.loc[:, "random_selection_score_list"] = df_val["random_selection_score_list"].apply(yaml.safe_load)
df_test.loc[:, "random_selection_score_list"] = df_test["random_selection_score_list"].apply(yaml.safe_load)

# Calculate median for each list and store in a new column
df_train.loc[:, "median_random"] = df_train["random_selection_score_list"].apply(np.median)
df_val.loc[:, "median_random"] = df_val["random_selection_score_list"].apply(np.median)
df_test.loc[:, "median_random"] = df_test["random_selection_score_list"].apply(np.median)

df_train['std_random'] = df_train["random_selection_score_list"].apply(np.std)
df_val['std_random'] = df_val["random_selection_score_list"].apply(np.std)
df_test['std_random'] = df_test["random_selection_score_list"].apply(np.std)


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

# Create traces for median_random
trace_val_random = go.Scatter(
    x=df_val['number_of_features'],
    y=df_val["median_random"],
    mode='lines',
    line=dict(color=color_map['val']),
    name='Val Random',
    error_y=dict(
        type='data',
        array=0.1 * np.arcsin(df_val["std_random"]),
        visible=True,
        color='rgba(0, 128, 0, 0.3)',  # This is an rgba value for semi-transparent green
        thickness=10,  # Increase the thickness
        width=0  # Set the width of the cap at the end of the error bars to 0 to remove it
    )
)

trace_train_random = go.Scatter(
    x=df_train['number_of_features'],
    y=df_train["median_random"],
    mode='lines',
    line=dict(color=color_map['train']),
    name='Train Random'
)

trace_test_random = go.Scatter(
    x=df_test['number_of_features'],
    y=df_test["median_random"],
    mode='lines',
    line=dict(color=color_map['test']),
    name='Test Random'
)

data = [trace_train_fs, trace_val_fs, trace_test_fs, trace_train_random, trace_val_random, trace_test_random]

# Layout
layout = go.Layout(
    title=title,
    xaxis=dict(title='Number of Features'),
    yaxis=dict(title=f'{score}')
)

# Create plot
fig = go.Figure(data=data, layout=layout)

# Write the plot to html
pio.write_html(fig, file="linear_regression_mrmr.html", auto_open=True)
