import pandas as pd
import plotly.graph_objs as go
import yaml

# Load data
df = pd.read_csv('datasets_analysis_linear_regression.csv')

# Load parameters
with open('../../src/data/params.yaml', 'r') as f:
    params = yaml.safe_load(f)

# Check model type
if params['model_type'] == 'classification':
    score_column = 'f1_score'
    title = 'F1 Score vs Number of Features'
elif params['model_type'] == 'regression':
    score_column = 'mse'
    title = 'MSE vs Number of Features'

# Filter data
df_train = df[df['data_set'] == 'train']
df_val = df[df['data_set'] == 'val']
df_test = df[df['data_set'] == 'test']

# Create traces
trace_train = go.Scatter(
    x=df_train['number_of_features'],
    y=df_train[score_column],
    mode='lines',
    name='Train'
)

trace_val = go.Scatter(
    x=df_val['number_of_features'],
    y=df_val[score_column],
    mode='lines',
    name='Val'
)

trace_test = go.Scatter(
    x=df_test['number_of_features'],
    y=df_test[score_column],
    mode='lines',
    name='Test'
)

data = [trace_train, trace_val, trace_test]

# Layout
layout = go.Layout(
    title=title,
    xaxis=dict(title='Number of Features'),
    yaxis=dict(title=f'{score_column}')
)

# Create plot
fig = go.Figure(data=data, layout=layout)
fig.show()
