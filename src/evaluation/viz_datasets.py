import pandas as pd
import plotly.graph_objs as go

# Load data
df = pd.read_csv('datasets_analysis_logistic_regression.csv')

# Filter data
df_train = df[df['data_set'] == 'train']
df_val = df[df['data_set'] == 'val']
df_test = df[df['data_set'] == 'test']

# Create traces
trace_train = go.Scatter(
    x=df_train['number_of_features'],
    y=df_train['score'],
    mode='lines',
    name='Train'
)

trace_val = go.Scatter(
    x=df_val['number_of_features'],
    y=df_val['score'],
    mode='lines',
    name='Val'
)

trace_test = go.Scatter(
    x=df_test['number_of_features'],
    y=df_test['score'],
    mode='lines',
    name='Test'
)

data = [trace_train, trace_val, trace_test]

# Layout
layout = go.Layout(
    title='F1 Score vs Number of Features',
    xaxis=dict(title='Number of Features'),
    yaxis=dict(title='F1 Score')
)

# Create plot
fig = go.Figure(data=data, layout=layout)
fig.show()
