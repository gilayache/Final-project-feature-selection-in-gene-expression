import pandas as pd
import plotly.express as px

# Read the CSV file
df = pd.read_csv('/Users/gilayache/PycharmProjects/Final-project-feature-selection-in-gene-expression/src/evaluation/results/regression_results.csv')

# Extract the baseline MSE
baseline_mse = df.loc[df['Param_model_type'] == 'regression - baseline', 'Metric_MSE'].values[0]

# Create a mask of values where MSE is better (i.e., less) than the baseline
mask = df['Metric_MSE'] < baseline_mse

# Filter the dataframe using the mask, drop duplicates based on 'Metric_MSE' and make a copy
df_filtered = df[mask].drop_duplicates(subset='Metric_MSE').copy()

# Create a new column for hover information
df_filtered['hover_info'] = df_filtered.apply(lambda row: 'Number of features for the first feature selection: ' + str(row['Param_max_features'])
                                              if row['Param_fs_method_1'] == 'genetic_selection'
                                              else 'Number of features for the first feature selection: ' + str(row['Param_K']),
                                              axis=1)


df_filtered['hover_info'] += df_filtered.apply(lambda row: '<br>Second feature selection is: None'
                                               if pd.isnull(row['Param_fs_method_2'])
                                               else '<br>Second feature selection is: ' + row['Param_fs_method_2'] +
                                               ', number of features: ' + str(row['Param_n_features_to_select']), axis=1)


# Plot the filtered results
fig = px.scatter(df_filtered, x='Param_fs_method_1', y='Metric_MSE', hover_data=['hover_info'])

# Add a horizontal line for the baseline MSE
fig.add_hline(y=baseline_mse, line_dash="dash", line_color="red", annotation_text="Baseline", annotation_position="bottom right")

# Set plot title and labels
fig.update_layout(title='For regression:  MSE vs feature selection methods',
                  xaxis_title='First feature selection method',
                  yaxis_title='MSE',
                  height=600,
                  width=800,
                  )

# Show the plot
fig.show()
