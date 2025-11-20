from comet_ml import API, ui
import pandas as pd
import streamlit as st
import plotly.express as px

# Get available metrics
api = API()

experiment_keys = api.get_panel_experiment_keys()
all_metric_names = api.get_panel_metrics_names()

all_metrics = st.sidebar.toggle("All Metrics", value=1, help="Display chart for all metrics logged to the project.")

if not all_metrics:
    metric_names = st.sidebar.multiselect("Metrics:", all_metric_names, help= "Metrics to display.")
else:
    metric_names = all_metric_names

if metric_names:
    metrics_df = api.get_metrics_df(
        experiment_keys, metrics=metric_names, interpolate=False
    )
    
    # Get min and max step and generate step slider
    step_max = metrics_df['step'].max()
    step_min = metrics_df['step'].min()
    
    slider = st.sidebar.slider("Step:", int(step_min), int(step_max))
    
    # Filter to data for the selected step
    step_data = metrics_df.loc[metrics_df['step'] == slider].copy()
    
    # OPTIONAL: if experiment_key is the index, reset it so it's a column
    if 'experiment_key' not in step_data.columns:
        step_data = step_data.reset_index().rename(columns={'index': 'experiment_key'})
    
    # Plot a bar chart for each metric
    for metric in metric_names:
        # Skip metrics that might not be present
        if metric not in step_data.columns:
            continue
    
        metric_df = step_data[['experiment_name', metric]].copy()
    
        fig = px.bar(
            metric_df,
            x='experiment_name',
            y=metric,
            title=f"{metric} at step {slider}",
            labels={
                'experiment_name': 'Experiment',
                metric: metric,
            }
        )
    
        st.plotly_chart(fig, use_container_width=True)