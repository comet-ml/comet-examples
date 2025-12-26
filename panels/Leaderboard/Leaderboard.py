%pip install tabulate
import streamlit as st
import pandas as pd
from comet_ml import API, ui
from comet_ml.api import Metadata, Parameter, Tag

# Initialize Comet API
api = API()
workspace = api.get_panel_workspace()
project_name = api.get_panel_project_name()

# Simple title
st.title("Experiment Leaderboard")
st.markdown("#### by Metric")

# Get available metrics and select one + define objective
available_metrics = api.get_panel_metrics_names()
selected_metric = st.sidebar.selectbox("Select a metric:", available_metrics)
objective = st.sidebar.selectbox("Objective Function:", ["max","min"]) # Change this to "min" for minimization

# Fetch experiment data 
experiments = api.get_experiments(workspace, project_name=project_name)
experiment_keys = [exp.key for exp in experiments]

if experiment_keys and selected_metric:
    # Fetch the selected metric data for all experiments
    df = api.get_metrics_df(experiment_keys, [selected_metric])
    # st.table(df)
    
    # Find best metric per experiment
    if objective == "max":
        best_per_experiment = df.groupby("experiment_name")[selected_metric].max().reset_index()
    else:
        best_per_experiment = df.groupby("experiment_name")[selected_metric].min().reset_index()

    # Merge to get experiment_key for URL generation
    best_per_experiment = best_per_experiment.merge(
        df[['experiment_name', 'experiment_key']], on='experiment_name', how='left'
    ).drop_duplicates()

    # Sort and get top 10
    leaderboard = best_per_experiment.sort_values(
        by=selected_metric, ascending=(objective == "min")
    ).head(10).reset_index(drop=True)
    
    # Create hyperlinks
    leaderboard["experiment_name"] = leaderboard.apply(
        lambda row: f"[{row['experiment_name']}](https://www.comet.com/{workspace}/{project_name}/{row['experiment_key']})", axis=1
    )

    # Rename metric column
    leaderboard = leaderboard.rename(
        columns={
            selected_metric: "Best Metric Value",
        }
    )
    leaderboard.drop('experiment_key', axis=1, inplace=True)
    
    # show table
    st.table(leaderboard)
