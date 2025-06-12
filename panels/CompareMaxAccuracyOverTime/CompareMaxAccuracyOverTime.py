
from comet_ml import API, ui
import plotly.graph_objects as go
from datetime import datetime
import pandas as pd

# Get available metrics
api = API()
metrics = api.get_panel_metrics_names()

# Make chart interactive by adding a dropdown menu
selected_metric = ui.dropdown('Select a metric:', metrics)

# Use API to fetch the metric data for all experiments in the panel scope
experiment_keys = api.get_panel_experiment_keys()

# Use API to fetch experiment colors
colors = api.get_panel_experiment_colors()
if colors:
    colors = {key: value["primary"] for key,value in colors.items()}
        
if experiment_keys and selected_metric:
    data = api.get_metrics_for_chart(experiment_keys, [selected_metric])
    # Prepare data for the scatter plot and calculate averages
    x_data = []
    y_data = []
    hover_text = []
    exp_color = []

    # To hold date and accuracy values for calculating the average per date
    date_accuracy_pairs = []

    for exp_id, exp_data in data.items():
        metrics = exp_data["metrics"]
        if metrics:
            accuracy_metrics = [m for m in metrics if m["metricName"] == selected_metric]
            if accuracy_metrics:
                max_accuracy = max(accuracy_metrics[0]["values"])
                max_accuracy_idx = accuracy_metrics[0]["values"].index(max_accuracy)
                timestamp = accuracy_metrics[0]["timestamps"][max_accuracy_idx]

                # Convert timestamp to datetime (only date part)
                timestamp_dt = datetime.fromtimestamp(timestamp / 1000).date()

                # Append data
                x_data.append(timestamp_dt)
                y_data.append(max_accuracy)
                hover_text.append(exp_data["experimentName"])
                exp_color.append(colors.get(exp_id, "#000000"))

                # Store date and accuracy for average calculation
                date_accuracy_pairs.append((timestamp_dt, max_accuracy))

    # Calculate the average accuracy per date
    df = pd.DataFrame(date_accuracy_pairs, columns=["date", selected_metric])
    average_data = df.groupby("date").mean().reset_index()

    # Create scatter plot using Plotly
    fig = go.Figure()

    # Scatter plot for individual experiment points
    fig.add_trace(go.Scatter(
        x=x_data,
        y=y_data,
        mode='markers',
        marker=dict(
            size=10,
            color=exp_color),
        text=hover_text,  # Experiment names for hover
        #hoverinfo='text',  # Show only hover text
        name=selected_metric,
    ))

    # Line plot for the average accuracy across all experiments
    fig.add_trace(go.Scatter(
        x=average_data["date"],
        y=average_data[selected_metric],
        mode='lines',
        name=f"Average {selected_metric}",
        line=dict(color='red', width=2)
    ))

    # Update layout
    fig.update_layout(
        title="Max Accuracy vs Date by Experiment",
        xaxis_title="Date",
        yaxis_title=f"Maximum {selected_metric}",
        xaxis=dict(tickformat='%Y-%m-%d'),
        hovermode='closest'
    )
    ui.display(fig)
else:
    ui.display("No data to plot")