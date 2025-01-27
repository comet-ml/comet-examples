%pip install streamlit-free-text-select

from streamlit_free_text_select import st_free_text_select
import io
import re
import zipfile
from fnmatch import fnmatch

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from comet_ml import API

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

if "plotly_chart_ranges" not in st.session_state:
    st.session_state["plotly_chart_ranges"] = {"xaxis": None}

if "metric_priorities" not in st.session_state:
    st.session_state["metric_priorities"] = ["train/", "optim/"]

@st.cache_data(persist="disk")
def get_metric_asset_df(_experiment, experiment_id, metric_name, x_axis, server_end_time):
    metric_name_original = metric_name
    metric_name = (
        metric_name.replace("/", "_")
        .replace(" ", "_")
        .replace("(", "_")
        .replace(")", "_")
        .replace("%", "_")
    )
    while "__" in metric_name:
        metric_name = metric_name.replace("__", "_")
    asset_list = _experiment.get_asset_list("ASSET_TYPE_FULL_METRIC")
    metric_list = sorted(
        [
            metric
            for metric in asset_list
            if re.match(metric_name + "_\\d+.csv.zip$", metric["fileName"])
        ],
        key=lambda item: item["fileName"],
    )
    dfs = []
    df = None
    for metric in metric_list:
        df = get_asset_df(experiment, experiment.id, metric["assetId"])
        dfs.append(df)
    if dfs:
        df = pd.concat(dfs)
    else:
        if x_axis == 'step':
            #If full fidelity assets do not exist, retrieve normal metric data via API
            df1 = api.get_metrics_df(experiment_keys=[experiment.id], metrics = [metric_name_original], x_axis = x_axis)
            column_name = [col for col in df1.columns if col in ['step', 'epoch', 'duration']][0]
            #Reformat to match full fidelity output
            df = pd.DataFrame({
                'value': df1[metric_name_original],
                'timestamp': None,
                'step': df1['step'],
                'epoch': None
            })
    return df

@st.cache_data(persist="disk")
def get_asset_df(_experiment, experiment_id, asset_id):
    df = None
    data = experiment.get_asset(asset_id, return_type="binary")
    with io.BytesIO(data) as fp:
        with zipfile.ZipFile(fp, "r") as zip_ref:
            for file_info in zip_ref.infolist():
                with zip_ref.open(file_info.filename) as file:
                    df = pd.read_csv(file)
    return df

def get_sampled_total_fidelity(df, size, xaxis=None):
    if xaxis is not None:
        xaxis["range"] = sorted(xaxis["range"])
        df = df.loc[
            (df[x_axis] >= xaxis["range"][0]) & (df[x_axis] <= xaxis["range"][1])
        ]
    total_in_range = len(df)
    if size < len(df):
        df = df.sample(size, random_state=42)
    return df.sort_values(by=x_axis), total_in_range


def get_metric_priority(metric_name: str) -> int:
    for priority, pattern in enumerate(st.session_state["metric_priorities"]):
        if fnmatch(metric_name, pattern + "*"):
            return priority
    return 1000

def handle_selection():
    if "plotly_chart" in st.session_state:
        if "box" in st.session_state["plotly_chart"]["selection"]:
            st.session_state["plotly_chart_ranges"] = {
                "xaxis": {"range": st.session_state["plotly_chart"]["selection"]["box"][0]["x"]},
            }    

def sort_metric_names(metric_names):
    return sorted(metric_names, key=lambda name: (get_metric_priority(name), name))

def add_metric():
    st.session_state["metric_priorities"].append(st.session_state.new_metric)
    st.session_state.new_metric = ""

api = API()

experiments = api.get_panel_experiments()
colors = api.get_panel_experiment_colors()

with st.sidebar:
    st.number_input(
        "Number of points per curve:", 
        min_value=100, 
        max_value=1500, 
        value=500, 
        step=100, 
        key="bins", 
    )
    with st.expander("Selection metrics"):
        st.text_input(
            "Add a metric priority:", 
            on_change=add_metric,
            key="new_metric"
        )
        st.multiselect(
            label="Priority metrics:",
            options=st.session_state["metric_priorities"],
            default=st.session_state["metric_priorities"],
            key="metric_priorities"
        )
    metric_names = sort_metric_names(api.get_panel_metrics_names())
    if len(metric_names) == 1:
        metric_name = metric_names[0]
    else:
        metric_name = st.selectbox("Select metric:", metric_names)
    y_axis_scale_type = st.selectbox("Y axis scale:", ["linear", "log"])
    x_axis = st_free_text_select(
        label="X axis:",
        options=["step", "duration", "timestamp"],
        index=0,
        delay=300,
        label_visibility="visible",
        #key="free-text",
    )

if metric_name:
    if st.button(
        "Reset", 
        icon=":material/home:",
        disabled=st.session_state["plotly_chart_ranges"] == {"xaxis": None}
    ):
        st.session_state["plotly_chart_ranges"] = {"xaxis": None}

    fig = go.Figure()
    bar = st.progress(0, "Loading %s ..." % metric_name)
    fig.update_layout(
        showlegend=False,
        title=f"Total Fidelity: {metric_name}",
        **st.session_state["plotly_chart_ranges"]
    )
    fig.update_yaxes(type=y_axis_scale_type)
    for i, experiment in enumerate(experiments):
        bar.progress(i / len(experiments), "Loading %s ..." % metric_name)
        df = get_metric_asset_df(
            experiment, experiment.id, metric_name, x_axis, experiment.end_server_timestamp
        )
        if df is not None:
            if x_axis == "duration":
                df["duration"] = df["timestamp"] - df["timestamp"].min()
            if x_axis in df:
                df, n = get_sampled_total_fidelity(df, 100_000_000, **st.session_state["plotly_chart_ranges"])
                num_bins = st.session_state["bins"]
                if not df.empty:
                    df["bin"] = pd.cut(df.index, bins=num_bins, labels=False)
                    bin_maxs = df.groupby('bin').max()
                    #print(df.groupby('bin').size())
                    fig.add_trace(go.Scatter(
                        x=bin_maxs[x_axis], 
                        y=bin_maxs["value"], 
                        mode='lines',
                        fill=None,
                        marker=dict(color=colors[experiment.id]["primary"] if colors else None),
                        name=experiment.name,
                    ))
                    bin_mins = df.groupby('bin').min()
                    fig.add_trace(go.Scatter(
                        x=bin_mins[x_axis], 
                        y=bin_mins["value"], 
                        mode='lines',
                        fill="tonexty",
                        marker=dict(color=colors[experiment.id]["primary"] if colors else None),
                        name=experiment.name,
                    ))
                    
        bar.empty()
    #st.plotly_chart(fig, use_container_width=True)
    st.plotly_chart(
        fig, 
        use_container_width=True, 
        on_select=handle_selection,
        selection_mode="box",
        key="plotly_chart",
        config={"displayModeBar": False},
    )