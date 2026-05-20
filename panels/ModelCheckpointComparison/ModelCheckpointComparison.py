from comet_ml import API, ui, APIExperiment, Artifact, start, ExistingExperiment
import json
import pandas as pd
import os
from ast import literal_eval

st.set_page_config(layout="wide")

api = API()
experiments = api.get_panel_experiments()

if len(experiments) > 1:
    api_experiment = ui.dropdown("Choose one:", experiments)
else:
    api_experiment = experiments[0]
    
selected_metric = st.sidebar.selectbox('Select Metric to Compare:',[data['name'] for data in api_experiment.get_metrics_summary()])

data = api_experiment.get_asset_list('model-element')

def create_df(data):
    #Convert metadata from string to json
    for item in data:
        item['metadata'] = json.loads(item['metadata'])

    #Filter data to relevant columns
    def filter_keys(data, keys):
        return [{key: item[key] for key in keys} for item in data]

    keys = ['fileName', 'dir', 'metadata']
    filtered_data = filter_keys(data, keys)

    # Create DataFrame
    df = pd.json_normalize(filtered_data)

    #drop the error_message and synced column if it exists (auto-logged by Comet)
    if 'metadata.error_message' in df.columns:
        df.drop('metadata.error_message', axis=1, inplace=True)
    if 'metadata.synced' in df.columns:
        df.drop('metadata.synced', axis=1, inplace=True)  

    df.rename(columns={'fileName': 'Asset-Name', 'dir': 'Model-Name'}, inplace=True)

    df['Model-Name'] = df['Model-Name'].str.replace('models/', '')

    # Swap the first and second columns
    cols = df.columns.tolist()
    cols[0], cols[1] = cols[1], cols[0]

    
    #Set step and epoch as 3rd & 4th columns if they exist
    if 'metadata.step' in cols:
        if 'metadata.epoch' in cols:
            cols.remove('metadata.step')
            cols.insert(2, 'metadata.step')
            cols.remove('metadata.epoch')
            cols.insert(3, 'metadata.epoch')
        else:
            cols.remove('metadata.step')
            cols.insert(2, 'metadata.step')
    elif 'metadata.epoch' in cols:
        cols.remove('metadata.epoch')
        cols.insert(2, 'metadata.epoch')

    df = df[cols]
    return df

if len(data) > 0:
    df = create_df(data)
    if 'metadata.epoch' in df.columns and 'metadata.step' in df.columns:
        step_or_epoch = st.sidebar.radio('Step or Epoch:', ["epoch", "step"], help="Map metric values to each checkpoint based on step or epoch")
    elif 'metadata.epoch' in df.columns:
        step_or_epoch = 'epoch'
    elif 'metadata.step' in df.columns:
        step_or_epoch = 'step'
    else:
        print("You must log 'step' or 'epoch' with your model metadata in order to use this panel")
        exit()
    metric_goal = st.sidebar.radio('Metric goal:', ["maximize", "minimize"])
    
    #Map to metric value at each step/epoch
    metric_data = api_experiment.get_metrics(selected_metric)
    epoch_to_metric_value = {dp[step_or_epoch]: literal_eval(dp["metricValue"]) for dp in metric_data}
    df[selected_metric] = df[f"metadata.{step_or_epoch}"].map(epoch_to_metric_value)
    
    if metric_goal == "maximize":
        df.sort_values(by=[selected_metric], inplace=True, ascending = False)
    else: 
        df.sort_values(by=[selected_metric], inplace=True, ascending = True)
        

    df.reset_index(drop=True, inplace=True)
    
    st.dataframe(df, use_container_width=True)


else:
    ui.display('No models logged to this experiment')