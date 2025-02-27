from comet_ml import API, ui, APIExperiment, Artifact, start, ExistingExperiment
import json
import pandas as pd
import os
from ast import literal_eval

st.set_page_config(layout="wide")

st.sidebar.header("Save Model as Artifact")
api_key = st.sidebar.text_input(
    "Comet API key:", 
    value=os.environ.get("COMET_REAL_API_KEY", ""), 
    type="password"
)
st.sidebar.write(
    """
    Your Comet API key is needed as this panel
    creates a new artifact version.
    """)
if not api_key:
    st.info("Enter your Comet API key on left sidebar", icon="ℹ️")
    st.stop()

os.environ["COMET_REAL_API_KEY"] = api_key

api = API(api_key)
experiments = api.get_panel_experiments()

root = api._client.server_url

if len(experiments) > 1:
    api_experiment = ui.dropdown("Choose one:", experiments)
else: 
    api_experiment = experiments[0]

metric_names = sorted([
    x["name"] for x in api_experiment.get_metrics_summary()
    if not x["name"].startswith("sys.")
])

# selected_metric = st.sidebar.selectbox('Select Metric to Compare:',[data['name'] for data in api_experiment.get_metrics_summary()])
selected_metric = st.selectbox('Select Metric to Compare:', metric_names)

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

    # Set step and epoch as 3rd & 4th columns if they exist
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
    
    # Map to metric value at each step/epoch
    metric_data = api_experiment.get_metrics(selected_metric)
    epoch_to_metric_value = {dp[step_or_epoch]: literal_eval(dp["metricValue"]) for dp in metric_data}
    df[selected_metric] = df[f"metadata.{step_or_epoch}"].map(epoch_to_metric_value)
    
    df = df.sort_values(by=selected_metric, ascending=False)  
    # df.reset_index(drop=True, inplace=True)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Select a model to register
    model_names = sorted(df['Model-Name'].unique().tolist())
    selected_model = st.selectbox("Select a Model to Save:", model_names)

    # Model name input
    workspace = api.get_panel_workspace()

    def create_artifact(artifact_name):
        api_experiment.download_model(selected_model, f"./{selected_model}")
        # Filter and get the "Asset-Name" where "Model-Name" is "checkpoint_0"
        asset_name = df.loc[df["Model-Name"] == selected_model, "Asset-Name"].values[0]
        path = f'./{selected_model}/{asset_name}'
        artifact = Artifact(name=artifact_name, metadata = {"model_checkpoint":path, f"{step_or_epoch}": df.loc[df["Model-Name"] == selected_model, f"metadata.{step_or_epoch}"].values[0]})
        artifact.add(path)
        exp = start(api_key = api_key, mode="get", experiment_key=api_experiment.key)
        exp.log_artifact(artifact)
        st.markdown(f''':green-background[Artifact [{artifact_name}]({root}/{workspace}/artifacts/{artifact_name.replace(" ", "%20")}) created successfully!]''')
        exp.end()

    option = st.selectbox("Select action:", ["", "Create a New Artifact", "Use Existing Artifact"])
    if option == "Use Existing Artifact":
        artifacts = api.get_artifact_list(workspace)["artifacts"]
        artifact_names = sorted([artifact["name"] for artifact in artifacts])
        artifact_name = st.selectbox("Select an Artifact for new version:", artifact_names, help = "A new version will be created with this artifact name")
    elif option == "Create a New Artifact":
        artifact_name = st.text_input("Enter New Artifact name:", selected_model)
    else:
        print("*Select an action*")
        
    if option:
        st.button("Create New Artifact Version", on_click=lambda: create_artifact(artifact_name))

else:
    ui.display('No models logged to this experiment')
