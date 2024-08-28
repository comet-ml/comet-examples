from comet_ml import API, ui
import streamlit as st
from scipy.stats import spearmanr
from collections import defaultdict
import json
from PIL import Image, ImageDraw
import base64
import io
import math

st.set_page_config(layout="wide")

api = API()

@st.cache_data(persist="disk")
def get_others_summary(_experiment, experiment_id):
    return _experiment.get_others_summary()

def get_from_summary(summary, key, value):
    for data in summary:
        if data[key] == value:
            return data["valueCurrent"]
    return None

selected_type = st.selectbox("Select Experiments:", ["All", "Selected"])

if selected_type == "All":
    experiments = api.get_experiments(
        api.get_panel_workspace(),
        api.get_panel_project_name(),
    )
else:
    experiments = api.get_panel_experiments()

# Get Optimizer IDs:
optimizers = defaultdict(list)
map_others = {}
for experiment in experiments:
    summary = get_others_summary(experiment, experiment.id)
    map_others[experiment.id] = summary
    optimizer_id = get_from_summary(summary, "name", "optimizer_id")
    optimizers[optimizer_id].append(experiment)

# Select which optimizer
#optimizer_ids = sorted(list(optimizers.keys()))
if len(optimizers) > 1:
    optimizer_id = st.selectbox("Optimizer:", list(optimizers.keys()))
else:
    optimizer_id = list(optimizers.keys())[0]

if optimizer_id is None:
    st.stop()
    
# Compute Spearman Correlation between each parameter and metric:
metric_values = []
parameter_values = defaultdict(list)
metric_name = None
for experiment in optimizers[optimizer_id]:
    if experiment.id not in map_others:
        continue
    value = get_from_summary(map_others[experiment.id], "name", "optimizer_metric_value")
    if value is None or value == "none":
        value = 0
    m = float(value)
    metric_name = get_from_summary(map_others[experiment.id], "name", "optimizer_metric")
    metric_values.append(m)
    params_str = get_from_summary(map_others[experiment.id], "name", "optimizer_parameters")
    if params_str:
        params = json.loads(params_str)
    else:
        params = {}
    for param, value in params.items():
        parameter_values[param].append(value)
   
data = []
for param in parameter_values:
    coef, p = spearmanr(metric_values, parameter_values[param])
    # 320, 40 for medium
    background_color = "lightgreen" if coef >= 0 else "lightpink"
    image = Image.new("RGB", (100, 15), background_color)
    draw = ImageDraw.Draw(image)
    color = "green" if coef >= 0 else "red"
    if not math.isnan(coef):
        width = int(abs(coef) * 100)
        draw.rectangle([(0, 0), (width, 15)], fill=color) 
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode()
    else:
        img_str = None
    data.append({"Hyperparameter": param, "Magnitude": img_str, "Correlation": coef, "p-value": p} )

print("### Optimizer Sweep Summary")
print(f"""
1. Experiments: **{len(experiments)}**
2. optimizer id: **{optimizer_id}**
3. metric: **{metric_name}**
""")
    
print("### Correlations")
print(f"Ranked hyperparameters based on Spearman Correlation with optimizer metric **{metric_name}**:")
config = {
    "Magnitude": st.column_config.ImageColumn(width="medium")
}
data = sorted(data, key=lambda row: row["Correlation"], reverse=True)
st.dataframe(data, use_container_width=True, hide_index=False, column_config=config)

data = []
for experiment in optimizers[optimizer_id]:
    metric_value = get_from_summary(map_others[experiment.id], "name", "optimizer_metric_value")
    params_str = get_from_summary(map_others[experiment.id], "name", "optimizer_parameters")
    if params_str:
        params = json.loads(params_str)
    else:
        params = {}
    row = {"Experiment": experiment.name, 
           "Link": "/api/experiment/redirect?experimentKey=%s" % experiment.id, 
           metric_name: metric_value}
    for param in params:
        row[param] = params[param]
    data.append(row)
print("### Experiments")
config = {
    "Link": st.column_config.LinkColumn(display_text="🔗"),
}
data = sorted(data, key=lambda row: row[metric_name], reverse=True)
st.dataframe(data, use_container_width=True, hide_index=True, column_config=config)
