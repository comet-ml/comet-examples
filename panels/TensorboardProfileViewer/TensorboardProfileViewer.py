%pip install tensorboard_plugin_profile

import streamlit as st
import os
from comet_ml import API
import streamlit.components.v1 as components
import zipfile
import argparse
import urllib.parse
import tensorboard.manager
from streamlit_js_eval import get_page_location

st.set_page_config(layout="wide")

api = API()
experiments = api.get_panel_experiments()

class EmptyExperiment:
    id = None
    name = ""

experiments_with_log = [EmptyExperiment()]
for experiment in experiments:
    asset_list = experiment.get_asset_list()
    for asset in asset_list:
        if asset["type"] == "tensorflow-file":
            experiments_with_log.append(experiment)
            break

if len(experiments_with_log) == 1:
    st.write("No experiments with log")
    st.stop()
elif len(experiments_with_log) == 2:
    selected_experiment = experiments_with_log[1]
else:
    selected_experiment = st.selectbox(
        "Select Experiment with log:", 
        experiments_with_log, 
        format_func=lambda aexp: aexp.name
    )

if selected_experiment.id:
    #print(os.listdir("./%s/logs/20240528-103409" % selected_experiment.id))
    download_path = selected_experiment.id
    
    if not os.path.exists(download_path):
        bar = st.progress(0, "Downloading log files...")
        selected_experiment.download_tensorflow_folder(download_path)
        bar.empty()

    selected_log = st.selectbox(
        "Select Profile Run:", 
        [""] + os.listdir("./%s/logs/" % selected_experiment.id)
    )
    if selected_log:
        log_dir = "./%s/logs/%s" % (selected_experiment.id, selected_log)
        parsed_args = [
            "--logdir", log_dir, 
            "--port", "6007", 
            "--reuse_port", "True",
        ]
        start_result = tensorboard.manager.start(parsed_args)
        #print(start_result)
        #print(start_result.stderr)
        page_location = get_page_location()
        if page_location is not None:
            path, _ = page_location["pathname"].split("/component")
            url = page_location["origin"] + path + "/port/6007/server#profile"
            st.markdown('<a href="%s">Open Tensorboard in new tab</a>' % url, unsafe_allow_html=True)
            components.iframe(src=url, height=700)
