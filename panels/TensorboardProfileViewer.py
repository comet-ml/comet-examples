# Comet Python Panel for visualizing Tensorboard Profile (and other) Data
# Log the tensorboard profile (and other data) with 
# experiment.log_tensorflow_folder("./logs")

from comet_ml import API
import streamlit as st
import streamlit.components.v1 as components

import os
import subprocess
import psutil
import time
import zipfile
import random
import signal

st.set_page_config(layout="wide") 

if "tensorboard_state" not in st.session_state:
    st.session_state["tensorboard_state"] = None

from streamlit_js_eval import get_page_location

api = API()
experiments = api.get_panel_experiments()

class EmptyExperiment:
    id = None
    name = ""

experiments_with_log = [EmptyExperiment()]
for experiment in experiments:
    asset_list = experiment.get_asset_list("tensorflow-file")
    if asset_list:
        experiments_with_log.append(experiment)

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
    page_location = get_page_location()
    if page_location is not None:
        if not os.path.exists("./%s" % selected_experiment.id):
            bar = st.progress(0, "Downloading log files...")
            selected_experiment.download_tensorflow_folder("./%s" % selected_experiment.id)
            bar.empty()
    
        selected_log = st.selectbox(
            "Select Profile to view:", 
            [""] + sorted(os.listdir("./%s/logs/" % selected_experiment.id))
        )
        if selected_log:
            command = f"/home/stuser/.local/bin/tensorboard --logdir ./{selected_experiment.id}/logs/{selected_log} --port 6007".split()
            env = {} # {"PYTHONPATH": "/.local/lib/python3.9/site-packages"}
            if st.session_state["tensorboard_state"] != (selected_experiment.id, selected_log):
                #print("Killing the hard way...")
                for process in psutil.process_iter():
                    try:
                        if "tensorboard" in process.exe():
                            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    except:
                        print("Can't kill the server; continuing ...")
            
                process = subprocess.Popen(command, preexec_fn=os.setsid, env=env)
                st.session_state["tensorboard_state"] = (selected_experiment.id, selected_log)
                
                # Allow to start
                seconds = 5
                bar = st.progress(0, "Starting Tensorboard...")
                for i in range(seconds):
                    bar.progress(((i + 1) / seconds), "Starting Tensorboard...")
                    time.sleep(1)
                bar.empty()
    
            path, _ = page_location["pathname"].split("/component")
            url = page_location["origin"] + path + f"/port/6007/server?x={random.randint(1,1_000_000)}#profile"
            st.markdown('<a href="%s" style="text-decoration: auto;">⛶ Open in tab</a>' % url, unsafe_allow_html=True)
            components.iframe(src=url, height=700)
