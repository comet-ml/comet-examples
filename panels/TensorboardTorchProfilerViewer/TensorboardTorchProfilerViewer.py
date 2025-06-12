# Comet Python Panel for visualizing Pytorch Profiler information through Tensorboard
# Log the torch profile .pt.trace.json file with 
# >>> experiment.log_tensorflow_folder("./logs")

# NOTE: there is only one Tensorboard Server for your
# Python Panels; logs are shared across them

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

if "tensorboard_state" not in st.session_state:
    st.session_state["tensorboard_state"] = None

from streamlit_js_eval import get_page_location

st.set_page_config(layout="wide") 

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
    names = [exp.name for exp in experiments_with_log]
    selected_experiment_name = st.selectbox(
        "Select Experiment with log:", 
        names, 
    )
    selected_experiment = [exp for exp in experiments_with_log if exp.name == selected_experiment_name][0]


if selected_experiment.id:
    page_location = get_page_location()
    if page_location is not None:
        if not os.path.exists("./%s" % selected_experiment.id):
            bar = st.progress(0, "Downloading log files...")
            selected_experiment.download_tensorflow_folder("./%s/logs/" % selected_experiment.id)
            bar.empty()
            print('hello')
    
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
            url = page_location["origin"] + path + f"/port/6007/server?x={random.randint(1,1_000_000)}#pytorch_profiler"
            st.markdown('<a href="%s" style="text-decoration: auto;">⛶ Open in tab</a>' % url, unsafe_allow_html=True)
            components.iframe(src=url, height=700)
