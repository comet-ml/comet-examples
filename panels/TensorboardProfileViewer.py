import streamlit as st
import os
import subprocess
import time
import psutil
import tempfile
from comet_ml import API
import streamlit.components.v1 as components
import zipfile
import argparse
import urllib.parse

st.set_page_config(layout="wide")

if "current_profile" not in st.session_state:
    st.session_state["current_profile"] = None

@st.cache_data(show_spinner="Installing Python Packages...", persist="disk")
def install_packages(*packages):
    packages_str = " ".join([("%s" % package) for package in packages])
    os.system('pip install %s' % packages_str)

install_packages(
    "tensorboard", "tensorflow", "tensorboard_plugin_profile", "streamlit_js_eval"
)

# After installations
from streamlit_js_eval import get_page_location

def lookup_proc(name):
    for pid in psutil.pids():
        try:
            process = psutil.Process(pid)
        except Exception:
            continue

        if (
            process.name().startswith(name)
        ):
            return process

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
    page_location = get_page_location()
    if page_location is not None:
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
            if st.session_state["current_profile"] != (selected_experiment.id, selected_log):
                proc = lookup_proc("tensorboard")
                if proc:
                    #print("Killing!")
                    proc.terminate()
                    proc.kill()
                    proc.wait()

                st.session_state["current_profile"] = (selected_experiment.id, selected_log)
                log_dir = "./%s/logs/%s" % (selected_experiment.id, selected_log)
                parsed_args = [
                    "--logdir", log_dir, 
                    "--port", "6007", 
                    #"--reuse_port", "True",
                ]
                (stdout_fd, stdout_path) = tempfile.mkstemp(prefix=".tensorboard-stdout-")
                (stderr_fd, stderr_path) = tempfile.mkstemp(prefix=".tensorboard-stderr-")
                try:
                    p = subprocess.Popen(
                        ["tensorboard"] + parsed_args,
                        stdout=stdout_fd,
                        stderr=stderr_fd,
                    )
                except OSError as e:
                    raise
                finally:
                    os.close(stdout_fd)
                    os.close(stderr_fd)

                seconds = 5
                bar = st.progress(0, "Starting Tensorboard...")
                for i in range(seconds):
                    bar.progress(((i + 1) / seconds), "Starting Tensorboard...")
                    time.sleep(1)
                bar.empty()
    
            path, _ = page_location["pathname"].split("/component")
            url = page_location["origin"] + path + "/port/6007/server#profile"
            st.markdown('<a href="%s" style="text-decoration: auto;">⛶ Open in tab</a>' % url, unsafe_allow_html=True)
            components.iframe(src=url, height=700)
