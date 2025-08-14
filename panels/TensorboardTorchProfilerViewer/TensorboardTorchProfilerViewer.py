%pip install torch_tb_profiler

# Comet Python Panel for visualizing Pytorch Profiler information through Tensorboard
# Log the torch profile .pt.trace.json file with 
# >>> experiment.log_tensorflow_folder("./logs")

# NOTE: there is only one Tensorboard Server for your
# Python Panels; logs are shared across them

# Comet Python Panel for visualizing Tensorboard Profile (and other) Data
# Log the tensorboard profile (and other data) with
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
import requests
import socket

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
    selected_experiment = [
        exp for exp in experiments_with_log if exp.name == selected_experiment_name
    ][0]


def wait_to_load(seconds):
    bar = st.progress(0, "Loading Tensorboard data...")
    for i in range(seconds):
        bar.progress(((i + 1) / seconds), "Loading Tensorboard data...")
        time.sleep(1)
    bar.empty()


def is_http_server_ready(port=6007, timeout=3):
    """Check if Tensorboard HTTP server is ready by making a request to the root endpoint."""
    try:
        import requests

        response = requests.get(f"http://localhost:{port}/", timeout=timeout)
        return response.status_code == 200
    except:
        return False


def wait_for_server_stop(port=6007, max_wait=10):
    """Wait for server to stop by checking if port is no longer accepting connections."""
    start_time = time.time()

    while time.time() - start_time < max_wait:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(("localhost", port))
            sock.close()
            if result != 0:  # Port is no longer accepting connections
                return True
        except:
            return True  # Assume stopped if we can't check

        time.sleep(0.5)

    return False  # Server didn't stop within timeout


def wait_for_server(port=6007, max_wait=30):
    """Wait for server to be ready with a progress bar."""
    bar = st.progress(0, "Starting Tensorboard...")
    start_time = time.time()

    while time.time() - start_time < max_wait:
        # Check if HTTP server is responding
        if is_http_server_ready(port):
            bar.progress(1.0, "Tensorboard ready!")
            time.sleep(0.5)  # Brief pause to show completion
            bar.empty()
            return True

        # Update progress bar
        elapsed = time.time() - start_time
        progress = min(elapsed / max_wait, 0.95)  # Cap at 95% until actually ready
        bar.progress(progress, f"Starting Tensorboard... ({int(elapsed)}s)")
        time.sleep(0.5)

    bar.empty()
    st.error(f"Tensorboard failed to start within {max_wait} seconds")
    return False


if selected_experiment.id:
    page_location = get_page_location()
    if page_location is not None:
        if not os.path.exists("./%s" % selected_experiment.id):
            bar = st.progress(0, "Downloading log files...")
            selected_experiment.download_tensorflow_folder(
                "./%s/logs/" % selected_experiment.id
            )
            bar.empty()

        selected_log = st.selectbox(
            "Select Profile to view:",
            [""] + sorted(os.listdir("./%s/logs/" % selected_experiment.id)),
        )
        if selected_log:
            # Check if we need to restart server
            needs_refresh = st.session_state["tensorboard_state"] != (
                selected_experiment.id,
                selected_log,
            )

            if needs_refresh:
                # Kill existing server
                kill_status = st.empty()  # Placeholder for dynamic message
                for process in psutil.process_iter():
                    try:
                        if "tensorboard" in process.exe():
                            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    except:
                        kill_status.warning("Can't kill the server; continuing ...")
                kill_status.empty()
                # Wait for server to stop before starting new one
                if not wait_for_server_stop(port=6007, max_wait=10):
                    st.warning("Previous Tensorboard server may still be running")

                # Start new server
                command = f"/home/stuser/.local/bin/tensorboard --logdir ./{selected_experiment.id}/logs/{selected_log} --port 6007".split()
                env = {}  # {"PYTHONPATH": "/.local/lib/python3.9/site-packages"}
                process = subprocess.Popen(command, preexec_fn=os.setsid, env=env)
                st.session_state["tensorboard_state"] = (
                    selected_experiment.id,
                    selected_log,
                )

                # Wait for server to be ready
                if wait_for_server(port=6007, max_wait=30):
                    path, _ = page_location["pathname"].split("/component")
                    url = (
                        page_location["origin"]
                        + path
                        + f"/port/6007/server?x={random.randint(1,1_000_000)}#pytorch_profiler"
                    )
                    st.markdown(
                        '<a href="%s" style="text-decoration: auto;">⛶ Open in tab</a>'
                        % url,
                        unsafe_allow_html=True,
                    )
                    wait_to_load(5)
                    components.iframe(src=url, height=700)
                else:
                    st.error("Failed to start Tensorboard server. Please try again.")

            else:
                # Server already running with correct state, just show the iframe
                path, _ = page_location["pathname"].split("/component")
                url = (
                    page_location["origin"]
                    + path
                    + f"/port/6007/server?x={random.randint(1,1_000_000)}#pytorch_profiler"
                )
                st.markdown(
                    '<a href="%s" style="text-decoration: auto;">⛶ Open in tab</a>'
                    % url,
                    unsafe_allow_html=True,
                )
                components.iframe(src=url, height=700)
