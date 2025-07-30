# Comet Python Panel for visualizing Tensorboard Data by Group
# >>> experiment.log_other("Group", "GROUP-NAME")
# >>> experiment.log_tensorflow_folder("./logs")
# In the UI, group on "Group"

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
import glob
import shutil
import requests
import socket
import signal

st.set_page_config(layout="wide")

from streamlit_js_eval import get_page_location

os.makedirs("./tb_cache", exist_ok=True)
os.makedirs("./logs", exist_ok=True)

DEBUG = False

# Clear cache and downloads
if DEBUG:
    if os.path.exists("./tb_cache"):
        shutil.rmtree("./tb_cache")
    if os.path.exists("./logs"):
        shutil.rmtree("./logs")

api = API()
experiments = api.get_panel_experiments()


def wait_to_load(seconds):
    bar = st.progress(0, "Loading Tensorboard data...")
    for i in range(seconds):
        bar.progress(((i + 1) / seconds), "Loading Tensorboard data...")
        time.sleep(1)
    bar.empty()


def is_http_server_ready(port=6007, timeout=3):
    """Check if Tensorboard HTTP server is ready by making a request to the root endpoint."""
    try:
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


if "tensorboard_state" not in st.session_state:
    st.session_state["tensorboard_state"] = None

need_refresh = False
page_location = get_page_location()
if page_location is not None:
    column = st.columns([0.7, 0.3])
    clear = column[1].checkbox("Clear previous logs", value=True)
    if column[0].button(
        "Copy Selected Experiment Logs to Tensorboard Server", type="primary"
    ):
        need_refresh = True
        if clear and os.path.exists("./logs"):
            for filename in glob.glob("./logs/*"):
                shutil.move(filename, "./tb_cache/")
        bar = st.progress(0, "Downloading log files...")
        for i, experiment in enumerate(experiments):
            bar.progress(i / len(experiments), "Downloading log files...")
            if not os.path.exists("./logs/%s" % experiment.name):
                if os.path.exists("./tb_cache/%s" % experiment.name):
                    if DEBUG:
                        print("found in cache!")
                    shutil.move(
                        "./tb_cache/%s" % experiment.name,
                        "./logs/%s" % experiment.name,
                    )
                else:
                    if DEBUG:
                        print("downloading...")
                    assets = experiment.get_asset_list("tensorflow-file")
                    if assets:
                        if DEBUG:
                            print(assets[0]["fileName"])
                        if assets[0]["fileName"].startswith("logs/"):
                            experiment.download_tensorflow_folder("./")
                        else:
                            experiment.download_tensorflow_folder("./logs/")
        bar.empty()

    # Check if we need to restart server
    if st.session_state["tensorboard_state"] != "group_viewer":
        # Kill existing server
        for process in psutil.process_iter():
            try:
                if "tensorboard" in process.exe():
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except:
                print("Can't kill the server; continuing ...")

        # Wait for server to stop before starting new one
        if not wait_for_server_stop(port=6007, max_wait=10):
            st.warning("Previous Tensorboard server may still be running")

        # Start new server
        command = f"/home/stuser/.local/bin/tensorboard --logdir ./logs --port 6007".split()
        env = (
            {}
        )  # {"PYTHONPATH": "/home/st_user/.local/lib/python3.9/site-packages"}
        process = subprocess.Popen(command, preexec_fn=os.setsid, env=env)
        st.session_state["tensorboard_state"] = "group_viewer"

        # Wait for server to be ready
        if wait_for_server(port=6007, max_wait=30):
            path, _ = page_location["pathname"].split("/component")
            url = (
                page_location["origin"]
                + path
                + f"/port/6007/server?x={random.randint(1,1_000_000)}"
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
            + f"/port/6007/server?x={random.randint(1,1_000_000)}"
        )
        st.markdown(
            '<a href="%s" style="text-decoration: auto;">⛶ Open in tab</a>' % url,
            unsafe_allow_html=True,
        )
        if need_refresh:
            wait_to_load(5)
        components.iframe(src=url, height=700)
