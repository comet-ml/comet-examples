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
import json
import fcntl
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

# --- Per-instance port assignment and TensorBoard state (6000-6009) ---
# Registry is stored in a file so it is shared across all panel processes
# and survives Streamlit session resets.

PORT_RANGE_START = 6000
PORT_RANGE_END = 6010  # exclusive
PORT_REGISTRY_FILE = "/tmp/tb_port_registry.json"


def _load_registry(f):
    """Read registry from an open file handle, migrating old int-only entries
    ({"id": port}) to the current dict format ({"id": {"port": N, "tb_state": ...}})."""
    try:
        data = json.load(f)
    except (json.JSONDecodeError, ValueError):
        return {}
    migrated = {}
    for k, v in data.items():
        if isinstance(v, int):
            migrated[k] = {"port": v, "tb_state": None}
        else:
            migrated[k] = v
    return migrated


def get_instance_port(instance_id):
    """Return the port assigned to instance_id, assigning the next available
    port if this instance hasn't been seen before.  Uses a file lock so
    concurrent panel startups don't race.  Raises RuntimeError when the port
    range is exhausted."""
    if not os.path.exists(PORT_REGISTRY_FILE):
        with open(PORT_REGISTRY_FILE, "w") as f:
            json.dump({}, f)
    with open(PORT_REGISTRY_FILE, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        registry = _load_registry(f)
        if instance_id not in registry:
            used_ports = {entry["port"] for entry in registry.values()}
            next_port = next(
                (p for p in range(PORT_RANGE_START, PORT_RANGE_END) if p not in used_ports),
                None,
            )
            if next_port is None:
                raise RuntimeError(
                    f"No available ports: all ports {PORT_RANGE_START}-{PORT_RANGE_END - 1} are in use."
                )
            registry[instance_id] = {"port": next_port, "tb_state": None}
            f.seek(0)
            f.truncate()
            json.dump(registry, f)
        return registry[instance_id]["port"]


def get_tb_state(instance_id):
    """Return the stored tb_state for instance_id from the registry file."""
    if instance_id is None or not os.path.exists(PORT_REGISTRY_FILE):
        return None
    with open(PORT_REGISTRY_FILE, "r") as f:
        fcntl.flock(f, fcntl.LOCK_SH)
        registry = _load_registry(f)
    return registry.get(instance_id, {}).get("tb_state")


def set_tb_state(instance_id, state):
    """Persist tb_state for instance_id in the registry file."""
    if instance_id is None:
        return
    if not os.path.exists(PORT_REGISTRY_FILE):
        with open(PORT_REGISTRY_FILE, "w") as f:
            json.dump({}, f)
    with open(PORT_REGISTRY_FILE, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        registry = _load_registry(f)
        if instance_id not in registry:
            registry[instance_id] = {"port": None, "tb_state": state}
        else:
            registry[instance_id]["tb_state"] = state
        f.seek(0)
        f.truncate()
        json.dump(registry, f)


def kill_process_on_port(port):
    """Kill the process listening on the given port, if any."""
    for proc in psutil.process_iter():
        try:
            for conn in proc.net_connections():
                if conn.laddr.port == port and conn.status == 'LISTEN':
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    return
        except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
            pass


instance_id = os.environ.get("COMET_PANEL_INSTANCE_ID")
if instance_id is None:
    port = 6007
else:
    port = get_instance_port(instance_id)

log_dir = f"./logs/{instance_id or 'default'}"
cache_dir = f"./tb_cache/{instance_id or 'default'}"

st.set_page_config(layout="wide")

from streamlit_js_eval import get_page_location

os.makedirs(cache_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

DEBUG = False

# Clear cache and downloads
if DEBUG:
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)

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


need_refresh = False
page_location = get_page_location()
if page_location is not None:
    column = st.columns([0.7, 0.3])
    clear = column[1].checkbox("Clear previous logs", value=True)
    if column[0].button(
        "Copy Selected Experiment Logs to Tensorboard Server", type="primary"
    ):
        need_refresh = True
        if clear and os.path.exists(log_dir):
            for filename in glob.glob(f"{log_dir}/*"):
                shutil.move(filename, cache_dir)
        bar = st.progress(0, "Downloading log files...")
        for i, experiment in enumerate(experiments):
            bar.progress(i / len(experiments), "Downloading log files...")
            if not os.path.exists(f"{log_dir}/{experiment.name}"):
                if os.path.exists(f"{cache_dir}/{experiment.name}"):
                    if DEBUG:
                        print("found in cache!")
                    shutil.move(
                        f"{cache_dir}/{experiment.name}",
                        f"{log_dir}/{experiment.name}",
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
                            downloaded = f"./logs/{experiment.name}"
                            if os.path.exists(downloaded):
                                shutil.move(downloaded, f"{log_dir}/{experiment.name}")
                        else:
                            experiment.download_tensorflow_folder(f"{log_dir}/")
        bar.empty()

    if get_tb_state(instance_id) == "group_viewer" and is_http_server_ready(port):
        # Server is already running with the correct state — just show the iframe
        path, _ = page_location["pathname"].split("/component")
        url = (
            page_location["origin"]
            + path
            + f"/port/{port}/server?x={random.randint(1,1_000_000)}"
        )
        st.markdown(
            '<a href="%s" style="text-decoration: auto;">⛶ Open in tab</a>' % url,
            unsafe_allow_html=True,
        )
        if need_refresh:
            wait_to_load(5)
        components.iframe(src=url, height=700)
    else:
        kill_process_on_port(port)

        # Wait for server to stop before starting new one
        if not wait_for_server_stop(port=port, max_wait=10):
            st.warning("Previous Tensorboard server may still be running")

        # Start new server
        command = f"/home/stuser/.local/bin/tensorboard --logdir {log_dir} --port {port}".split()
        env = {}  # {"PYTHONPATH": "/home/st_user/.local/lib/python3.9/site-packages"}
        process = subprocess.Popen(command, preexec_fn=os.setsid, env=env)
        set_tb_state(instance_id, "group_viewer")

        # Wait for server to be ready
        if wait_for_server(port=port, max_wait=30):
            path, _ = page_location["pathname"].split("/component")
            url = (
                page_location["origin"]
                + path
                + f"/port/{port}/server?x={random.randint(1,1_000_000)}"
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
