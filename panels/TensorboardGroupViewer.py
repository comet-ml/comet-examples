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

needs_refresh = False
page_location = get_page_location()
if page_location is not None:
    if True:
        column = st.columns([.7, .3])
        clear = column[1].checkbox("Clear previous logs", value=True)
        if column[0].button("Copy Selected Experiment Logs to Tensorboard Server", type="primary"):
            needs_refresh = True
            if clear and os.path.exists("./logs"):
                for filename in glob.glob("./logs/*"):
                    shutil.move(filename, "./tb_cache/")
            bar = st.progress(0, "Downloading log files...")
            for i, experiment in enumerate(experiments):
                bar.progress(i/len(experiments), "Downloading log files...")
                if not os.path.exists("./logs/%s" % experiment.name):
                    if os.path.exists("./tb_cache/%s" % experiment.name):
                        if DEBUG: print("found in cache!")
                        shutil.move(
                            "./tb_cache/%s" % experiment.name,
                            "./logs/%s" % experiment.name,
                        )
                    else:
                        if DEBUG: print("downloading...")
                        assets = experiment.get_asset_list("tensorflow-file")
                        if assets:
                            if DEBUG: print(assets[0]["fileName"])
                            if assets[0]["fileName"].startswith("logs/"):
                                experiment.download_tensorflow_folder("./")
                            else:
                                experiment.download_tensorflow_folder("./logs/")
            bar.empty()

        running = False
        for process in psutil.process_iter():
            try:
                if "tensorboard" in process.exe():
                    running = True
            except:
                pass
        if not running:
            command = f"/home/stuser/.local/bin/tensorboard --logdir ./logs --port 6007".split()
            env = {} # {"PYTHONPATH": "/home/st_user/.local/lib/python3.9/site-packages"}
            process = subprocess.Popen(command, preexec_fn=os.setsid, env=env)
            needs_refresh = True

        if needs_refresh:
            # Allow to start/update
            seconds = 5
            bar = st.progress(0, "Updating Tensorboard...")
            for i in range(seconds):
                bar.progress(((i + 1) / seconds), "Updating Tensorboard...")
                time.sleep(1)
            bar.empty()

        path, _ = page_location["pathname"].split("/component")
        url = page_location["origin"] + path + f"/port/6007/server?x={random.random()}"
        st.markdown('<a href="%s" style="text-decoration: auto;">⛶ Open in tab</a>' % url, unsafe_allow_html=True)
        components.iframe(src=url, height=700)
