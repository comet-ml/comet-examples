import os
os.system('pip install "st-audio-spectrogram>=0.0.5"')

from comet_ml import API
import io
import time
import streamlit as st
from st_audio_spectrogram import st_audio_spectrogram

try:
    st.set_page_config(layout="wide")
except Exception:
    pass

api = API()

@st.cache_data(persist="disk", show_spinner="Loading asset...")
def get_asset(_experiment, experiment_id, asset_id):
    asset = experiment.get_asset(
        asset_id=asset_id, 
        return_type='binary'
    )
    return asset

def get_asset_list(_experiment, experiment_id, asset_type):
    return _experiment.get_asset_list(asset_type=asset_type)

def get_all_audio_data(_experiments, experiment_ids):
    audio_data = set()
    # First, get a selection from asset names:
    bar = st.progress(0, "Loading audio list...")
    for i, experiment in enumerate(_experiments):
        for asset in get_asset_list(experiment, experiment.id, "audio"):
            bar.progress(i/len(experiments), "Loading audio...")
            audio_data.add((experiment.id, asset["fileName"], asset["assetId"], asset["step"], ))
    bar.empty()
    return audio_data

# ----------------------------------------
experiments = api.get_panel_experiments()
experiment_map = {exp.id: exp for exp in experiments}
experiment_ids = sorted([exp.id for exp in experiments])
audio_data = get_all_audio_data(experiments, experiment_ids)

asset_names = sorted(
    list(
        set([os.path.basename(item[1]) for item in audio_data])
    )
)

selected_names = st.multiselect("", asset_names, placeholder="Select Audio Files:")

steps = set()
for asset_name in selected_names:
    for experiment_id, filename, asset_id, step in audio_data:
        if filename.endswith(asset_name):
            if step is not None:
                steps.add(step)

if steps:
    if min(steps) != max(steps):
        STEP = st.slider(
            "Select Step:", 
            min_value=min(steps), 
            max_value=max(steps),
            value=max(steps),
        )
    else:
        STEP = None
else:
    STEP = None
                
for asset_name in selected_names:
    with st.expander("Compare: **%s**" % asset_name, expanded=len(selected_names) == 1):
        for experiment_id, filename, asset_id, step in sorted(
            audio_data, key=lambda item: item[0]
        ):
            if filename.endswith(asset_name) and ((step == STEP) or (STEP is None)):
                experiment = experiment_map[experiment_id]
                audio = get_asset(experiment, experiment_id, asset_id)
                st.markdown("*Experiment*: ***%s***, *step*: ***%s***" % (
                    experiment.name, step
                ))
                with st.spinner("Loading component..."):
                    time.sleep(1)
                    st_audio_spectrogram(
                        audio, 
                        key="%s: %s" % (experiment_id, asset_id)
                    )
                    st.divider()
