%pip install aitk

import os
from comet_ml import API

st.markdown("## Smoke Tests")

print("You are running compute engine %s" % os.environ["ENGINE_VERSION"])


st.markdown("### 1. Using `%pip magic` to install additional packages?")
print(":white_check_mark: Pass! `%pip` works")

st.markdown("### 2. Import additional packages?")

try:
    import aitk
    print(":white_check_mark: Pass! Can `import` %pip-installed packages")
except Exception:
    print(":x: Failed! Not a current compute-engine image.")

st.markdown("### 3. Import pre-installed packages?")
try:
    import st_audio_spectrogram
    print(":white_check_mark: Pass!")
except ImportError:
    print(":x: Failed! Pre-installed packages not found.")


print("### 4. Test number of experiments in this project?")
api = API()
count = api.get_panel_experiments()
print(f"There should be {len(count)} experiments selected in this project. Is this correct?")
    
print("### 5. Test parallel imports")

print("Add two copies of this Smoke Test panel to this view," +
      " and save the view.")
print("Press the **Restart Session** button below and refresh your browser.")
print("The two panels should load, first one, then the other.")

if st.button("Restart Session"):
    os.system("pkill -9 python")
