from comet_ml import API, ui
import os

print("1. Trying **to import** additional packages...")
try:
    import st_audio_spectrogram
    print(":white_check_mark: Pass!")
except ImportError:
    print(":x: Failed! Not a current compute-engine image.")
    
print("2. Trying **to install** and **import** additional packages...")
os.system("pip install snowflake-connector-python")
try:
    import snowflake.connector
    print(":white_check_mark: Pass!")
except ImportError:
    print(":x: Failed! Not a current compute-engine image.")

print("2. Test number of experiments in this project...")
api = API()
count = api.get_panel_experiments()
print(f"There are {len(count)} experiments in this project. Is this correct?")
    
print("3. You should see a nice message below, not a stack trace:")

api = API("645645")
api.get_workspaces()

