### TensorboardTorchProfilerViewer

The `TensorboardTorchProfilerViewer` panel is used to visualize Pytorch
Profile data via Tensorboard.


<table>
<tr>
<td>
<img src="https://raw.githubusercontent.com/comet-ml/comet-examples/refs/heads/master/panels/TensorboardTorchProfilerViewer/torch_profiler.png"
     style="max-width: 300px; max-height: 300px;">
</img>
</td>
</tr>
</table>

First, run your experiment, including writing and logging the
Tensorboard logdir:

```python
# Use the PyTorch profiler with trace saving
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU
    ],
    record_shapes=True,
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./logdir")  # Saves trace
) as prof:
    for _ in range(5):
        output = model(input)
        prof.step()  # Important: must call step() in each iteration


#Log the folder to Comet        
experiment.log_tensorflow_folder("./logdir")
```

Finally click on "Select Experiment with log:" in this panel.

#### Example

This example logs some dummy torch profiling data and logs the folder to Comet.

```python
import comet_ml
import torch
import torch.nn as nn
import torch.profiler

#Start Comet experiment
comet_ml.login()
experiment = comet_ml.start(project_name="tf-profiler")

# Define a simple model
model = nn.Linear(10, 1)
input = torch.randn(1, 10)

# Use the PyTorch profiler with trace saving
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU
    ],
    record_shapes=True,
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./logdir")  # Saves trace
) as prof:
    for _ in range(5):
        output = model(input)
        prof.step()  # Important: must call step() in each iteration



#Log the folder to Comet        
experiment.log_tensorflow_folder("./logdir")

experiment.end()
experiment.end()
```

#### Python Panel

To include this panel from the github repo, use this code in a Custom Python Panel:

```
%include https://raw.githubusercontent.com/comet-ml/comet-examples/refs/heads/master/panels/TensorboardTorchProfilerViewer/TensorboardTorchProfilerViewer.py
```

Or, you can simply [copy the code](https://raw.githubusercontent.com/comet-ml/comet-examples/refs/heads/master/panels/TensorboardTorchProfilerViewer/TensorboardTorchProfilerViewer.py) into a custom Python Panel.

#### How it works

The Python panel will start a Tensorboard server and make available
the logs from the experiment that is selected.

#### Resources

* [Example Comet Project](https://www.comet.com/chasefortier/tf-profiler/f156e9c72d9b4e12b8eae4ecf6db43a1?compareXAxis=step&experiment-tab=panels&prevPath=%2Fchasefortier%2Ftf-profiler%2Fview%2Fnew%2Fpanels&showOutliers=true&smoothing=0&xAxis=step)
* Documentation:
  * [Logging tensorflow folders](https://www.comet.com/docs/v2/api-and-sdk/python-sdk/reference/Experiment/#comet_ml.Experiment.log_tensorflow_folder)
  * [Automatic Tensorboard logging](https://www.comet.com/docs/v2/integrations/third-party-tools/tensorboard/#configure-comet-for-tensorboard)
  * [Download tensorboard folders](https://www.comet.com/docs/v2/api-and-sdk/python-sdk/reference/APIExperiment/#comet_ml.APIExperiment.download_tensorflow_folder)
