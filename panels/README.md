### AudioCompare

The `AudioCompare` panel is used to examine audio waveforms and spectrograms
in a single experiment or across experiments. See also the built-in Audio Panel.


<table>
<tr>
<td>
<img src="https://raw.githubusercontent.com/comet-ml/comet-examples/refs/heads/master/panels/AudioCompare/audio-compare.png" 
     style="max-width: 300px; max-height: 300px;">
</img>
</td>

<td>
<img src="https://raw.githubusercontent.com/comet-ml/comet-examples/refs/heads/master/panels/AudioCompare/built-in-audio-panel.png" 
     style="max-width: 300px; max-height: 300px;">
</img>
</td>

</tr>
</table>


For more information, see the panel <a href="https://github.com/comet-ml/comet-examples/blob/master/panels/AudioCompare/README.md">README.md</a>
### CompareMaxAccuracyOverTime

The `CompareMaxAccuracyOverTime` panel is used to help track how the
retraining of a model each week compares to the previous week. This panel
creates a scatter plot of the max average of a metric (of your choosing)
over time.


<table>
<tr>
<td>
<img src="https://raw.githubusercontent.com/comet-ml/comet-examples/refs/heads/master/panels/CompareMaxAccuracyOverTime/compare-max-accuracy-over-time.png" 
     style="max-width: 300px; max-height: 300px;">
</img>
</td>
</tr>
</table>


For more information, see the panel <a href="https://github.com/comet-ml/comet-examples/blob/master/panels/CompareMaxAccuracyOverTime/README.md">README.md</a>
### DataGridViewer

The `DataGridViewer` panel is used to visualize Comet `DataGrids` which
can contain Images, text, and numeric data.

The UX is a sophisticated approach to grouping data to see (and select)
images and other data in a tabular format, with a search feature that
allows fast querying of the data (including metadata) using Python syntax.

<table>
<tr>
<td>
<img src="https://raw.githubusercontent.com/comet-ml/comet-examples/refs/heads/master/panels/DataGridViewer/tabular-view.png" 
     style="max-width: 300px; max-height: 300px;">
</img>
</td>
<td>
<img src="https://raw.githubusercontent.com/comet-ml/comet-examples/refs/heads/master/panels/DataGridViewer/group-by.png" 
     style="max-width: 300px; max-height: 300px;">
</img>
</td>
<td>
<img src="https://raw.githubusercontent.com/comet-ml/comet-examples/refs/heads/master/panels/DataGridViewer/image-dialog.png" 
     style="max-width: 300px; max-height: 300px;">
</img>
</td>
</tr>
</table>


For more information, see the panel <a href="https://github.com/comet-ml/comet-examples/blob/master/panels/DataGridViewer/README.md">README.md</a>
### CompareMaxAccuracyOverTime

The `MetricsByStep` panel is used to compare the value of your metrics at a specific step across all of your experiments using bar charts.


<table>
<tr>
<td>
<img src="https://raw.githubusercontent.com/comet-ml/comet-examples/refs/heads/master/panels/MetricsByStep/metrics-by-step-panel.png" 
     style="max-width: 300px; max-height: 300px;">
</img>
</td>
</tr>
</table>


For more information, see the panel <a href="https://github.com/comet-ml/comet-examples/blob/master/panels/MetricsByStep/README.md">README.md</a>
### ModelCheckpointComparison

The `ModelCheckpointComparison` panel is used to compare performance of your model at each of the checkpoints logged. This is a useful tool to help determine which of your model checkpoints is best performing and should be promoted via the registry. 

<table>
<tr>
<td>
<img src="https://raw.githubusercontent.com/comet-ml/comet-examples/refs/heads/master/panels/ModelCheckpointComparison/model-comparison-panel.png"
     style="max-width: 300px; max-height: 300px;">
</img>
</td>
</tr>
</table>

First, run your experiment, including logging the model checkpoints and metrics at each step/epoch in your training loop. Each model checkpoint should log the step or epoch to the metadata field, and be uniquely named based on step/epoch, so that the panel can later match each checkpoint to performance at that step/epoch.

```python
#Log the model checkpoint directly to Comet at each epoch
for i in range(10):   
    experiment.log_model(f'checkpoint_{i}', '/path/to/your/model.pkl', metadata = {'epoch': i})
    experiment.log_metric('metric1', i, epoch=i)
    experiment.log_metric('metric2', 50-i, epoch=i)


#Or log a pointer to the model checkpoint at each epoch
for i in range(10):   
    experiment.log_remote_model(f'checkpoint_{i}', '/path/to/your/model.pkl', metadata = {'epoch': i})
    experiment.log_metric('metric1', i, epoch=i)
    experiment.log_metric('metric2', 50-i, epoch=i)
```


For more information, see the panel <a href="https://github.com/comet-ml/comet-examples/blob/master/panels/ModelCheckpointComparison/README.md">README.md</a>
### ncu-rep Viewer

This Python Panel is a viewer for NVIDIA Nsight Compute profiler reports (.ncu-rep files) that teams have logged as experiment assets. It turns those raw GPU-kernel profiles into an interactive, Nsight-style dashboard right inside the Comet UI.

<table>
<tr>
<td>
<img src="https://raw.githubusercontent.com/comet-ml/comet-examples/refs/heads/master/panels/NcuRepViewer/ncu-rep-viewer.png"
     style="max-width: 300px; max-height: 300px;">
</img>
</td>
</tr>
</table>


```python
from comet_ml import start
          
experiment = start(
  api_key="YOUR-API-KEY",
  project_name="ncu-rep-viewer",
  workspace="YOUR-WORKSPACE"
)

# examples:
experiment.log_asset("manual_nvtx.ncu-rep")
experiment.log_asset("CuVectorAddDrv.ncu-rep")
experiment.log_asset("mergeSort.ncu-rep")

experiment.end()
```

The flow:

1. Discovers reports — scans the experiments currently selected in your Comet project and collects every .ncu-rep asset attached to them.
2. Downloads on demand — when you pick a report from the sidebar dropdown, it streams that asset from Comet to local disk (cached, so re-selecting is instant) and extracts the per-kernel profiling data.
3. Lets you drill in — you choose a specific kernel launch and can filter its metrics.

What it presents, across three tabs:

- Summary — a table of all kernels in the report (duration, grid/block size, memory & compute throughput, occupancy, detected issues), plus a comparison chart when there's more than one kernel.
- Details — the Nsight-style analysis sections for the selected kernel:
  - a device header (GPU name, compute capability, SM count, memory, cache, clocks),
  - Speed of Light (compute vs memory throughput, bottleneck call),
  - Roofline (achieved performance vs arithmetic intensity),
  - Compute and Memory workload analysis, plus detailed per-unit memory tables (L1/TEX, L2, DRAM, shared),
  - scheduler / warp-stall, instruction, launch, and occupancy statistics,
  - NVTX ranges active at launch, and
  - analysis & recommendations surfaced from Nsight's own rules.
- Raw Metrics — the complete searchable metric table for the kernel.

A design principle worth noting: the richer sections (Roofline, memory tables) need reports captured with a fuller metric set. When a given report wasn't profiled that deeply, those sections show a short "needs --set full" note instead of failing — so the panel works gracefully on both lightweight and detailed captures.
### NotebookViewer

The `NotebookViewer` panel is used to render logged Notebooks, either from
[colab.research.google.com](https://colab.research.google.com/) or
any [Jupyter Notebook](https://jupyter.org/).

Comet will automatically log your Colab notebooks, both as a full
history of commenads as `Code.ipynb', but also as a completed notebook
with images and output. For Jupyter, you can use our
[cometx config --auto-log-notebook yes](https://github.com/comet-ml/cometx/blob/main/README.md#cometx-config)


<table>
<tr>
<td>
<img src="https://raw.githubusercontent.com/comet-ml/comet-examples/refs/heads/master/panels/NotebookViewer/notebookviewer.png" 
     style="max-width: 300px; max-height: 300px;">
</img>
</td>
</tr>
</table>


For more information, see the panel <a href="https://github.com/comet-ml/comet-examples/blob/master/panels/NotebookViewer/README.md">README.md</a>
### OptimizerAnalysis

The `OptimizerAnalysis` panel is used to explore results from an
Optimizer Search or Sweep. The [Comet Optimizer]() is used to
dynamically find the best set of hyperparameter values that will
minimize a Hyper Parameter Optimization tool (HPO) that can be used to
maximize a particular metric. The OptimizerAnalysis panel, combined
with the [Parallel Coordinate Chart](https://www.comet.com/docs/v2/guides/comet-ui/experiment-management/visualizations/parallel-coordinate-chart/)
allows detailed exploration of the results from your grid search or
sweep.


<table>
<tr>
<td>
<img src="https://raw.githubusercontent.com/comet-ml/comet-examples/refs/heads/master/panels/OptimizerAnalysis/optimizer-analysis.png" 
     style="max-width: 300px; max-height: 300px;">
</img>
</td>
</tr>
</table>


For more information, see the panel <a href="https://github.com/comet-ml/comet-examples/blob/master/panels/OptimizerAnalysis/README.md">README.md</a>
### SaveModelAsArtifact

This panel allows you to save a model as an artifact. Adding
metadata to the model when you log it allows examination,
and saving, by epoch. You can either create a new Artifact,
or use an existing artifact name.

<table>
<tr>
<td>
<img src="https://raw.githubusercontent.com/comet-ml/comet-examples/refs/heads/master/panels/SaveModelAsArtifact/save-model-as-artifact.png" 
     style="max-width: 300px; max-height: 300px;">
</img>
</td>
</tr>
</table>



For more information, see the panel <a href="https://github.com/comet-ml/comet-examples/blob/master/panels/SaveModelAsArtifact/README.md">README.md</a>
### TensorboardGroupViewer

The `TensorboardGroupViewer` panel is used to visualize
Tensorboard-logged items inside a Comet Custom Panel, by grouping. This
panel specifically is used to see a group of experiments' log folders.

<table>
<tr>
<td>
<img src="https://raw.githubusercontent.com/comet-ml/comet-examples/refs/heads/master/panels/TensorboardGroupViewer/tensorboard-group-viewer.png"
     style="max-width: 300px; max-height: 300px;">
</img>
</td>
</tr>
</table>

First, run your experiment, including writing and logging the
Tensorboard log folder:

```python
# Set up your experiment
writer = tf.summary.create_file_writer("./logs/%s" % experiment.name)
# Log items, including profile, to writer
# Then, log the folder:
experiment.log_tensorflow_folder("./logs")
```

Next, in the Comet UI you use the the "Group experiments" option on
the left-hand side of the project view. Select the group you'd like to
see the profiles. Finally click on "Copy Selected Experiment Logs to
Tensorboard Server" in this panel.


For more information, see the panel <a href="https://github.com/comet-ml/comet-examples/blob/master/panels/TensorboardGroupViewer/README.md">README.md</a>
### TensorboardProfileViewer

The `TensorboardProfileViewer` panel is used to visualize Tensorboard
Profile data logged data inside a Comet Custom Panel.


<table>
<tr>
<td>
<img src="https://raw.githubusercontent.com/comet-ml/comet-examples/refs/heads/master/panels/TensorboardProfileViewer/tensorboard-profile-viewer.png"
     style="max-width: 300px; max-height: 300px;">
</img>
</td>
</tr>
</table>

First, run your experiment, including writing and logging the
Tensorboard logdir:

```python
# Set up your experiment and callbacks:
tboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=logs,
    histogram_freq=1,
    profile_batch='500,520'
)
model.fit(
    ds_train,
    epochs=2,
    validation_data=ds_test,
    callbacks = [tboard_callback]
)
# Then, log the folder:
experiment.log_tensorflow_folder("./logs")
```

Finally click on "Select Experiment with log:" in this panel.


For more information, see the panel <a href="https://github.com/comet-ml/comet-examples/blob/master/panels/TensorboardProfileViewer/README.md">README.md</a>
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


For more information, see the panel <a href="https://github.com/comet-ml/comet-examples/blob/master/panels/TensorboardTorchProfilerViewer/README.md">README.md</a>
### TotalFidelityMetricPlot

The `TotalFidelityMetricPlot` panel is used to plot Total Fidelity Metrics --- metrics that are not sampled in any way.

You can have your Comet Adminstrator turn on "Store metrics without sampling" in the `Admin Dashboard` => `Organization settings`.

<table>
<tr>
<td>
<img src="https://raw.githubusercontent.com/comet-ml/comet-examples/refs/heads/master/panels/TotalFidelityMetricPlot/totalfidelity.png" 
     style="max-width: 300px; max-height: 300px;">
</img>
</td>
<td>
<img src="https://raw.githubusercontent.com/comet-ml/comet-examples/refs/heads/master/panels/TotalFidelityMetricPlot/organization-settings.png" 
     style="max-width: 300px; max-height: 300px;">
</img>
</td>
</tr>
</table>


For more information, see the panel <a href="https://github.com/comet-ml/comet-examples/blob/master/panels/TotalFidelityMetricPlot/README.md">README.md</a>
