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

#### Sample Code

Once the setting "Store metrics without sampling" is on, then you merely log metrics as usual:

```python
...
experiment.log_metric("loss", 1.23, step=23)
...
```

To retrieve Total Fidelity metrics, you use the method:

```python
df = APIExperiment.get_metric_total_df("loss")
```

The returned Pandas `DataFrame` contains the following columns:

* value - the value of the metric
* timestep - the time of the metric
* step - the step that the metric was logged at
* epoch - the epoch that the metric was logged at
* datetime - the timestamp as a datetime
* duration - the duration time between this row and the previous

#### Python Panel

To include this panel from the github repo, use this code in a Custom Python Panel:

```
%include https://raw.githubusercontent.com/comet-ml/comet-examples/refs/heads/master/panels/TotalFidelityMetricPlot/TotalFidelityMetricPlot.py
```

Or, you can simply [copy the code](https://raw.githubusercontent.com/comet-ml/comet-examples/refs/heads/master/panels/TotalFidelityMetricPlot/TotalFidelityMetricPlot.py) into a custom Python Panel.

#### Resources

* Example Comet Project: [www.comet.com/examples/comet-example-total-fidelity-metrics](https://www.comet.com/examples/comet-example-total-fidelity-metrics/view/PQkAHY0HubucyIAvFX9sKF9jI/panels)
* Documentation:
  * [Logging metrics](https://www.comet.com/docs/v2/api-and-sdk/python-sdk/reference/Experiment/#comet_ml.Experiment.log_metric)
  * [Retrieving Total Fidelity Metrics](https://www.comet.com/docs/v2/api-and-sdk/python-sdk/reference/APIExperiment/#comet_ml.APIExperiment.get_metric_total_df)
