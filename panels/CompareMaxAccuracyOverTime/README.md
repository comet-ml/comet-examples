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

#### Python Panel

To include this panel from the github repo, use this code in a Custom Python Panel:

```
%include https://raw.githubusercontent.com/comet-ml/comet-examples/refs/heads/master/panels/CompareMaxAccuracyOverTime/CompareMaxAccuracyOverTime.py
```

Or, you can simply [copy the code](https://raw.githubusercontent.com/comet-ml/comet-examples/refs/heads/master/panels/CompareMaxAccuracyOverTime/CompareMaxAccuracyOverTime.py) into a custom Python Panel.

#### Resources

* [Colab Notebook](https://colab.research.google.com/github/comet-ml/comet-examples/blob/master/panels/CompareMaxAccuracyOverTime/Notebook.ipynb)
* Example Comet Project: [www.comet.com/comet-demos/cifar10-vision](https://www.comet.com/comet-demos/cifar10-vision/view/kV9XoIkTfTSN0qyKCS1lKCzaF/panels)
* Documentation:
  * [Logging metrics](https://www.comet.com/docs/v2/api-and-sdk/python-sdk/reference/Experiment/#comet_ml.Experiment.log_metric)
  * [Retrieving metrics](https://www.comet.com/docs/v2/api-and-sdk/python-sdk/reference/APIExperiment/#comet_ml.APIExperiment.get_metrics)
  * [Plotly Graph Objects and Scatter Plots](https://plotly.com/python/line-and-scatter/)



