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

#### Example

This example logs some dummy metric + model checkpoint data to Comet so that you can test out the panel. 

```python
import comet_ml

#Start Comet experiment
comet_ml.login()
experiment = comet_ml.start(project_name="tf-profiler")

for i in range(10):   
    experiment.log_remote_model(f'checkpoint_{i}', '/path/to/your/model.pkl', metadata = {'epoch': i})
    experiment.log_metric('metric1', i, epoch=i)
    experiment.log_metric('metric2', 50-i, epoch=i)

experiment.end()
```

#### Python Panel

To include this panel from the github repo, use this code in a Custom Python Panel:

```
%include https://raw.githubusercontent.com/comet-ml/comet-examples/refs/heads/master/panels/ModelCheckpointComparison/ModelCheckpointComparison.py
```

Or, you can simply [copy the code](https://raw.githubusercontent.com/comet-ml/comet-examples/refs/heads/master/panels/ModelCheckpointComparison/ModelCheckpointComparison.py) into a custom Python Panel.

#### How it works

The Python panel will retrieve a list of your model checkpoints, then use the epoch values logged to the checkpoint metadata to find the value of the specific metric at that epoch.

Once you have decided on which checkpoint you want to move forward with, you can then click the 'Register Model' button in the top right corner of the experiment page to promote that checkpoint to the Comet Model Registry.

<table>
<tr>
<td>
<img src="https://raw.githubusercontent.com/comet-ml/comet-examples/refs/heads/master/panels/ModelCheckpointComparison/register-model-button.png"
     style="max-width: 300px; max-height: 300px;">
</img>
</td>
</tr>
</table>


See [here](https://www.comet.com/docs/v2/guides/model-registry/using-model-registry/#register-a-model-from-comet-ui) for further documentation on promoting models to the model registry.
