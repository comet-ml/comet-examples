### SaveModelAsAsset

This panel allows you to save a model as an asset. Adding
metadata to the model when you log it allows examination,
and saving, by epoch. You can either create a new Asset,
or use an existing asset name.

<table>
<tr>
<td>
<img src="https://raw.githubusercontent.com/comet-ml/comet-examples/refs/heads/master/panels/SaveModelAsAsset/save-model-as-asset.png" 
     style="max-width: 300px; max-height: 300px;">
</img>
</td>
</tr>
</table>


#### Example

This demo creates fake model checkpoints with fake metrics.

Note the metadata on the model. That will be used in the custom panel.

```python
import comet_ml
import random

comet_ml.login()

experiment = comet_ml.start(
    project_name="model-to-artifact",
)

EPOCHS = 200
MODEL_FILE = 'model-a.pkl'

# Create a dummy checkpoint file:
with open(MODEL_FILE, "w") as fp:
    fp.write("This is the model checkpoint")

last_saved = 0
for i in range(EPOCHS):
    experiment.log_metric('metric1', i*2 + random.randint(1, EPOCHS), epoch=i)
    experiment.log_metric('metric2', 5000-(i*2 + random.randint(1, EPOCHS)), epoch=i)
    if i % 30 == 0:
        last_saved = i
        experiment.log_model(
            name=f'model_chk_{i}',
            file_or_folder=MODEL_FILE,
            metadata={'epoch': i}
        )
else:
    if i != last_saved:
        # Always log model for the last epoch
        experiment.log_model(
            name=f'model_chk_{i}',
            file_or_folder=MODEL_FILE,
            metadata={'epoch': i}
        )

experiment.end()
```

#### Python Panel

To include this panel from the github repo, use this code in a Custom Python Panel:

```
%include https://raw.githubusercontent.com/comet-ml/comet-examples/refs/heads/master/panels/SaveModelAsArtifact/SaveModelAsArtifact.py
```

Or, you can simply [copy the code](https://raw.githubusercontent.com/comet-ml/comet-examples/refs/heads/master/panels/SaveModelAsArtifact/SaveModelAsArtifact.py) into a custom Python Panel.

#### Resources

* Example Comet Project: [www.comet.com/examples/comet-example-save-model-as-artifact](https://www.comet.com/examples/comet-example-save-model-as-artifact)
* Documentation:
  * [Logging a model](https://www.comet.com/docs/v2/guides/experiment-management/log-data/models/)
  * [Artifacts](https://www.comet.com/docs/v2/guides/artifacts/using-artifacts/)
