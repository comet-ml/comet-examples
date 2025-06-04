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

#### Example

This example logs some dummy data to Tensorflow, and
then logs the Tensorflow folder to Comet.

```python
import random
import os
import shutil
from comet_ml import Experiment
import tensorflow as tf

for e in range(12):
    experiment = Experiment(
        project_name="tensorboard-group"
    )
    if os.path.exists("./logs"):
        shutil.rmtree("./logs")
    writer = tf.summary.create_file_writer("./logs/%s" % experiment.name)
    with writer.as_default():
        current_loss = random.random()
        current_accuracy = random.random()
        for i in range(100):
            tf.summary.scalar("loss", current_loss, step=i)
            tf.summary.scalar("accuracy", current_accuracy, step=i)
            current_loss += random.random() * random.choice([1, -1])
            current_accuracy += random.random() * random.choice([1, -1])
    experiment.log_other("Group", "group-%s" % ((e % 3) + 1))
    experiment.log_tensorflow_folder("./logs")
    experiment.end()
```

#### Python Panel

To include this panel from the github repo, use this code in a Custom Python Panel:

```
%include https://raw.githubusercontent.com/comet-ml/comet-examples/refs/heads/master/panels/TensorboardGroupViewer/TensorboardGroupViewer.py
```

Or, you can simply [copy the code](https://raw.githubusercontent.com/comet-ml/comet-examples/refs/heads/master/panels/TensorboardGroupViewer/TensorboardGroupViewer.py) into a custom Python Panel.

#### How it works

The Python panel will start a Tensorboard server and make available
the logs from those experiments that are in the group.

#### Resources

* Example Comet Project: [www.comet.com/dsblank/tensorboard-group](https://www.comet.com/dsblank/tensorboard-group/view/0xR3Fm81cXMPlXp7pN63jeVgS/panels)
* Documentation:
  * [Logging tensorflow folders](https://www.comet.com/docs/v2/api-and-sdk/python-sdk/reference/Experiment/#comet_ml.Experiment.log_tensorflow_folder)
  * [Automatic Tensorboard logging](https://www.comet.com/docs/v2/integrations/third-party-tools/tensorboard/#configure-comet-for-tensorboard)
  * [Download tensorboard folders](https://www.comet.com/docs/v2/api-and-sdk/python-sdk/reference/APIExperiment/#comet_ml.APIExperiment.download_tensorflow_folder)
