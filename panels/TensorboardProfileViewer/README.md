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

#### Example

This example logs some dummy data to Tensorflow, and
then logs the Tensorflow folder to Comet.

```python
import comet_ml
import tensorflow as tf
import tensorflow_datasets as tfds
from datetime import datetime
from packaging import version
import os

comet_ml.login()
tfds.disable_progress_bar()

experiment = comet_ml.Experiment(project_name="tensorboard-profile")
device_name = tf.test.gpu_device_name()
print('Found GPU at: {}'.format(device_name))

(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label

ds_train = ds_train.map(normalize_img)
ds_train = ds_train.batch(128)

ds_test = ds_test.map(normalize_img)
ds_test = ds_test.batch(128)

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=['accuracy']
)

# Create a TensorBoard callback
logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

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

experiment.log_tensorflow_folder("./logs")
experiment.end()
```

#### Python Panel

To include this panel from the github repo, use this code in a Custom Python Panel:

```
%include https://raw.githubusercontent.com/comet-ml/comet-examples/refs/heads/master/panels/TensorboardProfileViewer/TensorboardProfileViewer.py
```

Or, you can simply [copy the code](https://raw.githubusercontent.com/comet-ml/comet-examples/refs/heads/master/panels/TensorboardProfileViewer/TensorboardProfileViewer.py) into a custom Python Panel.

#### How it works

The Python panel will start a Tensorboard server and make available
the logs from the experiment that is selected.

#### Resources

* Example Comet Project: [www.comet.com/dsblank/tensorboard-profile](https://www.comet.com/dsblank/tensorboard-profile/)
* Documentation:
  * [Logging tensorflow folders](https://www.comet.com/docs/v2/api-and-sdk/python-sdk/reference/Experiment/#comet_ml.Experiment.log_tensorflow_folder)
  * [Automatic Tensorboard logging](https://www.comet.com/docs/v2/integrations/third-party-tools/tensorboard/#configure-comet-for-tensorboard)
  * [Download tensorboard folders](https://www.comet.com/docs/v2/api-and-sdk/python-sdk/reference/APIExperiment/#comet_ml.APIExperiment.download_tensorflow_folder)
