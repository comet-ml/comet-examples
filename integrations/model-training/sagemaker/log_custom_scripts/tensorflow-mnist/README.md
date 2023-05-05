# Using Comet with Tensorflow and Sagemaker

The preferred way to use Comet with Sagemaker is to add Comet to your script before launching your Sagemaker job.

Executing your training jobs in this manner has several advantages over migrating data from completed runs, including:

1. Being able to leverage Comet's auto-logging capabilities.
2. Supporting real-time reporting of metrics with step/epoch information.
3. Being able to take advantage of Comet's advanced logging capabilities, such as:
   - logging media (image, text, audio)
   - logging interactive confusion matrices
   - auto-logging system metrics (CPU/GPU usage)
   - auto-logging the model graph
   - logging models to Comet's model registry.

## Setup
In order to use a customized script with Sagemaker, you must create a directory to hold your training script and requirements file.

```shell
├── README.md
├── src
│   ├── mnist.py
│   └── requirements.txt
└── train_mnist.ipynb
```

Next, when creating the Estimator object specify your Comet credentials as environment variables.

```python
from sagemaker.tensorflow import TensorFlow

COMET_API_KEY = "<Your Comet API Key>"
COMET_WORKSPACE = "<Your Comet Workspace>"
COMET_PROJECT_NAME =  "<Your Comet Project Name>"

estimator = TensorFlow(
    source_dir="src",
    entry_point="mnist.py",
    role=role,
    framework_version="2.2",
    py_version="py37",
    environment={
        "COMET_API_KEY": COMET_API_KEY,
        "COMET_PROJECT_NAME": COMET_PROJECT_NAME,
        "COMET_WORKSPACE": COMET_WORKSPACE
    }
)
```

## Run the Example

To run this example, you will need both a Sagemaker account and a [Comet account](https://comet.com/signup)

1. Upload the contents of the `src` directory to your Sagemaker Notebook instance.
2. Upload `train_mnist.ipynb` to your Sagemake Notebook instance.
3. Run the `train_mnist.ipynb` Notebook to create a Sagemaker Training Job and log the data to Comet.

## Example Project

Here is an example of a completed training run that has been logged from Sagemaker:

[Sagemaker Tensorflow MNIST project](https://www.comet.com/examples/comet-example-sagemaker-tensorflow-custom-mnist/3766c3d4519844509ca4dab662730598?experiment-tab=panels&showOutliers=true&smoothing=0&transformY=smoothing&xAxis=step)
