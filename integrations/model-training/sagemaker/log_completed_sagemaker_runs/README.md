# Logging Sagemaker runs to Comet

Comet supports transferring data from Sagemaker runs to Comet's Experiment Management tool. This approach requires no changes
to existing Sagemake code. It simply transfers data from completed Sagemaker runs to Comet.

There are three ways to log data from a completed Sagemaker Training Job to Comet

### 1. Using the Estimator object

Teh first method involves passing the Sagemaker `estimator` object directly into Comet's `log_sagemaker_training_job_v1` utility function.

```python
from comet_ml.integration.sagemaker import log_sagemaker_training_job_v1

COMET_API_KEY = "<Your Comet API Key>"
COMET_WORKSPACE = "<Your Comet Workspace>"
COMET_PROJECT_NAME =  "Your Comet Project Name"

log_sagemaker_training_job_v1(
    estimator,
    api_key=COMET_API_KEY,
    workspace=COMET_WORKSPACE,
    project_name=COMET_PROJECT_NAME
)
```

### 2. Using the Training Job Name

```python
from comet_ml.integration.sagemaker import log_sagemaker_training_job_v1

COMET_API_KEY = "<Your Comet API Key>"
COMET_WORKSPACE = "<Your Comet Workspace>"
COMET_PROJECT_NAME =  "Your Comet Project Name"

TRAINING_JOB_NAME = "<Your training job name>"

log_sagemaker_training_job_by_name_v1(
    TRAINING_JOB_NAME,
    api_key=COMET_API_KEY,
    workspace=COMET_WORKSPACE,
    project_name=COMET_PROJECT_NAME
)
```

### 3. Automatically log data from the last completed Training Job

```python
from comet_ml.integration.sagemaker import log_last_sagemaker_training_job_v1

COMET_API_KEY = "<Your Comet API Key>"
COMET_WORKSPACE = "<Your Comet Workspace>"
COMET_PROJECT_NAME =  "Your Comet Project Name"

log_last_sagemaker_training_job_v1(
    api_key=COMET_API_KEY,
    workspace=COMET_WORKSPACE,
    project_name=COMET_PROJECT_NAME
)
```

**Known Limitations:**

- Data transfer is only compatible with Training Jobs that use Sagemaker's [builtin algorithms](https://docs.aws.amazon.com/sagemaker/latest/dg/algos.html)
- This method only supports logging the following information from Sagemaker
  - Hyperparameters
  - Metrics
  - Sagemaker specific metadata (BillableTimeInSeconds, TrainingImage, etc)
  - Sagemaker notebook code
- Real time data logging is not supported from the Sagemaker job
- Metrics are logged based on wall clock time. Step/Epoch information is not captured


For more information, please refer to our [Sagemaker Documentation](https://www.comet.com/docs/v2/api-and-sdk/python-sdk/reference/integration.sagemaker/)


## Run the Example

To run this example, you will need both a Sagemaker account and a [Comet account](https://comet.com/signup)

1. Upload the `mnist.py` and `train_mnist.ipynb` to your Sagemaker Notebook instance.

2. Run the `train_mnist.ipynb` Notebook to create a Sagemaker Training Job and log the data to Comet.


## Example Project

Here is an example of a completed training run that has been logged from Sagemaker:

[Sagemaker Pytorch MNIST project](https://www.comet.com/examples/comet-example-sagemaker-completed-run-pytorch-mnist/fb5b85fa59b24110b9e786e4d237df91?experiment-tab=panels&showOutliers=true&smoothing=0&transformY=smoothing&xAxis=wall)

