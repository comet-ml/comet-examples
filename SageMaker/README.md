<img src="https://comet.ml/images/logo_comet_light.png" width="350" alt="Drawing" style="width: 350px;"/>

## SageMaker Integration with Comet.ml

Comet's SageMaker integration is available to Enterprise customers only. If you are interested in learning more about Comet Enterprise, or are in a trial period with Comet.ml and would like to evaluate the SageMaker integration, please email support@comet.ml and credentials can be shared to download the correct packages.

## Examples Repository

This repository contains examples of using Comet.ml with SageMaker built-in Algorithms Linear Learner and Random Cut Forests. 


## Documentation

Full [documentation](http://www.comet.ml/docs/) and additional training examples are available on our website. 


## Installation

Please contact us for installation instructions.

## Configuration

The SageMaker integration is following the [Comet.ml Python SDK configuration](http://docs.comet.ml/python-sdk/advanced/#python-configuration) for configuring your Rest API Key, your workspace and project_name for created experiments. It's also following the [Boto configuration](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/configuration.html) to find your SageMaker training jobs.

## Logging SageMaker training runs to Comet

Below find three different ways you can log your SageMaker jobs to Comet: with an existing regressor/estimator object, with a SageMaker Job Name, or with the last SageMaker job. 

### comet_ml_sagemaker.log_sagemaker_job(estimator/regressor, api_key, workspace, project_name)
Logs a Sagemaker job based on an estimator/regressor object 

* estimator/regressor = Sagemaker estimator/regressor object
* api_key = your Comet REST API key
* workspace = your Comet workspace
* project_name = your Comet project_name

### comet_ml_sagemaker.log_sagemaker_job_by_name(job_name, api_key, workspace, project_name)
Logs a specific Sagemaker training job based on the jobname from the Sagemaker SDK.

* job_name = Cloudwatch/Sagemaker training job name
* api_key = your Comet REST API key
* workspace = your Comet workspace
* project_name = your Comet project_name

### comet_ml_sagemaker.log_last_sagemaker_job(api_key, workspace, project_name)
Will log the last *started* Sagemaker training job based on the current config.

* api_key = your Comet REST API key
* workspace = your Comet workspace
* project_name = your Comet project_name


## Tutorials + Examples
- [Linear Learner](Linear_example.ipynb)
- [Random Cut Forests](random_forest.ipynb)	


## Support 
Have questions? We have answers - 
- Try checking our [FAQ Page](https://www.comet.ml/faq)
- Email us at <info@comet.ml>
- For the fastest response, ping us on [Slack](https://join.slack.com/t/cometml/shared_invite/enQtMzM0OTMwNTQ0Mjc5LTM4ZDViODkyYTlmMTVlNWY0NzFjNGQ5Y2Q1Y2EwMjQ5MzQ4YmI2YjhmZTY3YmYxYTYxYTNkYzM4NjgxZmJjMDI)


## Feature Spotlight
Check out new product features and updates through our [Release Notes](https://www.notion.so/cometml/Comet-ml-Release-Notes-93d864bcac584360943a73ae9507bcaa). Also checkout our articles on [Medium](https://medium.com/comet-ml).

