# Kubeflow integration with Comet.ml

Comet integrates with Kubeflow.

[Kubeflow](https://github.com/kubeflow/kubeflow) is an open-source machine learning platform that enables using machine learning pipelines to orchestrate complicated workflows running on Kubernetes.

## Documentation

For more information on using and configuring the Kubeflow integration, see: https://www.comet.com/docs/v2/integrations/third-party-tools/kubeflow/

## See it

Take a look at this [public Comet Project](https://www.comet.com/examples/comet-example-kubeflow-hello-world/).

## Setup

Install dependencies

```bash
python -m pip install -r requirements.txt
```

## Run the example

The following example demonstrates how to use the Comet pipelines integration to track the state of pipelines run on Kubeflow. Before running, make sure that you have access to a [Kubeflow environment](https://www.kubeflow.org/docs/started/installing-kubeflow/) or that you have [installed Kubeflow locally](https://www.kubeflow.org/docs/components/pipelines/installation/localcluster-deployment/).

```bash
python pipeline.py
```
