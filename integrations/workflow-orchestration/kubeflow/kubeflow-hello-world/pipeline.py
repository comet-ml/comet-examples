# coding: utf-8

import os

from comet_ml import init

import kfp
import kfp.dsl as dsl
from kubernetes.client.models import V1EnvVar

# Login to Comet if needed
init()


COMET_PROJECT_NAME = "comet-example-kubeflow-hello-world"


def data_preprocessing(a: str = None, b: str = None) -> str:
    import math
    import random
    import time

    import comet_ml.integration.kubeflow

    workflow_uid = (
        "{{workflow.uid}}"  # The workflow uid is replaced at run time by Kubeflow
    )
    pod_name = "{{pod.name}}"
    experiment = comet_ml.Experiment()
    comet_ml.integration.kubeflow.initialize_comet_logger(
        experiment, workflow_uid, pod_name
    )

    for i in range(60):
        experiment.log_metric("accuracy", math.log(i + random.random()))
        time.sleep(0.1)
    experiment.end()

    return a


def model_training(a: str = None, b: str = None) -> str:
    import math
    import random
    import time

    import comet_ml.integration.kubeflow

    workflow_uid = (
        "{{workflow.uid}}"  # The workflow uid is replaced at run time by Kubeflow
    )
    pod_name = "{{pod.name}}"
    experiment = comet_ml.Experiment()
    comet_ml.integration.kubeflow.initialize_comet_logger(
        experiment, workflow_uid, pod_name
    )

    for i in range(60):
        experiment.log_metric("accuracy", math.log(i + random.random()))
        time.sleep(0.1)
    experiment.end()

    return a


def model_evaluation(a: str = None, b: str = None) -> str:
    import math
    import random
    import time

    import comet_ml.integration.kubeflow

    workflow_uid = (
        "{{workflow.uid}}"  # The workflow uid is replaced at run time by Kubeflow
    )
    pod_name = "{{pod.name}}"
    experiment = comet_ml.Experiment()
    comet_ml.integration.kubeflow.initialize_comet_logger(
        experiment, workflow_uid, pod_name
    )

    for i in range(60):
        experiment.log_metric("accuracy", math.log(i + random.random()))
        time.sleep(0.1)
    experiment.end()

    return a


data_preprocessing_op = kfp.components.create_component_from_func(
    func=data_preprocessing, packages_to_install=["comet_ml", "kfp"]
)


model_training_op = kfp.components.create_component_from_func(
    func=model_training, packages_to_install=["comet_ml", "kfp"]
)

model_evaluation_op = kfp.components.create_component_from_func(
    func=model_evaluation, packages_to_install=["comet_ml", "kfp"]
)


@dsl.pipeline(name="Basic Comet Pipeline")
def add_pipeline():
    import comet_ml.integration.kubeflow

    comet_api_key = os.getenv("COMET_API_KEY")
    comet_project_name = os.getenv("COMET_PROJECT_NAME", COMET_PROJECT_NAME)
    comet_workspace = os.getenv("COMET_WORKSPACE")

    comet_ml.integration.kubeflow.comet_logger_component(
        # api_key=XXX,
        project_name=COMET_PROJECT_NAME,
        # workspace="jacques-comet"
    )

    def add_comet_env(task):
        if comet_api_key:
            task = task.add_env_variable(
                V1EnvVar(name="COMET_API_KEY", value=comet_api_key)
            )

        if comet_project_name:
            task = task.add_env_variable(
                V1EnvVar(name="COMET_PROJECT_NAME", value=comet_project_name)
            )

        if comet_workspace:
            task = task.add_env_variable(
                V1EnvVar(name="COMET_WORKSPACE", value=comet_workspace)
            )

        return task

    task_1 = add_comet_env(data_preprocessing_op("test"))

    task_2 = add_comet_env(model_training_op(task_1.output))

    task_3 = add_comet_env(model_training_op(task_1.output))

    _ = add_comet_env(model_evaluation_op(task_2.output, task_3.output))


if __name__ == "__main__":
    print("Running pipeline")
    client = kfp.Client()

    client.create_run_from_pipeline_func(
        add_pipeline,
        arguments={},
        enable_caching=False,
        mode=kfp.dsl.PipelineExecutionMode.V2_COMPATIBLE,
    )
