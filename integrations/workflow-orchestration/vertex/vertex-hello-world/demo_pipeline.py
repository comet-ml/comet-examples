# coding: utf-8
import os

from comet_ml import init

import google.cloud.aiplatform as aip
import kfp
import kfp.v2.dsl as dsl

# Login to Comet if needed
init()


COMET_PROJECT_NAME = "comet-example-vertex-hello-world"


def data_preprocessing(a: str = None, b: str = None) -> str:
    import math
    import random
    import time

    import comet_ml

    experiment = comet_ml.Experiment()

    for i in range(60):
        experiment.log_metric("accuracy", math.log(i + random.random()))
        time.sleep(0.1)
    experiment.end()

    return a


def model_training(a: str = None, b: str = None) -> str:
    import math
    import random
    import time

    import comet_ml

    experiment = comet_ml.Experiment()

    for i in range(60):
        experiment.log_metric("accuracy", math.log(i + random.random()))
        time.sleep(0.1)
    experiment.end()

    return a


def model_evaluation(a: str = None, b: str = None) -> str:
    import math
    import random
    import time

    import comet_ml

    experiment = comet_ml.Experiment()

    for i in range(60):
        experiment.log_metric("accuracy", math.log(i + random.random()))
        time.sleep(0.1)
    experiment.end()

    return a


data_preprocessing_op = kfp.components.create_component_from_func(
    func=data_preprocessing, packages_to_install=["comet_ml"]
)

model_training_op = kfp.components.create_component_from_func(
    func=model_training, packages_to_install=["comet_ml"]
)

model_evaluation_op = kfp.components.create_component_from_func(
    func=model_evaluation, packages_to_install=["comet_ml"]
)


@dsl.pipeline(name="comet-integration-example")
def pipeline():
    import comet_ml.integration.vertex

    logger = comet_ml.integration.vertex.CometVertexPipelineLogger(
        # api_key=XXX,
        project_name=COMET_PROJECT_NAME,
        # workspace=XXX
        share_api_key_to_workers=True,
    )

    task_1 = logger.track_task(data_preprocessing_op("test"))

    task_2 = logger.track_task(model_training_op(task_1.output))

    task_3 = logger.track_task(model_training_op(task_1.output))

    _ = logger.track_task(model_evaluation_op(task_2.output, task_3.output))


if __name__ == "__main__":
    print("Running pipeline")
    kfp.v2.compiler.Compiler().compile(
        pipeline_func=pipeline, package_path="demo_pipeline.json"
    )

    job = aip.PipelineJob(
        display_name="comet-integration-example",
        template_path="demo_pipeline.json",
        pipeline_root=os.getenv("PIPELINE_ROOT"),
        project=os.getenv("GCP_PROJECT"),
        enable_caching=False,
    )

    job.submit()
