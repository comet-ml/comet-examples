# Vertex AI pipelines

The following example demonstrates how to use the Comet pipelines integration to track the state of pipelines run on Vertex. You can learn more about the integration [here](https://www.comet.ml/docs/partnerships/vertex-ai/)

The pipeline can be run using:
```
# Vertex pipelines environment variables
export PIPELINE_ROOT=""
export GCP_PROJECT=""

# Comet environment variables
export COMET_API_KEY=""
export COMET_WORKSPACE=""
export COMET_PROJECT_NAME=""

# Running the pipeline
python demo_pipeline.py
```