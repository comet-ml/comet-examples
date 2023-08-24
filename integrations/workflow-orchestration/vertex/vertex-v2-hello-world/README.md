# Vertex AI integration with Comet.ml

Comet integrates with Google Vertex AI.

[Google Vertex AI](https://cloud.google.com/vertex-ai/) lets you build, deploy, and scale ML models faster, with pre-trained and custom tooling within a unified artificial intelligence platform.

## Documentation

For more information on using and configuring the Vertex integration, see: [https://www.comet.com/docs/v2/integrations/third-party-tools/vertex-ai/](https://www.comet.com/docs/v2/integrations/third-party-tools/vertex-ai/?utm_source=comet-examples&utm_medium=referral&utm_campaign=github_repo_2023&utm_content=vertex)

## See it

Take a look at this [public Comet Project](https://www.comet.com/examples/comet-example-vertex-hello-world/?utm_source=comet-examples&utm_medium=referral&utm_campaign=github_repo_2023&utm_content=vertex).

## Setup

Install dependencies

```bash
python -m pip install -r requirements.txt
```

## Run the example

The following example demonstrates how to use the Comet pipelines integration to track the state of pipelines run on Vertex. Before running, make sure that you are correctly authenticated against your Google Cloud Platform account and project, the easiest way to do so is by using the [Google Cloud CLI](https://cloud.google.com/sdk/docs/).

```bash
python demo_pipeline.py
```
