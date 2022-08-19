# Vertex AI integration with Comet.ml

Comet integrates with Google Vertex AI.

[Google Vertex AI](https://cloud.google.com/vertex-ai/?utm_source=google&utm_medium=cpc&utm_campaign=emea-il-all-en-dr-skws-all-all-trial-e-gcp-1011340&utm_content=text-ad-none-any-DEV_c-CRE_574561340133-ADGP_Hybrid%20%7C%20SKWS%20-%20EXA%20%7C%20Txt%20~%20%20AI%20%26%20ML%20~%20Vertex%20AI-KWID_43700066526085804-kwd-553582750299-userloc_1007973&utm_term=KW_vertex%20ai-NET_g-PLAC_&gclid=CjwKCAjwoduRBhA4EiwACL5RP84_0P9FaDWoXsVlA3FOCozwNkqqNaZZFIjaWgadRde-KbCnfZRduRoCCzwQAvD_BwE&gclsrc=aw.ds) lets you build, deploy, and scale ML models faster, with pre-trained and custom tooling within a unified artificial intelligence platform.

## Documentation

For more information on using and configuring the Vertex integration, see: https://www.comet.com/docs/v2/integrations/third-party-tools/vertex-ai/

## See it

Take a look at this [public Comet Project](https://www.comet.com/examples/comet-example-vertex-hello-world/).

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
