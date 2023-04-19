# Metaflow integration with Comet.ml

Comet integrates with [Metaflow](https://metaflow.org/).

Metaflow is a human-friendly Python/R library that helps scientists and engineers build and manage real-life data science projects. Metaflow was originally developed at Netflix to boost productivity of data scientists who work on a wide variety of projects from classical statistics to state-of-the-art deep learning.

## Documentation

For more information on using and configuring Metaflow integration, please see: [https://www.comet.com/docs/v2/integrations/third-party-tools/metaflow/](https://www.comet.com/docs/v2/integrations/third-party-tools/metaflow/?utm_source=github.com&utm_medium=referral&utm_campaign=github_repo_2023&utm_content=metaflow)

## See it

Take a look at this [public Comet Project](https://www.comet.com/examples/comet-example-metaflow-regression?utm_source=github.com&utm_medium=referral&utm_campaign=github_repo_2023&utm_content=metaflow).

## Setup

Install dependencies

```bash
python -m pip install -r requirements.txt
```

## Run the example

This example is a Metaflow example training three Regression models on the same toy dataset. The models prediction is then logged as an interactive Plotly chart and saved as a Metaflow card.

```bash
python metaflow-regression-example.py run
```
