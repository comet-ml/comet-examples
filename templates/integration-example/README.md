# {Framework} integration with Comet

<!-- TODO: one paragraph — what the framework is, and what instrumenting it with Comet gives you
     (experiment tracking, hyperparameter logging, reproducibility, collaboration). -->

Instrument {Framework} with Comet to start managing experiments, create dataset versions, and track
hyperparameters for faster, easier reproducibility and collaboration.

## Documentation

For more information on using and configuring the integration, see:
[https://www.comet.com/docs/v2/](https://www.comet.com/docs/v2/) <!-- TODO: link the specific integration page -->

## See it

<!-- TODO: link a public Comet project showing this example's runs, if one exists. -->

## Setup

Install dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Run the example

```bash
uv run python example_integration.py
```

No Comet account? Run it offline (logs to a local archive instead of the Comet UI):

```bash
COMET_MODE=offline uv run python example_integration.py
```
