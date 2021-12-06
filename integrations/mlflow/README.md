# MLFlow with Comet

The example `mlflow-with-comet.py` in this folder shows an example of including `import comet_ml` in an MLFlow script and then running as usual.

## Setup

Install dependencies

```bash
pip install -r requirements.txt
```

Set your API key
```
export COMET_API_KEY=<Your API Key>
```

## To Run

```python
python mlflow-with-comet.py
```

## Example Experiment

You can find an example of a completed run in this [Experiment](https://www.comet.ml/team-comet-ml/mlflow-demo/view/new/panels).

# Comet-for-MLFlow

For more information on using Comet's built-in, core support for MLFlow, please see:

https://www.comet.ml/docs/python-sdk/mlflow/

If you have previous MLFlow runs that you would like to visualize in Comet.ml, please see:

https://githib.com/comet-ml/comet-for-mlflow
