# Tranformers with Comet

The example `transformers_example.py` in this folder shows an example of including `import comet_ml` in a HuggingFace Transformers script and then running as usual.

It is recommended to set enviornment variables before running the code with the following commands:
For Linux/Mac:
```shell
export COMET_API_KEY=YOUR_KEY
export COMET_PROJECT_NAME=PROJECT_NAME
```

For Windows:
```batch
set COMET_API_KEY=YOUR_KEY
set COMET_PROJECT_NAME=PROJECT_NAME
```

To run:

```python
python transformers_example.py
```

For more information on using Comet's built-in, core support for Transformers, please see:

https://www.comet.ml/docs/python-sdk/huggingface/

Link to example project: https://www.comet.ml/comet-integrations/comet-transformers-example/view/new/panels
