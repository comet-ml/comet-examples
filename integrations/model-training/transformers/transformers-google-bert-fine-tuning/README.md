# Transformers integration with Comet.ml

[Hugging Face Transformers](https://github.com/huggingface/transformers) provide
general-purpose Machine Learning models for Natural Language
Understanding (NLP). Transformers give you easy access to pre-trained model
weights, and interoperability between PyTorch and TensorFlow.

Instrument Transformers with Comet to start managing experiments, create dataset versions and track hyperparameters for faster and easier reproducibility and collaboration.

Instrument Transformers with Comet to start managing experiments, create dataset versions and track hyperparameters for faster and easier reproducibility and collaboration.


## Documentation

For more information on using and configuring the Transformers integration, see: [https://www.comet.com/docs/v2/integrations/ml-frameworks/huggingface/](https://www.comet.com/docs/v2/integrations/ml-frameworks/transformers/?utm_source=comet-examples&utm_medium=referral&utm_campaign=github_repo_2023&utm_content=huggingface)

## See it

Take a look at this [public Comet Project](https://www.comet.com/examples/comet-example-transformers-google-bert-fine-tuning/25d673e1153047eda82096f74142e2d0?utm_source=comet-examples&utm_medium=referral&utm_campaign=github_repo_2023&utm_content=pytorch).

## Setup

Install dependencies

```bash
python -m pip install -r requirements.txt
```

## Run the example


This example shows how to use Comet in a HuggingFace Transformers script.


```bash
python transformers-distilbert-fine-tuning.py
```
