# FastAI integration with Comet.ml

[fastai](https://github.com/fastai/fastai) is a deep learning library which provides practitioners with high-level components that can quickly and easily provide state-of-the-art results in standard deep learning domains, and provides researchers with low-level components that can be mixed and matched to build new approaches.

Instrument fastai with Comet to start managing experiments, create dataset versions and track hyperparameters for faster and easier reproducibility and collaboration.

## Documentation

For more information on using and configuring the fastai integration, see: [https://www.comet.com/docs/v2/integrations/ml-frameworks/fastai/](https://www.comet.com/docs/v2/integrations/ml-frameworks/fastai/?utm_source=comet-examples&utm_medium=referral&utm_campaign=github_repo_2023&utm_content=fastai)

## See it

Take a look at this [public Comet Project](https://www.comet.com/examples/comet-examples-fastai-hello-world/view/new/panels?utm_source=comet-examples&utm_medium=referral&utm_campaign=github_repo_2023&utm_content=fastai).

## Setup

Install dependencies

```bash
python -m pip install -r requirements.txt
```

## Run the example

This example is fine-tuning a pre-trained resnet 28 model on the Mnist Tiny dataset for 5 epochs:


```bash
python fastai-hello-world.py
```
