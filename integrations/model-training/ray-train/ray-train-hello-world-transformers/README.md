# Ray-Train integration with Comet.ml

[Ray Train](https://docs.ray.io/en/latest/train/train.html) scales model training for popular ML frameworks such as Torch, XGBoost, TensorFlow, and more. It seamlessly integrates with other Ray libraries such as Tune and Predictors.

Comet integrates with Ray Train by allowing you to easily monitor the resource usage of all of your workers, making sure you are fully using your expensive GPUs and that your CPUs are not the bottleneck in your training.

## Documentation

For more information on using and configuring the Ray-Train integration, see: [https://www.comet.com/docs/v2/integrations/ml-frameworks/ray/#ray-train](https://www.comet.com/docs/v2/integrations/ml-frameworks/ray/#ray-train/?utm_source=comet-examples&utm_medium=referral&utm_campaign=github_repo_2023&utm_content=ray-train)

## See it

Take a look at this [public Comet Project](https://www.comet.com/examples/comet-example-ray-train-hugginface-transformers/).

## Setup

Install dependencies

```bash
python -m pip install -r requirements.txt
```

## Run the example


```bash
python Comet_with_ray_train_huggingface_transformers.py
```
