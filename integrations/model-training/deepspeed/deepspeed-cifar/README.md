# DeepSpeed integration with Comet.ml

[DeepSpeed](https://github.com/microsoft/DeepSpeed) empowers ChatGPT-like model training with a single click, offering 15x speedup over SOTA RLHF systems with unprecedented cost reduction at all scales.

Instrument your runs with Comet to start managing experiments, create dataset versions and track hyperparameters for faster and easier reproducibility and collaboration.

[Find more information about our integration with DeepSpeed](https://www.comet.ml/docs/v2/integrations/ml-frameworks/deepspeed/)

## Documentation

For more information on using and configuring the DeepSpeed integration, see: [https://www.comet.com/docs/v2/integrations/ml-frameworks/deepspeed/](https://www.comet.com/docs/v2/integrations/ml-frameworks/deepspeed/)

## See it

Take a look at this [public Comet Project](https://www.comet.com/examples/comet-example-deepspeed-cifar/).

## Setup

Install dependencies

```bash
python -m pip install -r requirements.txt
```

## Run the example

This example is based on official example from [DeepSpeed](https://github.com/microsoft/DeepSpeedExamples/tree/master/training/cifar).


```bash
deepspeed --bind_cores_to_rank cifar10_deepspeed.py
```
