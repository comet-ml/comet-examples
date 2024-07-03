# DeepSpeed integration with Comet.ml

[DeepSpeed](https://github.com/microsoft/DeepSpeed) empowers ChatGPT-like model training with a single click, offering 15x speedup over SOTA RLHF systems with unprecedented cost reduction at all scales.



## Documentation

For more information on using and configuring the DeepSpeed integration, see: [https://www.comet.com/docs/v2/integrations/ml-frameworks/deepspeed/](https://www.comet.com/docs/v2/integrations/ml-frameworks/deepspeed/)

## See it

Take a look at this [public Comet Project](TODO).

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
