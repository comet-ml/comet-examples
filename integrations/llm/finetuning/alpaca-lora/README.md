# Finetuning Alpaca-Lora with Comet

The [Alpaca LoRA](https://github.com/tloen/alpaca-lora/tree/main) repository is built with Hugging Face Transformers, which means Comet logging is available right out of the box when finetuning the model.

In this guide, we will demonstrate how you can configure Comet to log the results of your finetuning run.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/comet-ml/comet-examples/blob/master/integrations/llm/finetuning/alpaca-lora/notebooks/Alpaca_Lora_Finetuning_with_Comet.ipynb)


## Setup

### Setup the Alpaca-LoRA repository

```shell
git clone https://github.com/tloen/alpaca-lora.git
cd alpaca-lora/ && pip install -r requirements.txt
```

### Install Comet

```shell
pip install comet_ml
```

### Configure your Comet Credentials

```shell
export COMET_API_KEY="Your Comet API Key"
export COMET_PROJECT_NAME="Your Comet Project Name"
```

## Run the finetuning script

```shell
python finetune.py \
    --base_model 'decapoda-research/llama-7b-hf' \
    --data_path 'yahma/alpaca-cleaned' \
    --output_dir './lora-alpaca'
```

## Try it out!

Finetune an Alpaca model using Colab. Try it out here.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/comet-ml/comet-examples/blob/master/integrations/llm/finetuning/alpaca-lora/notebooks/Alpaca_Lora_Finetuning_with_Comet.ipynb)

Can't wait? See a completed [experiment here](https://www.comet.com/team-comet-ml/comet-example-alpaca-lora-finetuning/3709d2137e1f410e89648ff926a5dd0a?experiment-tab=panels&showOutliers=true&smoothing=0&xAxis=step)