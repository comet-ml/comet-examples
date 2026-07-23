<img src="https://www.comet.com/images/logo_comet_light.png" width="350" alt="Drawing" style="width: 350px;"/>

## Comet for Machine Learning Experiment Management
**Our Mission:** Comet is the AI developer platform. We give teams experiment tracking for their datasets, code changes, experiment history, and production models, and we make [Opik](https://github.com/comet-ml/opik), the open-source LLM observability and evaluation platform.

We all strive to be data driven and yet every day valuable experiment results are lost and forgotten. Comet provides a dead simple way of fixing that. It works with any workflow, any ML task, any machine, and any piece of code.

## Looking for LLM or AI agent examples?

Comet also makes [Opik](https://github.com/comet-ml/opik), the open-source LLM observability and evaluation platform. For LLM tracing, evaluation, and AI agent observability examples, see the [`opik/`](./opik) folder in this repo and the dedicated [opik-examples](https://github.com/comet-ml/opik-examples) repository. Full Opik documentation: https://www.comet.com/docs/opik/

## Examples Repository

This repository contains examples of using Comet in many Machine Learning Python libraries, including fastai, torch, sklearn, chainer, caffe, keras, tensorflow, mxnet, Jupyter notebooks, and with plain Python.

If you don't see something you need, just let us know! See contact methods below.

## Repository structure

```
comet-examples/
├── integrations/   # Add Comet to a framework, grouped by ML task
│   ├── model-training/        # PyTorch, Keras, fastai, scikit-learn, XGBoost, Transformers, …
│   ├── model-evaluation/
│   ├── model-optimization/
│   ├── model-deployment/
│   ├── workflow-orchestration/
│   ├── reinforcement-learning/
│   ├── llm/
│   └── data-management/
├── opik/           # Opik (LLM observability) examples: tracing + offline evaluation
├── guides/         # How-to notebooks for Comet workflows
├── panels/         # Custom Comet panel (visualization) examples
├── notebooks/      # General/standalone notebooks
└── templates/      # Starter template for new examples
```

New examples live under `integrations/<category>/<framework>/<example-name>/`.

## Documentation
[![PyPI version](https://badge.fury.io/py/comet-ml.svg)](https://badge.fury.io/py/comet-ml)

Full documentation and additional training examples are available on http://www.comet.com/docs/v2

## Installation

- [Sign up for free!](https://www.comet.com/signup)

- **Install Comet from PyPI:**

```sh
pip install comet_ml
```
Comet Python SDK is compatible with: __Python 3.5-3.13__.

## Tutorials + Examples

- [fastai](https://github.com/comet-ml/comet-examples/tree/master/integrations/model-training/fastai/)
- [keras](https://github.com/comet-ml/comet-examples/tree/master/integrations/model-training/keras/)
- [pytorch](https://github.com/comet-ml/comet-examples/tree/master/integrations/model-training/pytorch/)
- [scikit](https://github.com/comet-ml/comet-examples/tree/master/integrations/model-training/scikit-learn/)
- [tensorflow](https://github.com/comet-ml/comet-examples/tree/master/integrations/model-training/tensorflow/)
- [transformers](https://github.com/comet-ml/comet-examples/tree/master/integrations/model-training/transformers/)

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow, standards, and
PR checklist, and [AGENTS.md](AGENTS.md) for the conventions AI coding agents should follow. New
examples can be scaffolded from [`templates/integration-example/`](templates/integration-example/).

## Support
Have questions? We have answers -
- Email us at <info@comet.com>
- For the fastest response, ping us on [Slack](https://chat.comet.com/)

**Want to request a feature?**
We take feature requests through github at: https://github.com/comet-ml/issue-tracking

## Feature Spotlight
Check out new product features and updates through our [Release Notes](https://www.comet.com/docs/v2/api-and-sdk/python-sdk/releases/). Also check out our [blog](https://www.comet.com/site/blog/).
