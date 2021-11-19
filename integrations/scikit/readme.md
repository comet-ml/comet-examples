# Comet and Scikit-learn models

An example for using Comet Experiment Management with Scikit-learn models. 

The two examples include 1) a classification problem using the breast cancer wisconsin dataset, and 2) an NLP classification problem using the 20 newsgroups dataset.

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

```
python comet-scikit-classification-example.py

python comet-scikit-nlp-example.py
```

## Example Experiment
You can find an example of a completed run for the classification model in this [Experiment](https://www.comet.ml/team-comet-ml/scikit-learn-classification/3f482b8afe694315a07680b4fcd6c678?experiment-tab=chart&showOutliers=true&smoothing=0&transformY=smoothing&xAxis=wall)


You can find an example of a completed run for the NLP model in this [Experiment](https://www.comet.ml/team-comet-ml/scikit-learn-nlp/169f62f21c04418d9a11dee9886fd83c?experiment-tab=chart&showOutliers=true&smoothing=0&transformY=smoothing&xAxis=wall)