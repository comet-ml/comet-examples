# -*- coding: utf-8 -*-
import json

from comet_ml import Experiment, init

import numpy as np
import shap
import tensorflow.keras.backend as K
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

# Login to Comet if needed
init()

experiment = Experiment(project_name="comet-example-shap-hello-world")

# load pre-trained model and choose two images to explain
model = VGG16(weights="imagenet", include_top=True)

X, y = shap.datasets.imagenet50()
to_explain = X[[39, 41]]

# load the ImageNet class names
url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"  # noqa: E501
fname = shap.datasets.cache(url)
with open(fname) as f:
    class_names = json.load(f)

# explain how the input to the 7th layer of the model explains
# the top two classes


def map2layer(x, layer):
    feed_dict = dict(zip([model.layers[0].input], [preprocess_input(x.copy())]))
    return K.get_session().run(model.layers[layer].input, feed_dict)


e = shap.GradientExplainer(
    (model.layers[7].input, model.layers[-1].output),
    map2layer(preprocess_input(X.copy()), 7),
)

shap_values, indexes = e.shap_values(
    map2layer(to_explain, 7),
    ranked_outputs=2,
    nsamples=10,
)

# get the names for the classes
index_names = np.vectorize(lambda x: class_names[str(x)][1])(indexes)

# plot the explanations
shap.image_plot(shap_values, to_explain, index_names)

# explain how the input to the 7th layer of the model explains the top two classes
explainer = shap.GradientExplainer(
    (model.layers[7].input, model.layers[-1].output),
    map2layer(preprocess_input(X.copy()), 7),
    local_smoothing=100,
)
shap_values, indexes = explainer.shap_values(
    map2layer(to_explain, 7),
    ranked_outputs=2,
    nsamples=10,
)

# get the names for the classes
index_names = np.vectorize(lambda x: class_names[str(x)][1])(indexes)

# plot the explanations
shap.image_plot(shap_values, to_explain, index_names)
