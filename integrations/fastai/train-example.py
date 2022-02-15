## MNIST Example in fastai

## Note: this uses fastai version 1.0.38
## pip install fastai==1.0.38

from comet_ml import Experiment

import fastai
import fastai.vision
import glob
import os

# The model, also known as wrn_22:
model = fastai.vision.models.WideResNet(num_groups=3,
                                        N=3,
                                        num_classes=10,
                                        k=6,
                                        drop_p=0.)

## Get the MNIST_TINY dataset:
path = fastai.datasets.untar_data(fastai.datasets.URLs.MNIST_TINY)
print("data path:", path)

## Still too many for a CPU, so we trim it down to 10 in each category:
dirname = os.path.dirname(path)
for group in ["mnist_tiny/train/3/*.png",
              "mnist_tiny/train/7/*.png",
              "mnist_tiny/valid/3/*.png",
              "mnist_tiny/valid/7/*.png"]:
    for filename in glob.glob(os.path.join(dirname, group))[10:]:
        os.remove(filename)

experiment = Experiment(project_name="fastai")

## Now we get the image data from the folder:
data = fastai.vision.ImageDataBunch.from_folder(path, bs=10) # bs: batch size

if data.device.type == 'cpu':
    learn = fastai.basic_train.Learner(data, model, metrics=fastai.metrics.accuracy)
else: # GPU:
    learn = fastai.basic_train.Learner(data, model, metrics=fastai.metrics.accuracy).to_fp16()

with experiment.train():
    learn.fit_one_cycle(10, 3e-3, wd=0.4, div_factor=10, pct_start=0.5)
