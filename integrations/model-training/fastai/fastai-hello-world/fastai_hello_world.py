# coding: utf-8
import comet_ml

from fastai.vision.all import (
    Categorize,
    Datasets,
    GrandparentSplitter,
    IntToFloatTensor,
    PILImageBW,
    ToTensor,
    URLs,
    error_rate,
    get_image_files,
    parent_label,
    resnet18,
    untar_data,
    vision_learner,
)

EPOCHS = 5

comet_ml.init(project_name="comet-examples-fastai-hello-world")
experiment = comet_ml.Experiment()

path = untar_data(URLs.MNIST_TINY)

items = get_image_files(path)
tds = Datasets(
    items,
    [PILImageBW.create, [parent_label, Categorize()]],
    splits=GrandparentSplitter()(items),
)
dls = tds.dataloaders(after_item=[ToTensor(), IntToFloatTensor()])

learn = vision_learner(dls, resnet18, pretrained=True, metrics=error_rate)

with experiment.train():
    learn.fit_one_cycle(EPOCHS)

experiment.end()
