"""
Trains a simple deep NN on the MNIST dataset.

Assumes the COMET_API_KEY is defined as an environment variable
or in the file .env.

Based on:
https://github.com/chainer/chainer/blob/master/examples/mnist/train_mnist.py

Requires:
    dot - operating system executable
    pydot - python package
    matplotlib, mpl_toolkits - standard python package
    numpy - standard python package
"""

from comet_ml import Experiment

import argparse
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer.training import extension
from chainer.training import trigger as trigger_module

import pydot
import logging
import itertools
import os
import io

LOGGER = logging.getLogger(__name__)

class MLP(chainer.Chain):
    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_units)
            self.l2 = L.Linear(None, n_units)
            self.l3 = L.Linear(None, n_out)

    def forward(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

class CometChainerExtension(extension.Extension):
    """
    Report metrics back to comet.ml.
    """
    def __init__(self, experiment, model, function, trigger=(1, 'epoch')):
        self.experiment = experiment
        self.model = model
        self.function = function
        self._trigger = trigger_module.get_trigger(trigger)

    def __call__(self, trainer):
        if self._trigger(trainer):
            epoch = trainer.updater.epoch
            step = trainer.updater.iteration
            self.function(self.experiment, self.model, trainer, epoch, step)

def comet_logger(experiment, model, trainer, epoch, step):
    experiment.log_current_epoch(epoch)
    for metric_name, key in [("loss", 'main/loss'),
                             ("val_loss", 'validation/main/loss'),
                             ("acc", "main/accuracy"),
                             ("val_acc", "validation/main/accuracy")]:
        value = trainer.observation[key]
        if isinstance(value, (int, float)):
            experiment.log_metric(metric_name,
                                  value,
                                  step=step)
        else:
            experiment.log_metric(metric_name,
                                  value.data.item(),
                                  step=step)

def log_confusion_matrix(experiment, model, trainer, epoch, step):
    filename = "confusion_matrix_%d.png" % epoch
    title = "MNIST Confusion Matrix Epoch #%d" % epoch
    ## First, build the matrix:
    matrix = [[0 for i in range(10)] for j in range(10)]
    for i in range(len(test)):
        output, target = (model.predictor(test[i][0][None]).data.argmax(),
                          test[i][1])
        matrix[target][output] += 1
    make_confusion_matrix(matrix, range(10), normalize=False,
                          data_format="",
                          filename=filename,
                          title=title)
    experiment.log_image(filename)

def make_confusion_matrix(matrix, labels=None, title="Confusion Matrix",
                          annotate=True, colormap="Blues", colorbar=True,
                          colorbar_orientation='vertical',
                          normalize=True, size=(8,6), interpolation='nearest',
                          xlabel=None, xlabel_rotate=45, ylabel=None,
                          ylabel_rotate=0, data_format="0.2f",
                          filename="confusion_matrix.png"):
    """
    Create a confusion matrix image.

    Arguments:
        matrix: an N x N confusion matrix given as either counts, or accuracies
        labels: the labels for each category. Must match the order of matrix
        title: title of figure, or None
        annotate: if True, then display the matrix value in cell
        colormap: None, or a valid matplotlib colormap name
        colorbar: if True, display a colorbar
        colorbar_orientation: 'vertical' or 'horizontal'
        normalize: if True, then matrix values are the taken as counts;
            otherwise, matrix values are taken as raw values
        size: size of the figure, in inches
        interpolation: method for mapping matrix values to color. Valid values
            are: None, 'nearest', 'none', 'bilinear', 'bicubic', 'spline16',
           'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
           'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', and 'lanczos'
        xlabel: title for the y-axis
        xlabel_rotate: degrees to rotate the x-labels
        ylabel: title for the y-axis
        ylabel_rotate: degrees to rotate the y-labels
        data_format: the Python format, e.g. '0.4f' or None
        filename: name of filename to save image to, or None. Should end in
            'png', 'jpg', etc. if given. If given, can begin with "~" meaning
            HOME.
    """
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        from matplotlib.figure import Figure
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        import numpy as np
    except:
        LOGGER.info("Unable to create confusion matrix: please install:\n"
                    "matplotlib, mpl_toolkits, and numpy")
        return

    matrix = np.array(matrix)
    fig = Figure(figsize=size, frameon=False)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    im = ax.imshow(matrix, interpolation=interpolation, cmap=colormap)
    if colorbar:
        layout = make_axes_locatable(ax)
        if colorbar_orientation == "vertical":
            cax = layout.append_axes('right', size='5%', pad=0.05)
        else:
            cax = layout.append_axes('top', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax,
                     orientation=colorbar_orientation)
    if labels is not None:
        tick_marks = np.arange(len(labels))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(labels, rotation=xlabel_rotate)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(labels, rotation=ylabel_rotate)
    if normalize:
        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if annotate:
        thresh = matrix.mean()
        for i, j in itertools.product(range(matrix.shape[0]),
                                      range(matrix.shape[1])):
            if data_format:
                str_format = "{:" + data_format + "}"
            else:
                str_format = "{:,}"
            ax.text(j, i, str_format.format(matrix[i, j]),
                    horizontalalignment="center",
                    color="white" if matrix[i, j] > thresh else "black")
    fig.tight_layout()
    filename = os.path.expanduser(filename)
    LOGGER.debug("Saving confusion_matrix image to '%s'..." % filename)
    fig.savefig(filename)
    return fig

experiment = Experiment(project_name="chainer")

parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--batchsize', '-b', type=int, default=100,
                    help='Number of images in each mini-batch')
parser.add_argument('--epoch', '-e', type=int, default=20,
                    help='Number of sweeps over the dataset to train')
parser.add_argument('--frequency', '-f', type=int, default=-1,
                    help='Frequency of taking a snapshot')
parser.add_argument('--gpu', '-g', type=int, default=-1,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--out', '-o', default='result',
                    help='Directory to output the result')
parser.add_argument('--resume', '-r', default='',
                    help='Resume the training from snapshot')
parser.add_argument('--unit', '-u', type=int, default=1000,
                    help='Number of units')
parser.add_argument('--noplot', dest='plot', action='store_false',
                    help='Disable PlotReport extension')
args = parser.parse_args()

for arg in ["batchsize", "epoch", "frequency", "gpu", "out", "resume",
            "unit"]:
    experiment.log_parameter(arg, getattr(args, arg))
    print(arg, ":", getattr(args, arg))
print('')

model = L.Classifier(MLP(args.unit, 10))
if args.gpu >= 0:
    chainer.backends.cuda.get_device_from_id(args.gpu).use()
    model.to_gpu()

optimizer = chainer.optimizers.Adam()
optimizer.setup(model)

train, test = chainer.datasets.get_mnist()
experiment.log_dataset_hash(train)

train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                             repeat=False, shuffle=False)

updater = training.updaters.StandardUpdater(
    train_iter, optimizer, device=args.gpu)
trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
## Extensions:
trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.dump_graph('main/loss'))
trainer.extend(CometChainerExtension(experiment, model, comet_logger))
trainer.extend(CometChainerExtension(experiment, model, log_confusion_matrix))

if args.plot and extensions.PlotReport.available():
    trainer.extend(
        extensions.PlotReport(['main/loss', 'validation/main/loss'],
                              'epoch', file_name='loss.png'))
    trainer.extend(
        extensions.PlotReport(
            ['main/accuracy', 'validation/main/accuracy'],
            'epoch', file_name='accuracy.png'))

trainer.extend(extensions.PrintReport(
    ['epoch', 'main/loss', 'validation/main/loss',
     'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

# Print a progress bar to stdout
trainer.extend(extensions.ProgressBar())

if args.resume:
    # Resume from a snapshot
    chainer.serializers.load_npz(args.resume, trainer)

# Get confusion matrix picture before training:
log_confusion_matrix(experiment, model, trainer, 0, 0)

# Run the training
trainer.run()

# Report created images to comet.ml:
## If you want to include a graph made by chainer, you can:
#if args.plot and extensions.PlotReport.available():
#    experiment.log_image('result/loss.png')
#    experiment.log_image('result/accuracy.png')

# Report the graph, as dot language:
(graph,) = pydot.graph_from_dot_file('result/cg.dot')
graph.write_png('result/cg.png')
experiment.log_image('result/cg.png')
with open("result/cg.dot") as fp:
    desc = fp.readlines()
    experiment.set_model_graph("\n".join(desc))

# Report a URL:
experiment.log_html_url("https://github.com/chainer/chainer/"
                        "blob/master/examples/mnist/train_mnist.py",
                        label="This MNIST example is based on")
