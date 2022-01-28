import comet_ml
import sys
import time

from comet_ml import Experiment, Optimizer

def run():  

    # access existing Optimizer object using key
    opt = Optimizer()   
    
    # loop over experiments in Optimizer generator
    for experiment in opt.get_experiments():
        x = experiment.get_parameter("x")
        experiment.log_parameter("x", x)
        print("Current hyperparameter value: ", x)

        # add model training functions here
        print("Training model")

        # add sleep to simulate training
        time.sleep(60)


if __name__ == '__main__':
    run()