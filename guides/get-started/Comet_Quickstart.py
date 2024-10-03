#!/usr/bin/env python
# coding: utf-8

# 💥 Login to Comet and grab your Credentials

import comet_ml

comet_ml.login(project_name="comet-example-intro-to-comet")


# 🚀 Let's start logging Experiments!

# A Comet Experiment is a unit of measurable research that defines a single
# run with some data/parameters/code/metrics.

experiment = comet_ml.Experiment()


# Comet supports logging metrics, parameters, source code, system information,
# models and media. You name it, we can log it!

# In the sections below, we will walkthrough the basic methods for logging
# data to Comet. In addition to these methods, Comet also supports
# auto-logging data based on the framework you are using. This means that once
# you have created the Experiment object in your code, you can run it as is,
# and Comet will take care of the logging for you!

# If Auto-Logging isn't enough, Comet is infinitely customizable to your
# specific needs!

# Learn more about Auto-Logging:
# https://www.comet.com/docs/v2/guides/experiment-management/log-data/overview/#automated-logging

# Logging Metrics

metrics = {"accuracy": 0.65, "loss": 0.01}
experiment.log_metrics(metrics)


# Logging Metrics Over Time

for step, value in enumerate(range(0, 100)):
    metrics = {"train/accuracy": value / 10, "validation/accuracy": value / 20}
    experiment.log_metrics(metrics, step=step)


# Logging Parameters

parameters = {"batch_size": 32, "num_samples": 10000}
experiment.log_parameters(parameters)


# End the Experiment

experiment.end()
