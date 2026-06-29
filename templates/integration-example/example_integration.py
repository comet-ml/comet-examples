# coding: utf-8
import comet_ml

# WHY: comet_ml.login() reads COMET_API_KEY from the environment — never hardcode it.
comet_ml.login(project_name="comet-example-example-integration")
experiment = comet_ml.start()

# TODO: replace this stub with your framework's real training / inference loop.
#       Log whatever is useful, e.g. experiment.log_parameters({...}),
#       experiment.log_metrics({...}), experiment.log_model(...).
for step in range(10):
    experiment.log_metric("loss", 1.0 / (step + 1), step=step)

experiment.end()
