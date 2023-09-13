# coding: utf-8
from comet_ml import Experiment, init

import optuna

# Login to Comet if needed
init()


def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    objective = (x - 2) ** 2

    experiment = Experiment(project_name="comet-example-optuna-hello-world")

    experiment.log_optimization(
        optimization_id=trial.study.study_name,
        metric_name="objective",
        metric_value=objective,
        parameters={"x": x},
        objective="minimize",
    )

    return objective


study = optuna.create_study()
study.optimize(objective, n_trials=20)

best_params = study.best_params
found_x = best_params["x"]
print("Found x: {}, (x - 2)^2: {}".format(found_x, (found_x - 2) ** 2))
