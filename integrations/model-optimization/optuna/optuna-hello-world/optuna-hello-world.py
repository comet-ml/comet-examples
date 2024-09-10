# coding: utf-8
from comet_ml import login

import optuna
from optuna_integration.comet import CometCallback

# Login to Comet if needed
login()

study = optuna.create_study()
comet = CometCallback(
    study, project_name="comet-example-optuna-hello-world", metric_names=["score"]
)


@comet.track_in_comet()
def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    objective = (x - 2) ** 2

    return objective


study.optimize(objective, n_trials=20, callbacks=[comet])

best_params = study.best_params
found_x = best_params["x"]
print("Found x: {}, (x - 2)^2: {}".format(found_x, (found_x - 2) ** 2))
