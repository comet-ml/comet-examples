import comet_ml

import os
import pandas as pd
import numpy as np
import lightgbm as lgb

def get_training_data(artifact_name: str) -> pd.DataFrame:
    exp = comet_ml.get_running_experiment()

    artifact = exp.get_artifact(artifact_name)
    artifact.download(path="./")
    
    df = pd.read_csv("preprocessed_data.csv")
    for c in ['EmpStatus', 'OtherCC', 'ResStatus']:
        df[c] = df[c].astype('category')

    return df

def train_model(training_data: pd.DataFrame, model_name: str) -> lgb.Booster:
    exp = comet_ml.get_running_experiment()
    
    # Create training dataset
    X_train = training_data.drop('probability_default', axis=1)
    y_train = (training_data['probability_default'] >= 0.5)

    training_dataset = lgb.Dataset(data = X_train,
                                   label = y_train)

    # Train model
    params = {
        'num_iterations': 30,
        'max_depth': 2,
        'objective': 'binary',
        'metric': ['auc', 'average_precision', 'l1', 'l2']
    }
    model = lgb.train(params = params,
                      train_set = training_dataset,
                      valid_sets = training_dataset)

    # Evaluate model
    y_pred = np.where(model.predict(X_train) > 0.5, 1, 0)
    experiment.log_confusion_matrix(
        y_true=y_train,
        y_predicted=y_pred
    )

    # Save model and log to Comet
    model.save_model('./model.txt')
    experiment.log_model(model_name, './model.txt')
    os.remove('./model.txt')

    return model
    

if __name__ == '__main__':
    ARTIFACT_NAME = os.environ["COMET_PROJECT_NAME"]
    WORKSPACE = os.environ["COMET_WORKSPACE"]
    MODEL_REGISTRY_NAME = os.environ["COMET_MODEL_REGISTRY_NAME"]
    
    # Model training script
    experiment = comet_ml.Experiment()

    training_data = get_training_data(artifact_name = f"{ARTIFACT_NAME}_preprocessed")
    model = train_model(training_data, model_name = MODEL_REGISTRY_NAME)
    
    
    experiment.register_model(MODEL_REGISTRY_NAME)