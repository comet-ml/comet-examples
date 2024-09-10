import comet_ml
import pandas as pd
from io import StringIO
import os

def get_raw_data(workspace_name: str, artifact_name: str):
    """
    In this function, we will check if the raw data exists in Comet Artifacts. If it does, we will download it from there,
    if not we will upload it from the local directory.

    Once the file is available locally, we will load it into a pandas dataframe and return it.
    """
    exp = comet_ml.get_running_experiment()

    try:
        artifact = exp.get_artifact(artifact_name=f"{artifact_name}_raw")
        
        # Download the artifact
        artifact.download(path="./")
    except Exception as e:
        print(f"Error downloading artifact: {e}")
        artifact = comet_ml.Artifact(name=f"{artifact_name}_raw", artifact_type="dataset")
        artifact.add("./credit_scoring_dataset.csv")
        exp.log_artifact(artifact)
    
    df = pd.read_csv("./credit_scoring_dataset.csv")
    return df

def preprocess_data(df: pd.DataFrame):
    """
    In this function, we will preprocess the data to make it ready for the model. We will store the preprocessed data in a 
    new Comet Artifact.
    """
    # Select the relevant columns
    df = df.loc[:, ['CustAge', 'CustIncome', 'EmpStatus', 'UtilRate', 'OtherCC', 'ResStatus', 'TmAtAddress', 'TmWBank',
                    'probdefault']]
    
    # Rename the target column
    df.rename({'probdefault': 'probability_default'}, inplace=True, axis=1)

    # Convert the categorical columns to category type
    for c in ['EmpStatus', 'OtherCC', 'ResStatus']:
        df[c] = df[c].astype('category')
    
    # Save the preprocessed data to a new Comet Artifact
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    artifact = comet_ml.Artifact(name=f"{artifact_name}_preprocessed", artifact_type="dataset")
    artifact.add(local_path_or_data=csv_buffer, logical_path="preprocessed_data.csv")
    
    exp = comet_ml.get_running_experiment()
    exp.log_artifact(artifact)
    
    return df
    
if __name__ == "__main__":
    workspace_name = os.environ["COMET_WORKSPACE"]
    project_name = os.environ["COMET_PROJECT_NAME"]
    artifact_name = os.environ["COMET_PROJECT_NAME"]

    exp = comet_ml.Experiment(workspace=workspace_name, project_name=project_name)
    df =  get_raw_data(workspace_name, artifact_name)

    processed_df = preprocess_data(df)

    print("Data preprocessing complete.")