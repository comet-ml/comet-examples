# coding: utf-8

from comet_ml import init
from comet_ml.integration.metaflow import comet_flow

from metaflow import FlowSpec, Parameter, step


def script_path(filename):

    import os

    filepath = os.path.join(os.path.dirname(__file__))
    return os.path.join(filepath, filename)


@comet_flow(project_name="comet-example-metaflow-mnist")
class MnistFlow(FlowSpec):
    """
    The flow performs the following steps:
    1) Ingest the MNIST csv data into Pandas DataFrame
    2) Clean and wrangle data
    3) Split data into train and test
    4) Fit model on train data (multiple models with branches)
    5) Predict on test data
    6) Evaluate result
    """

    mnist_train_data = Parameter(
        "mnist_data",
        help="The path to mnist data file.",
        default=script_path("mnist_train.csv"),
    )

    test_size = Parameter(
        "test_size",
        help=(
            "Float, should be between 0.0 and 1.0 and represent the proportion of the"
            " dataset to include in the test split."
        ),
        default=0.5,
    )

    @step
    def start(self):
        """
        Load the data
        """
        import pandas as pd

        # Read data from csv file
        self.mnist_df = pd.read_csv(self.mnist_train_data)

        self.next(self.prepare_data)

    @step
    def prepare_data(self):

        """
        Prepare data
        """
        # Extract the features and the label from the data
        self.X_df = self.mnist_df.drop(["label"], axis=1)
        self.Y_df = self.mnist_df.label.values

        self.next(self.split_data)

    @step
    def split_data(self):
        """
        Split train data for modelling
        """
        from sklearn.model_selection import train_test_split

        # Split data into train and test set
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.X_df, self.Y_df, test_size=self.test_size, random_state=42
        )
        self.next(self.fit_predict_gaussian, self.fit_predict_forest)

    @step
    def fit_predict_gaussian(self):
        """
        Fit a gaussian naive bayes model to the data
        """
        # Import model
        from sklearn.metrics import accuracy_score
        from sklearn.naive_bayes import GaussianNB

        modelA = GaussianNB()

        # Fit the model
        modelA.fit(self.X_train, self.Y_train)

        # Predict
        self.predictionA = modelA.predict(self.X_test)

        self.comet_experiment.log_confusion_matrix(self.Y_test, self.predictionA)

        self.accuracy_gaussian = accuracy_score(self.predictionA, self.Y_test)

        self.comet_experiment.log_metric("accuracy", self.accuracy_gaussian)

        self.next(self.join)

    @step
    def fit_predict_forest(self):
        """
        Fit a Random forest model to the data
        """
        # Import model
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score

        modelB = RandomForestClassifier(random_state=1, n_jobs=-1)

        # Fit the model
        modelB.fit(self.X_train, self.Y_train)

        # Predict
        self.predictionB = modelB.predict(self.X_test)

        self.comet_experiment.log_confusion_matrix(self.Y_test, self.predictionB)

        self.accuracy_random_forest = accuracy_score(self.predictionB, self.Y_test)

        self.comet_experiment.log_metric("accuracy", self.accuracy_random_forest)

        self.next(self.join)

    @step
    def join(self, inputs):
        """
        Merge the data artifact from the models
        """

        # merge artificate during a join
        self.merge_artifacts(inputs)

        self.next(self.evaluate)

    @step
    def evaluate(self):
        """
        Evaluate the score of the models
        """
        from comet_ml import API

        # Measure accuracy
        self.comet_experiment.log_metric("accuracy_gaussian", self.accuracy_gaussian)
        print("Accuracy score for GaussianNB {}".format(self.accuracy_gaussian))

        self.comet_experiment.log_metric(
            "accuracy_random_forest", self.accuracy_random_forest
        )
        print("Accuracy score for RandomForest {}".format(self.accuracy_random_forest))

        # Logs which model was the best to the Run Experiment to easily
        # compare between different Runs
        if self.accuracy_gaussian > self.accuracy_random_forest:
            best_model = "GaussianNB"
        else:
            best_model = "RandomForest"

        run_experiment = API().get_experiment_by_key(self.run_comet_experiment_key)
        run_experiment.log_other("Best Model", best_model)

        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    # Login to Comet if needed
    init()

    MnistFlow()
