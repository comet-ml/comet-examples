# coding: utf-8

from comet_ml import init
from comet_ml.integration.metaflow import comet_flow

from metaflow import FlowSpec, JSONType, Parameter, card, step


@comet_flow(project_name="comet-example-metaflow-regression")
class RegressionFlow(FlowSpec):

    models = Parameter(
        "models",
        help=("A list of models class to train."),
        type=JSONType,
        default='["Regression", "Decision Tree", "k-NN"]',
    )

    @step
    def start(self):
        """
        Load the data
        """
        import plotly.express as px

        self.input_df = px.data.tips()

        self.next(self.split_data)

    @step
    def split_data(self):
        """
        Split train data for modelling
        """
        from sklearn.model_selection import train_test_split

        self.X = self.input_df.total_bill.values[:, None]
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.X, self.input_df.tip, random_state=42
        )

        self.next(self.train_model, foreach="models")

    @card(type="html")
    @step
    def train_model(self):
        import numpy as np
        import plotly.graph_objects as go
        from sklearn import linear_model, neighbors, tree

        model_name = self.input

        if model_name == "Regression":
            model = linear_model.LinearRegression()
        elif model_name == "Decision Tree":
            model = tree.DecisionTreeRegressor()
        elif model_name == "k-NN":
            model = neighbors.KNeighborsRegressor()
        else:
            raise ValueError("Invalid model name")

        self.comet_experiment.log_parameter("model", model)

        model.fit(self.X_train, self.Y_train)

        self.score = model.score(self.X_test, self.Y_test)
        self.comet_experiment.log_metric("score", self.score)

        # Visualize predictions
        x_range = np.linspace(self.X.min(), self.X.max(), 100)
        y_range = model.predict(x_range.reshape(-1, 1))

        fig = go.Figure(
            [
                go.Scatter(
                    x=self.X_train.squeeze(),
                    y=self.Y_train,
                    name="train",
                    mode="markers",
                ),
                go.Scatter(
                    x=self.X_test.squeeze(), y=self.Y_test, name="test", mode="markers"
                ),
                go.Scatter(x=x_range, y=y_range, name="prediction"),
            ],
            layout=go.Layout(
                title=go.layout.Title(text="Predictions for model %s" % model_name)
            ),
        )
        self.html = fig.to_html()

        self.next(self.join)

    @step
    def join(self, inputs):
        """
        Merge the data artifact from the models
        """
        from comet_ml import API

        # merge artificate during a join
        best_model, best_score = None, float("-inf")

        for _input in inputs:
            self.comet_experiment.log_metric("%s_score" % _input.input, _input.score)

            if _input.score > best_score:
                best_score = _input.score
                best_model = _input.input

        # Logs which model was the best to the Run Experiment to easily
        # compare between different Runs
        run_experiment = API().get_experiment_by_key(self.run_comet_experiment_key)
        run_experiment.log_parameter("Best Model", best_model)

        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    # Login to Comet if needed
    init()

    RegressionFlow()
