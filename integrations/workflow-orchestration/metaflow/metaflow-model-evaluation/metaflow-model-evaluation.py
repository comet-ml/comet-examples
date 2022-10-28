# coding: utf-8

import os

import comet_ml
from comet_ml.integration.metaflow import comet_flow
from comet_ml.integration.pytorch import log_model

from metaflow import FlowSpec, JSONType, Parameter, step


def collate_fn(examples):
    import PIL
    import torch
    import torchvision.transforms as transforms

    # define custom transform function
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    images = []
    labels = []

    for example in examples:
        img = transform(
            example["image"].convert("L").resize((224, 224), PIL.Image.LANCZOS)
        )
        label = torch.tensor(example["label"])

        images.append(img.unsqueeze(0))
        labels.append(label.unsqueeze(0))

    images = torch.cat(images)
    labels = torch.tensor(labels, dtype=torch.int)

    return images, labels


def fetch_latest_model_metrics(registry_name, max_model_version):
    from comet_ml import API

    api = API()

    try:
        model_assets = api.get_model_registry_version_assets(
            workspace=os.environ["COMET_WORKSPACE"],
            registry_name=registry_name,
            version=max_model_version,
        )
        model_experiment_key = model_assets["experimentModel"]["experimentKey"]
        experiment = api.get_experiment_by_key(model_experiment_key)

        metrics_summary = experiment.get_metrics_summary()
        metrics_summary_map = {
            x["name"]: float(x["valueCurrent"]) for x in metrics_summary
        }

        return metrics_summary_map

    except Exception:
        return None


def update_model(candidate_model_score, metric_name, registry_name):
    import comet_ml

    api = comet_ml.API()

    try:
        existing_models = api.get_registry_model_names(os.getenv("COMET_WORKSPACE"))
        if registry_name not in existing_models:
            # Register the model if it doesn't exist
            return True

        model_versions = api.get_registry_model_versions(
            workspace=os.environ["COMET_WORKSPACE"], registry_name=registry_name
        )
        max_model_version = max(model_versions)

        latest_model_metrics = fetch_latest_model_metrics(
            registry_name, max_model_version
        )

        current_model_score = latest_model_metrics[metric_name]
        if candidate_model_score > current_model_score:
            return True
        else:
            return False

    except Exception:
        return False


def register_model(best_model, registry_name):
    from comet_ml import API

    api = API()

    try:
        existing_models = api.get_registry_model_versions(
            workspace=os.environ["COMET_WORKSPACE"], registry_name=registry_name
        )
        max_model_version = max(existing_models)

        new_model_version = max_model_version.split(".")
        new_model_version[0] = str(int(new_model_version[0]) + 1)
        new_model_version = ".".join(new_model_version)
    except Exception:
        new_model_version = "1.0.0"

    api_experiment = api.get_experiment_by_key(best_model["experiment_id"])
    api_experiment.register_model(
        best_model["model_name"], registry_name=registry_name, version=new_model_version
    )


@comet_flow(project_name="comet-example-metaflow-model-evaluation")
class ModelEvaluationFlow(FlowSpec):
    models = Parameter(
        "models",
        help=("Models to evaluate"),
        type=JSONType,
        default='["resnet18", "efficientnet_b0"]',
    )
    dataset_name = Parameter(
        "dataset_name",
        help=("Name of the dataset to use for evaluation"),
        default="imagenet_sketch",
    )
    dataset_split = Parameter(
        "dataset_split",
        help=("Dataset Split to use for evaluation"),
        default="train",
    )
    batch_size = Parameter(
        "batch_size",
        help=("Batch Size to Use"),
        default=32,
    )
    n_samples = Parameter(
        "n_samples",
        help=("Number of Samples"),
        default=1000,
    )
    seed = Parameter(
        "seed",
        help=("Random Seed"),
        default=42,
    )

    @step
    def start(self):
        """
        Load the data
        """
        import json

        with open("imagenet_labels.json", "rb") as f:
            metadata = json.load(f)
            self.label_names = metadata["labels"]

        if not isinstance(self.models, list) or len(self.models) == 0:
            raise ValueError(
                """--models argument is supposed to be a list with at least one item,"""
                """ not %r""" % self.models
            )

        self.next(self.evaluate_classification_metrics, foreach="models")

    @step
    def evaluate_classification_metrics(self):
        import timm
        import torch
        from datasets import load_dataset
        from sklearn.metrics import accuracy_score, classification_report
        from torch.utils.data import DataLoader

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dataset = load_dataset(
            self.dataset_name, split=self.dataset_split, streaming=True
        )
        dataset = dataset.shuffle(self.seed, buffer_size=10_000)
        dataset = dataset.take(self.n_samples)
        dataset = dataset.with_format("torch")

        dataloader = DataLoader(
            dataset, collate_fn=collate_fn, batch_size=self.batch_size
        )

        model = timm.create_model(self.input, pretrained=True, in_chans=1)
        model.to(device)
        model.eval()
        self.comet_experiment.log_parameters({"model_name": self.input})

        labels = []
        predictions = []
        for images, label in dataloader:
            probs = torch.nn.functional.softmax(model(images.to(device)), dim=1)

            predictions.append(probs.cpu())
            labels.append(label)

        predictions = torch.cat(predictions)
        labels = torch.cat(labels)

        clf_metrics = classification_report(
            labels,
            torch.argmax(predictions, dim=1),
            labels=[i for i in range(1000)],
            target_names=self.label_names,
            output_dict=True,
        )
        accuracy = accuracy_score(labels, torch.argmax(predictions, dim=1))

        self.comet_experiment.log_metrics(clf_metrics["micro avg"], prefix="micro_avg")
        self.comet_experiment.log_metrics(clf_metrics["macro avg"], prefix="macro_avg")
        self.comet_experiment.log_metrics({"accuracy": accuracy})

        log_model(self.comet_experiment, model, self.input)

        self.results = clf_metrics
        self.results.update(
            {"model_name": self.input, "experiment_id": self.comet_experiment.id}
        )
        self.next(self.join)

    @step
    def join(self, inputs):
        self.results = [input.results for input in inputs]

        # Find best model based on macro averaged recall
        best_model = max(self.results, key=lambda x: x["macro avg"]["recall"])

        candidate_score = best_model["macro avg"]["recall"]
        if update_model(candidate_score, "macro_avg_recall", "sketch-model"):
            print("Updating Registry Model")
            register_model(best_model, "sketch-model")
        else:
            print("Not Updating Registry Model")

        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    comet_ml.init()

    ModelEvaluationFlow()
