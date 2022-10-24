import json

import PIL
import timm
import torch
import torchvision.transforms as transforms
from comet_ml.integration.metaflow import comet_flow
from comet_ml.integration.pytorch import log_model
from datasets import load_dataset
from metaflow import FlowSpec, JSONType, Parameter, card, step
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader

# define custom transform function
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)


def collate_fn(examples):
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


@comet_flow(project_name="comet-example-metaflow-model-evaluation")
class ModelEvaluationFlow(FlowSpec):
    models = Parameter(
        "models",
        help=("Models to evaluate"),
        default=["resnet18", "efficientnet_b0"],
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
        with open("imagenet_labels.json", "rb") as f:
            metadata = json.load(f)
            self.label_names = metadata["labels"]

        self.next(self.evaluate_classification_metrics, foreach="models")

    @step
    def evaluate_classification_metrics(self):
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
        from comet_ml import API

        self.results = [input.results for input in inputs]

        # Find best model based on macro averaged recall
        best_model = max(self.results, key=lambda x: x["macro avg"]["recall"])

        run_experiment = API().get_experiment_by_key(best_model["experiment_id"])
        run_experiment.register_model(
            best_model["model_name"],
            registry_name="sketch-model",
            description="Image Classification Model for Sketch Recognition",
        )

        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    ModelEvaluationFlow()
