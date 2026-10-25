import argparse
import copy
import csv
import os
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


SEED = 42
IMAGE_SIZE = 224


def set_seed():
    random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def build_model(model_name):
    if model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)

        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 2)
    
    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)

        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 2)

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model


def create_dataloaders(data_dir, batch_size, model_name):
    if model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT

    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    eval_transform = weights.transforms()

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        eval_transform,
    ])

    datasets_by_split = {
        "train": datasets.ImageFolder(
            os.path.join(data_dir, "train"),
            transform=train_transform,
        ),
        "val": datasets.ImageFolder(
            os.path.join(data_dir, "val"),
            transform=eval_transform,
        ),
        "test": datasets.ImageFolder(
            os.path.join(data_dir, "test"),
            transform=eval_transform,
        ),
    }

    train_classes = datasets_by_split["train"].class_to_idx

    for split in ["val", "test"]:
        if datasets_by_split[split].class_to_idx != train_classes:
            raise ValueError(
                f"Class mapping for {split} does not match training data."
            )

    loaders = {
        "train": DataLoader(
            datasets_by_split["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        ),
        "val": DataLoader(
            datasets_by_split["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        ),
        "test": DataLoader(
            datasets_by_split["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        ),
    }

    return datasets_by_split, loaders


def calculate_auc(labels, scores):
    positives = sum(labels)
    negatives = len(labels) - positives

    if positives == 0 or negatives == 0:
        return float("nan")

    indexed_scores = sorted(
        enumerate(scores),
        key=lambda pair: pair[1]
    )

    ranks = [0.0] * len(scores)
    index = 0

    while index < len(indexed_scores):
        end = index

        while (
            end + 1 < len(indexed_scores)
            and indexed_scores[end + 1][1] == indexed_scores[index][1]
        ):
            end += 1

        average_rank = ((index + 1) + (end + 1)) / 2

        for position in range(index, end + 1):
            original_index = indexed_scores[position][0]
            ranks[original_index] = average_rank

        index = end + 1

    positive_rank_sum = sum(
        rank
        for rank, label in zip(ranks, labels)
        if label == 1
    )

    auc = (
        positive_rank_sum
        - positives * (positives + 1) / 2
    ) / (positives * negatives)

    return auc


def calculate_metrics(actual_labels, fake_probabilities, fake_index):
    binary_actual = [
        1 if label == fake_index else 0
        for label in actual_labels
    ]

    binary_predictions = [
        1 if probability >= 0.5 else 0
        for probability in fake_probabilities
    ]

    tp = sum(
        1
        for actual, predicted
        in zip(binary_actual, binary_predictions)
        if actual == 1 and predicted == 1
    )

    tn = sum(
        1
        for actual, predicted
        in zip(binary_actual, binary_predictions)
        if actual == 0 and predicted == 0
    )

    fp = sum(
        1
        for actual, predicted
        in zip(binary_actual, binary_predictions)
        if actual == 0 and predicted == 1
    )

    fn = sum(
        1
        for actual, predicted
        in zip(binary_actual, binary_predictions)
        if actual == 1 and predicted == 0
    )

    accuracy = (tp + tn) / len(binary_actual)

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    auc = calculate_auc(
        binary_actual,
        fake_probabilities
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
    }


def evaluate(model, loader, fake_index, device):
    model.eval()

    actual_labels = []
    fake_probabilities = []

    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            probabilities = torch.softmax(logits, dim=1)

            total_loss += loss.item() * images.size(0)

            actual_labels.extend(
                labels.cpu().tolist()
            )

            fake_probabilities.extend(
                probabilities[:, fake_index].cpu().tolist()
            )

    metrics = calculate_metrics(
        actual_labels,
        fake_probabilities,
        fake_index,
    )

    metrics["loss"] = total_loss / len(loader.dataset)

    return metrics, actual_labels, fake_probabilities


def train_model(
    model_name,
    datasets_by_split,
    loaders,
    epochs,
    learning_rate,
    device,
):
    print(f"\n===== {model_name} =====")

    model = build_model(model_name)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    trainable_parameters = model.parameters()

    optimizer = torch.optim.Adam(
        trainable_parameters,
        lr=learning_rate,
    )

    fake_index = (
        datasets_by_split["train"]
        .class_to_idx["fake"]
    )

    best_validation_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in loaders["train"]:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            predictions = torch.argmax(logits, dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

        train_loss = running_loss / total
        train_accuracy = correct / total

        val_metrics, _, _ = evaluate(
            model,
            loaders["val"],
            fake_index,
            device,
        )

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train loss={train_loss:.4f} | "
            f"train acc={train_accuracy:.4f} | "
            f"val loss={val_metrics['loss']:.4f} | "
            f"val acc={val_metrics['accuracy']:.4f}"
        )

        if val_metrics["loss"] < best_validation_loss:
            best_validation_loss = val_metrics["loss"]
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)

    os.makedirs("models", exist_ok=True)

    checkpoint_path = (
        f"models/{model_name}_baseline.pth"
    )

    torch.save(
        {
            "model_name": model_name,
            "state_dict": model.state_dict(),
            "class_to_idx": datasets_by_split[
                "train"
            ].class_to_idx,
        },
        checkpoint_path,
    )

    test_metrics, labels, fake_probabilities = evaluate(
        model,
        loaders["test"],
        fake_index,
        device,
    )

    print("\nTest results:")

    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")

    return test_metrics, labels, fake_probabilities


def save_predictions(
    model_name,
    dataset,
    labels,
    probabilities,
    class_to_idx,
):
    os.makedirs("results", exist_ok=True)

    idx_to_class = {
        value: key
        for key, value in class_to_idx.items()
    }

    output_path = (
        f"results/{model_name}_baseline_predictions.csv"
    )

    with open(output_path, "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "filename",
            "actual_label",
            "predicted_label",
            "fake_probability",
        ])

        for (
            sample,
            actual_index,
            fake_probability,
        ) in zip(
            dataset.samples,
            labels,
            probabilities,
        ):
            image_path = sample[0]

            predicted_label = (
                "fake"
                if fake_probability >= 0.5
                else "real"
            )

            writer.writerow([
                os.path.basename(image_path),
                idx_to_class[actual_index],
                predicted_label,
                fake_probability,
            ])

    print(f"Predictions saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        required=True,
    )

    parser.add_argument(
        "--model",
        choices=[
            "efficientnet_b0",
            "resnet50",
            "both",
        ],
        default="both",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.0001,
    )

    args = parser.parse_args()

    set_seed()

    device = get_device()

    print(f"Using device: {device}")


    if args.model == "both":
        model_names = [
            "efficientnet_b0",
            "resnet50",
        ]
    else:
        model_names = [args.model]

    os.makedirs("results", exist_ok=True)

    for model_name in model_names:
        datasets_by_split, loaders = create_dataloaders(
            args.data_dir,
            args.batch_size,
            model_name,
        )

        print(f"\nPreparing data for {model_name}")

        print(
            "Classes:",
            datasets_by_split["train"].class_to_idx
        )

        print(
            f"Train: {len(datasets_by_split['train'])}"
        )

        print(
            f"Validation: {len(datasets_by_split['val'])}"
        )

        print(
            f"Test: {len(datasets_by_split['test'])}"
        )

        metrics, labels, probabilities = train_model(
            model_name,
            datasets_by_split,
            loaders,
            args.epochs,
            args.learning_rate,
            device,
        )

        metrics_path = (
            f"results/{model_name}_baseline_metrics.csv"
        )

        with open(
            metrics_path,
            "w",
            newline="",
        ) as file:
            writer = csv.writer(file)

            writer.writerow([
                "model",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "loss",
            ])

            writer.writerow([
                model_name,
                metrics["accuracy"],
                metrics["precision"],
                metrics["recall"],
                metrics["f1"],
                metrics["roc_auc"],
                metrics["loss"],
            ])

        save_predictions(
            model_name,
            datasets_by_split["test"],
            labels,
            probabilities,
            datasets_by_split[
                "train"
            ].class_to_idx,
        )

        print(
            f"Metrics saved to {metrics_path}"
        )


if __name__ == "__main__":
    main()