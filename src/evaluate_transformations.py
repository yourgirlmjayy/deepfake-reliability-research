import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models

from train_baseline import calculate_metrics, get_device


BATCH_SIZE = 8

CLEAN_ROOT = Path(
    "../deepfake_reliability_project/data/aligned_faces/test"
)

TRANSFORMED_ROOT = Path(
    "../deepfake_reliability_project/data/transformed_test"
)

CONDITIONS = {
    "clean": CLEAN_ROOT,

    "jpeg_q70": TRANSFORMED_ROOT / "jpeg_q70",
    "jpeg_q50": TRANSFORMED_ROOT / "jpeg_q50",
    "jpeg_q30": TRANSFORMED_ROOT / "jpeg_q30",

    "crop_90": TRANSFORMED_ROOT / "crop_90",
    "crop_80": TRANSFORMED_ROOT / "crop_80",
    "crop_70": TRANSFORMED_ROOT / "crop_70",

    "resize_75": TRANSFORMED_ROOT / "resize_75",
    "resize_50": TRANSFORMED_ROOT / "resize_50",
    "resize_30": TRANSFORMED_ROOT / "resize_30",

    "blur_r1": TRANSFORMED_ROOT / "blur_r1",
    "blur_r2": TRANSFORMED_ROOT / "blur_r2",
    "blur_r4": TRANSFORMED_ROOT / "blur_r4",
}

CHECKPOINTS = {
    "efficientnet_b0": Path(
        "models/efficientnet_b0_baseline.pth"
    ),
    "resnet50": Path(
        "models/resnet50_baseline.pth"
    ),
}


def build_model_for_evaluation(model_name):
    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(
            weights=None
        )

        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(
            in_features,
            2,
        )

        weights = models.EfficientNet_B0_Weights.DEFAULT

    elif model_name == "resnet50":
        model = models.resnet50(
            weights=None
        )

        in_features = model.fc.in_features
        model.fc = nn.Linear(
            in_features,
            2,
        )

        weights = models.ResNet50_Weights.DEFAULT

    else:
        raise ValueError(
            f"Unsupported model: {model_name}"
        )

    return model, weights.transforms()


def evaluate_condition(
    model,
    transform,
    data_root,
    class_to_idx,
    device,
):
    dataset = datasets.ImageFolder(
        str(data_root),
        transform=transform,
    )

    if dataset.class_to_idx != class_to_idx:
        raise ValueError(
            f"Class mapping mismatch for {data_root}: "
            f"{dataset.class_to_idx}"
        )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    fake_index = class_to_idx["fake"]

    actual_labels = []
    fake_probabilities = []

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0

    model.eval()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)

            loss = criterion(
                logits,
                labels,
            )

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            total_loss += (
                loss.item() * images.size(0)
            )

            actual_labels.extend(
                labels.cpu().tolist()
            )

            fake_probabilities.extend(
                probabilities[
                    :, fake_index
                ].cpu().tolist()
            )

    metrics = calculate_metrics(
    actual_labels,
    fake_probabilities,
    fake_index,
    )

    # Convert fake probabilities into predicted class labels.
    # "fake" is treated as the positive class.
    predicted_labels = [
        fake_index if probability >= 0.5
        else 1 - fake_index
        for probability in fake_probabilities
    ]

    true_positives = sum(
        actual == fake_index and predicted == fake_index
        for actual, predicted in zip(
            actual_labels,
            predicted_labels,
        )
    )

    false_negatives = sum(
        actual == fake_index and predicted != fake_index
        for actual, predicted in zip(
            actual_labels,
            predicted_labels,
        )
    )

    false_positives = sum(
        actual != fake_index and predicted == fake_index
        for actual, predicted in zip(
            actual_labels,
            predicted_labels,
        )
    )

    true_negatives = sum(
        actual != fake_index and predicted != fake_index
        for actual, predicted in zip(
            actual_labels,
            predicted_labels,
        )
    )

    metrics["true_positives"] = true_positives
    metrics["true_negatives"] = true_negatives
    metrics["false_positives"] = false_positives
    metrics["false_negatives"] = false_negatives

    metrics["loss"] = (
        total_loss / len(dataset)
    )
    return metrics


def main():
    device = get_device()

    print(f"Using device: {device}")

    Path("results").mkdir(
        exist_ok=True
    )

    output_path = Path(
        "results/transformation_results.csv"
    )

    rows = []

    for model_name, checkpoint_path in CHECKPOINTS.items():
        print(
            f"\n===== {model_name} ====="
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )

        class_to_idx = checkpoint[
            "class_to_idx"
        ]

        model, transform = (
            build_model_for_evaluation(
                model_name
            )
        )

        model.load_state_dict(
            checkpoint["state_dict"]
        )

        model = model.to(device)

        clean_accuracy = None

        for condition, data_root in CONDITIONS.items():
            metrics = evaluate_condition(
                model,
                transform,
                data_root,
                class_to_idx,
                device,
            )

            if condition == "clean":
                clean_accuracy = metrics[
                    "accuracy"
                ]

            accuracy_drop = (
                clean_accuracy
                - metrics["accuracy"]
            )

            rows.append({
                "model": model_name,
                "condition": condition,
                "accuracy": metrics["accuracy"],
                "accuracy_drop": accuracy_drop,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "true_positives": metrics["true_positives"],
                "true_negatives": metrics["true_negatives"],
                "false_positives": metrics["false_positives"],
                "false_negatives": metrics["false_negatives"],
                "roc_auc": metrics["roc_auc"],
                "loss": metrics["loss"],
            })

            print(
                f"{condition}: "
                f"accuracy={metrics['accuracy']:.4f}, "
                f"drop={accuracy_drop:.4f}, "
                f"precision={metrics['precision']:.4f}, "
                f"recall={metrics['recall']:.4f}, "
                f"f1={metrics['f1']:.4f}, "
                f"auc={metrics['roc_auc']:.4f}"
            )

    with open(
        output_path,
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "model",
                "condition",
                "accuracy",
                "accuracy_drop",
                "precision",
                "recall",
                "f1",
                "true_positives",
                "true_negatives",
                "false_positives",
                "false_negatives",
                "roc_auc",
                "loss",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        f"\nResults saved to {output_path}"
    )


if __name__ == "__main__":
    main()
