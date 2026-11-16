
"""Evaluate augmented deepfake detectors and compare them with original baselines."""

import argparse
import csv
from pathlib import Path

BASELINE_CSV = Path("results/transformation_results.csv")
RESULT_COLUMNS = (
    "model", "condition", "accuracy", "accuracy_drop", "precision",
    "recall", "f1", "true_positives", "true_negatives",
    "false_positives", "false_negatives", "roc_auc", "loss",
)


def read_results(path, model=None):
    """Read condition results, optionally filtering a multi-model CSV."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing results: {path}")

    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        required = {"condition", "accuracy"} | (
            {"model"} if model else set()
        )
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Missing columns in {path}: {sorted(missing)}"
            )

        rows = {}
        for row in reader:
            if model and row["model"] != model:
                continue

            condition = row["condition"]
            if condition in rows:
                raise ValueError(
                    f"Duplicate condition {condition} in {path}"
                )
            rows[condition] = row

    if "clean" not in rows:
        raise ValueError(
            f"Missing clean condition for {model or path}"
        )
    return rows


def evaluate_model(model_name, checkpoint_path, output_path):
    """Evaluate a saved model on clean and transformed test images."""
    import torch
    from torchvision import datasets

    from evaluate_transformations import (
        CONDITIONS,
        build_model_for_evaluation,
        evaluate_condition,
    )
    from train_baseline import get_device

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Missing checkpoint: {checkpoint_path}"
        )

    # Verify that all conditions contain identical test samples and labels.
    expected_mapping = None
    expected_images = None

    for condition, root in CONDITIONS.items():
        dataset = datasets.ImageFolder(str(root))
        images = {
            (Path(path).parent.name, Path(path).name)
            for path, _ in dataset.samples
        }

        if expected_images is None:
            expected_images = images
            expected_mapping = dataset.class_to_idx
        elif (
            dataset.class_to_idx != expected_mapping
            or images != expected_images
        ):
            raise ValueError(
                f"Different test samples or labels: {condition}"
            )

    print(
        f"Verified {len(expected_images)} matching images per condition."
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    if checkpoint.get("model_name") != model_name:
        raise ValueError(
            "Checkpoint architecture does not match --model"
        )

    class_to_idx = checkpoint["class_to_idx"]
    if class_to_idx != expected_mapping:
        raise ValueError(
            "Checkpoint labels do not match the test dataset"
        )

    model, preprocessing = build_model_for_evaluation(model_name)
    model.load_state_dict(checkpoint["state_dict"])

    device = get_device()
    model = model.to(device)
    model.eval()

    print(f"Evaluating {checkpoint_path.name} on {device}...")

    rows = []
    clean_accuracy = None

    for condition, root in CONDITIONS.items():
        metrics = evaluate_condition(
            model,
            preprocessing,
            root,
            class_to_idx,
            device,
        )

        if condition == "clean":
            clean_accuracy = metrics["accuracy"]

        rows.append({
            "model": f"{model_name}_augmented",
            "condition": condition,
            "accuracy": metrics["accuracy"],
            "accuracy_drop": (
                clean_accuracy - metrics["accuracy"]
            ),
            **{
                field: metrics[field]
                for field in RESULT_COLUMNS[4:]
            },
        })

        print(
            f"{condition:12s}: "
            f"accuracy={metrics['accuracy'] * 100:.2f}%"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("x", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=RESULT_COLUMNS
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved evaluation: {output_path}")


def compare_results(model_name, augmented_path, comparison_path):
    """Compare baseline and augmented accuracy across all test conditions."""
    baseline = read_results(
        BASELINE_CSV, model=model_name
    )
    augmented = read_results(
        augmented_path,
        model=f"{model_name}_augmented",
    )

    if set(baseline) != set(augmented):
        raise ValueError(
            "Baseline and augmented conditions do not match"
        )

    baseline_clean = float(
        baseline["clean"]["accuracy"]
    )
    augmented_clean = float(
        augmented["clean"]["accuracy"]
    )

    rows = []

    print(
        "\nCondition      Original   Augmented   "
        "Change (pp)   Drop reduction (pp)"
    )

    for condition, row in augmented.items():
        original = baseline[condition]

        old_accuracy = float(original["accuracy"])
        new_accuracy = float(row["accuracy"])

        # Accuracy change measures absolute improvement. Drop reduction
        # accounts for differences in each model's clean accuracy.
        change_pp = 100 * (
            new_accuracy - old_accuracy
        )

        drop_reduction_pp = 100 * (
            (baseline_clean - old_accuracy)
            - (augmented_clean - new_accuracy)
        )

        rows.append({
            "condition": condition,
            "baseline_accuracy": old_accuracy,
            "augmented_accuracy": new_accuracy,
            "accuracy_change_pp": change_pp,
            "drop_reduction_pp": drop_reduction_pp,
            "baseline_false_positives": original.get(
                "false_positives", ""
            ),
            "augmented_false_positives": row.get(
                "false_positives", ""
            ),
            "baseline_false_negatives": original.get(
                "false_negatives", ""
            ),
            "augmented_false_negatives": row.get(
                "false_negatives", ""
            ),
        })

        print(
            f"{condition:14s} "
            f"{old_accuracy * 100:7.2f}% "
            f"{new_accuracy * 100:10.2f}% "
            f"{change_pp:+11.2f} "
            f"{drop_reduction_pp:+17.2f}"
        )

    if comparison_path.exists():
        print(
            f"\nExisting comparison left unchanged: {comparison_path}"
        )
    else:
        comparison_path.parent.mkdir(
            parents=True, exist_ok=True
        )

        with comparison_path.open("x", newline="") as file:
            writer = csv.DictWriter(
                file, fieldnames=list(rows[0])
            )
            writer.writeheader()
            writer.writerows(rows)

        print(f"\nSaved comparison: {comparison_path}")


def main():
    """Select the model and augmentation version from command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--model",
        required=True,
        choices=("efficientnet_b0", "resnet50"),
    )
    parser.add_argument(
        "--strategy",
        required=True,
        choices=("v1", "v2"),
    )

    args = parser.parse_args()

    run_name = f"{args.model}_augmented_{args.strategy}"

    checkpoint_path = (
        Path("models") / f"{run_name}.pth"
    )
    results_path = (
        Path("results")
        / f"{run_name}_transformation_results.csv"
    )
    comparison_path = (
        Path("results")
        / f"{run_name}_vs_baseline.csv"
    )

    # Reuse previously evaluated results to avoid unnecessary inference.
    if results_path.is_file():
        print(f"Using saved evaluation: {results_path}")
    else:
        evaluate_model(
            args.model,
            checkpoint_path,
            results_path,
        )

    compare_results(
        args.model,
        results_path,
        comparison_path,
    )


if __name__ == "__main__":
    main()
