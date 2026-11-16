
"""Train EfficientNet-B0 or ResNet-50 with controlled image degradations."""

import argparse
import copy
import csv
import random
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

from train_baseline import build_model, evaluate, get_device, set_seed
from transformations import TRANSFORMATIONS


class RandomImageEdit:
    """V1: apply one existing image transformation with probability p."""

    def __init__(self, probability=0.5):
        if not 0 <= probability <= 1:
            raise ValueError("probability must be in [0, 1]")
        self.probability = probability

    def __call__(self, image):
        if random.random() >= self.probability:
            return image
        return TRANSFORMATIONS[random.choice(tuple(TRANSFORMATIONS))](image)


class StochasticDegradation:
    """V2: sample a degradation chain adapted from Lu & Ebrahimi (2024).

    Each operation is optional, with an independently sampled intensity.
    This preserves relatively clean examples while exposing the detector
    to combined image-processing artifacts.
    """

    def __call__(self, image):
        image = image.convert("RGB")

        if random.random() < 0.50:
            enhancer = random.choice(
                (ImageEnhance.Brightness, ImageEnhance.Contrast)
            )
            image = enhancer(image).enhance(random.uniform(0.5, 1.5))

        if random.random() < 0.50:
            kernel = random.randrange(3, 16, 2)
            if random.random() < 0.5:
                image = image.filter(
                    ImageFilter.GaussianBlur(radius=kernel / 6)
                )
            else:
                image = image.filter(
                    ImageFilter.BoxBlur(radius=(kernel - 1) / 2)
                )

        if random.random() < 0.30:
            sigma = random.uniform(0, 50)
            pixels = np.asarray(image, dtype=np.float32)
            noise = np.random.normal(0, sigma, pixels.shape)
            pixels = np.clip(pixels + noise, 0, 255).astype(np.uint8)
            image = Image.fromarray(pixels, mode="RGB")

        if random.random() < 0.70:
            buffer = BytesIO()
            image.save(
                buffer,
                format="JPEG",
                quality=random.randint(10, 95),
            )
            buffer.seek(0)
            with Image.open(buffer) as decoded:
                image = decoded.convert("RGB")

        return image


def build_loaders(
    data_dir, batch_size, model_name, strategy, augmentation_prob
):
    """Apply random edits to training data, retaining clean validation/test sets."""
    weights = (
        models.EfficientNet_B0_Weights.DEFAULT
        if model_name == "efficientnet_b0"
        else models.ResNet50_Weights.DEFAULT
    )
    preprocess = weights.transforms()

    augmentation = (
        RandomImageEdit(augmentation_prob)
        if strategy == "v1"
        else StochasticDegradation()
    )
    train_preprocess = transforms.Compose([
        augmentation,
        transforms.RandomHorizontalFlip(p=0.5),
        preprocess,
    ])

    data = {
        split: datasets.ImageFolder(
            str(data_dir / split),
            transform=train_preprocess if split == "train" else preprocess,
        )
        for split in ("train", "val", "test")
    }

    if set(data["train"].class_to_idx) != {"real", "fake"}:
        raise ValueError("Expected real/ and fake/ class folders")

    for split in ("val", "test"):
        if data[split].class_to_idx != data["train"].class_to_idx:
            raise ValueError(f"Inconsistent class mapping in {split}")

    loaders = {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=0,
        )
        for split, dataset in data.items()
    }
    return data, loaders, preprocess


def build_robust_validation(data_dir, preprocess, batch_size, seed=42):
    """Create fixed, balanced validation samples under severe image edits.

    Checkpoint selection uses validation data rather than test results.
    Both transformations are applied to identical source images.
    """
    root = data_dir / "val"
    reference = datasets.ImageFolder(str(root))
    rng = random.Random(seed)
    indices = []

    for label in reference.class_to_idx.values():
        candidates = [
            i for i, (_, y) in enumerate(reference.samples) if y == label
        ]
        indices.extend(
            rng.sample(candidates, min(128, len(candidates)))
        )
    indices.sort()

    loaders = {}
    for condition in ("jpeg_q30", "blur_r4"):
        # Fixed transformations provide comparable scores across epochs.
        edit = TRANSFORMATIONS[condition]
        transform = transforms.Compose([edit, preprocess])
        dataset = datasets.ImageFolder(str(root), transform=transform)

        loaders[condition] = DataLoader(
            Subset(dataset, indices),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
    return loaders


def save_csv(path, rows):
    """Write experiment results without overwriting an existing file."""
    with path.open("x", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    """Train, select the best validation checkpoint, and save clean-test results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument(
        "--model",
        choices=("efficientnet_b0", "resnet50"),
        default="efficientnet_b0",
    )
    parser.add_argument(
        "--strategy", choices=("v1", "v2"), default="v1"
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--augmentation-prob", type=float, default=0.5)
    parser.add_argument("--run-name")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error("epochs, batch-size, and learning-rate must be positive")
    if not 0 <= args.augmentation_prob <= 1:
        parser.error("augmentation-prob must be in [0, 1]")
    if not args.data_dir.is_dir():
        parser.error(f"Dataset directory not found: {args.data_dir}")

    run_name = args.run_name or f"{args.model}_augmented_{args.strategy}"
    if not run_name or not all(
        c.isalnum() or c in "_-" for c in run_name
    ):
        parser.error("run-name must contain only letters, numbers, '_' or '-'")

    checkpoint_path = Path("models") / f"{run_name}.pth"
    metrics_path = Path("results") / f"{run_name}_metrics.csv"
    predictions_path = Path("results") / f"{run_name}_predictions.csv"
    history_path = Path("results") / f"{run_name}_history.csv"
    outputs = (
        checkpoint_path, metrics_path, predictions_path, history_path
    )

    # Check for existing outputs before starting a long training run.
    if not args.check_only:
        existing = [str(path) for path in outputs if path.exists()]
        if existing:
            parser.error(
                f"Existing files would be overwritten: {existing}"
            )

    set_seed()
    np.random.seed(42)
    data, loaders, preprocess = build_loaders(
        args.data_dir,
        args.batch_size,
        args.model,
        args.strategy,
        args.augmentation_prob,
    )

    print("Classes:", data["train"].class_to_idx)
    for split in ("train", "val", "test"):
        print(f"{split}: {len(data[split])} images")

    if args.check_only:
        images, labels = next(iter(loaders["train"]))
        print(
            "Augmented batch:",
            tuple(images.shape),
            "labels:",
            labels.tolist(),
        )
        print("Data check complete; no files saved.")
        return

    device = get_device()
    print("Using device:", device)
    model = build_model(args.model).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate
    )
    fake_index = data["train"].class_to_idx["fake"]

    robust_loaders = (
        build_robust_validation(
            args.data_dir, preprocess, args.batch_size
        )
        if args.strategy == "v2"
        else {}
    )

    best_score = float("-inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        correct = seen = 0
        summed_loss = 0.0

        for images, labels in loaders["train"]:
            images = images.to(device)
            labels = labels.to(device)

            # Update model parameters using classification loss.
            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            n = images.size(0)
            seen += n
            summed_loss += loss.item() * n
            correct += (logits.argmax(1) == labels).sum().item()

        clean_metrics, _, _ = evaluate(
            model, loaders["val"], fake_index, device
        )

        row = {
            "epoch": epoch,
            "train_loss": summed_loss / seen,
            "train_accuracy": correct / seen,
            "val_loss": clean_metrics["loss"],
            "val_accuracy": clean_metrics["accuracy"],
        }

        if args.strategy == "v2":
            # Balance clean-image accuracy against severe JPEG/blur
            # performance to avoid selecting only for clean data.
            jpeg, _, _ = evaluate(
                model, robust_loaders["jpeg_q30"], fake_index, device
            )
            blur, _, _ = evaluate(
                model, robust_loaders["blur_r4"], fake_index, device
            )
            score = (
                0.6 * clean_metrics["accuracy"]
                + 0.2 * jpeg["accuracy"]
                + 0.2 * blur["accuracy"]
            )
            row.update({
                "val_jpeg_q30_accuracy": jpeg["accuracy"],
                "val_blur_r4_accuracy": blur["accuracy"],
                "selection_score": score,
            })
        else:
            # Preserve V1's original checkpoint-selection criterion.
            score = -clean_metrics["loss"]

        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch

        history.append(row)
        message = (
            f"Epoch {epoch}/{args.epochs}: "
            f"train={row['train_accuracy']:.4f}, "
            f"val={row['val_accuracy']:.4f}, "
            f"val_loss={row['val_loss']:.4f}"
        )
        if args.strategy == "v2":
            message += (
                f", val_jpeg30={row['val_jpeg_q30_accuracy']:.4f}, "
                f"val_blur4={row['val_blur_r4_accuracy']:.4f}, "
                f"score={row['selection_score']:.4f}"
            )
        print(message, flush=True)

    # Evaluate the selected checkpoint on the held-out clean test set.
    model.load_state_dict(best_state)
    test_metrics, labels, fake_probs = evaluate(
        model, loaders["test"], fake_index, device
    )

    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model_name": args.model,
        "state_dict": model.state_dict(),
        "class_to_idx": data["train"].class_to_idx,
        "strategy": args.strategy,
        "best_epoch": best_epoch,
        "seed": 42,
    }, checkpoint_path)

    save_csv(
        metrics_path,
        [{"model": run_name, **test_metrics}],
    )
    save_csv(history_path, history)

    labels_by_id = {
        v: k for k, v in data["test"].class_to_idx.items()
    }
    predictions = [
        {
            "filename": str(path),
            "actual_label": labels_by_id[actual],
            "predicted_label": "fake" if prob >= 0.5 else "real",
            "fake_probability": prob,
        }
        for (path, _), actual, prob in zip(
            data["test"].samples, labels, fake_probs
        )
    ]
    save_csv(predictions_path, predictions)

    print(f"Best epoch: {best_epoch}")
    print(f"Clean test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Saved: {checkpoint_path} and 3 results CSVs")


if __name__ == "__main__":
    main()
