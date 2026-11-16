
"""Evaluate trained deepfake detectors on unseen Mirage-Test human faces."""

import argparse
import csv
from pathlib import Path

import dlib
import numpy as np
import torch
from datasets import load_dataset, Image as HFImage
from huggingface_hub import hf_hub_download
from datasets import load_dataset
from PIL import Image, ImageOps

from evaluate_transformations import (
    build_model_for_evaluation,
    evaluate_condition,
)
from preprocess_faces import (
    PREDICTOR_PATH,
    align_and_crop,
    get_keypoints,
)
from train_baseline import get_device


DATA_ROOT = Path("../deepfake_reliability_project/data/mirage_test_faces")
RESULTS_ROOT = Path("results")

MODEL_RUNS = {
    "efficientnet_b0": ("baseline", "augmented_v1", "augmented_v2"),
    "resnet50": ("baseline", "augmented_v1", "augmented_v2"),
}


def prepare_faces(per_class):
    """Collect labeled images and align their largest detectable face."""
    if not PREDICTOR_PATH.is_file():
        raise FileNotFoundError(
            f"Face landmark predictor missing: {PREDICTOR_PATH}"
        )

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(str(PREDICTOR_PATH))

    manifest = DATA_ROOT / "source_manifest.csv"
    counts = {"real": 0, "fake": 0}
    processed_sources = set()

    # Resume collection from the existing manifest when possible.
    if manifest.exists():
        with manifest.open(newline="") as file:
            for row in csv.DictReader(file):
                if Path(row["aligned_path"]).exists():
                    processed_sources.add(row["source_path"])
                    counts[row["label"]] += 1

    if all(count >= per_class for count in counts.values()):
        print(f"Using existing Mirage-Test images: {counts}")
        return

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    
    dataset = load_dataset(
        "Yunncheng/Mirage-Test",
        split="test",
        streaming=True,
    ).cast_column("image", HFImage(decode=False))

    labels = {"0_real": "real", "1_fake": "fake"}
    skipped_no_face = 0
    scanned = 0

    with manifest.open("a", newline="") as file:
        writer = csv.writer(file)

        if file.tell() == 0:
            writer.writerow(
                ("label", "source_path", "aligned_path")
            )

        for item in dataset:
            scanned += 1

            if item.get("content_type") != "Human":
                continue

            label = labels.get(item.get("is_real"))
            source_path = str(item.get("file_name", ""))

            if (
                label is None
                or counts[label] >= per_class
                or source_path in processed_sources
            ):
                continue

            image_path = hf_hub_download(
                repo_id="Yunncheng/Mirage-Test",
                filename=source_path,
                repo_type="dataset",
            )

            image = ImageOps.exif_transpose(
                Image.open(image_path)
            ).convert("RGB")

            rgb = np.asarray(image)
            faces = detector(rgb, 1)

            if not faces:
                skipped_no_face += 1
                continue

            # Use the same landmark-based alignment as the training pipeline.
            face = max(
                faces,
                key=lambda f: f.width() * f.height(),
            )

            landmarks = get_keypoints(
                rgb, face, predictor
            )

            aligned = align_and_crop(rgb, landmarks)

            if aligned is None:
                continue

            target = (
                DATA_ROOT
                / label
                / f"{label}_{counts[label] + 1:04d}.png"
            )

            target.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            if target.exists():
                raise FileExistsError(
                    f"Existing untracked image: {target}"
                )

            Image.fromarray(aligned).save(target)

            # Record the original dataset path for traceability.
            writer.writerow(
                (label, source_path, str(target))
            )
            file.flush()

            processed_sources.add(source_path)
            counts[label] += 1

            if sum(counts.values()) % 20 == 0:
                print(
                    f"Collected: {counts} | scanned: {scanned}",
                    flush=True,
                )

            if all(
                count >= per_class
                for count in counts.values()
            ):
                break

    print(
        f"Collected: {counts}; "
        f"skipped images without faces: {skipped_no_face}"
    )

    if any(count < per_class for count in counts.values()):
        raise RuntimeError(
            "Insufficient aligned images for the requested sample."
        )


def evaluate_models(per_class):
    """Evaluate all six checkpoints on the same external test images."""
    result_path = (
        RESULTS_ROOT
        / f"mirage_test_{per_class}_per_class_results.csv"
    )

    if result_path.exists():
        print(f"Existing results preserved: {result_path}")
        return

    results = []
    device = get_device()

    print(f"Evaluating on {device}")

    for architecture, versions in MODEL_RUNS.items():
        for version in versions:
            run_name = (
                f"{architecture}_baseline"
                if version == "baseline"
                else f"{architecture}_{version}"
            )

            checkpoint_path = (
                Path("models") / f"{run_name}.pth"
            )

            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"Missing checkpoint: {checkpoint_path}"
                )

            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )

            if checkpoint["model_name"] != architecture:
                raise ValueError(
                    f"Checkpoint architecture mismatch: {checkpoint_path}"
                )

            model, preprocessing = build_model_for_evaluation(
                architecture
            )

            model.load_state_dict(
                checkpoint["state_dict"]
            )

            model = model.to(device)

            metrics = evaluate_condition(
                model,
                preprocessing,
                DATA_ROOT,
                checkpoint["class_to_idx"],
                device,
            )

            results.append({
                "model": architecture,
                "version": version,
                "dataset": "Mirage-Test Human aligned faces",
                "images_per_class": per_class,
                **metrics,
            })

            print(
                f"{run_name}: "
                f"accuracy={metrics['accuracy']:.4f}, "
                f"FP={metrics['false_positives']}, "
                f"FN={metrics['false_negatives']}"
            )

            del model

    RESULTS_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    with result_path.open("x", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys(),
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved external evaluation: {result_path}")


def main():
    """Collect a balanced external sample and evaluate saved models."""
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--per-class",
        type=int,
        default=25,
    )

    args = parser.parse_args()

    if args.per_class < 1:
        parser.error("--per-class must be positive")

    prepare_faces(args.per_class)
    evaluate_models(args.per_class)


if __name__ == "__main__":
    main()
