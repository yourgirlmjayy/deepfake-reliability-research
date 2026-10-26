from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_FILE = Path("results/transformation_results.csv")
OUTPUT_DIR = Path("results/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


df = pd.read_csv(RESULTS_FILE)


PLOTS = {
    "jpeg": {
        "conditions": ["clean", "jpeg_q70", "jpeg_q50", "jpeg_q30"],
        "labels": ["Clean", "Q70", "Q50", "Q30"],
        "title": "Effect of JPEG Compression on Detection Accuracy",
    },
    "crop": {
        "conditions": ["clean", "crop_90", "crop_80", "crop_70"],
        "labels": ["Clean", "90%", "80%", "70%"],
        "title": "Effect of Cropping on Detection Accuracy",
    },
    "resize": {
        "conditions": ["clean", "resize_75", "resize_50", "resize_30"],
        "labels": ["Clean", "75%", "50%", "30%"],
        "title": "Effect of Resizing on Detection Accuracy",
    },
    "blur": {
        "conditions": ["clean", "blur_r1", "blur_r2", "blur_r4"],
        "labels": ["Clean", "Radius 1", "Radius 2", "Radius 4"],
        "title": "Effect of Gaussian Blur on Detection Accuracy",
    },
}


for plot_name, config in PLOTS.items():
    plt.figure(figsize=(8, 5))

    for model_name in ["efficientnet_b0", "resnet50"]:
        model_df = df[
            (df["model"] == model_name)
            & (df["condition"].isin(config["conditions"]))
        ].copy()

        accuracy_lookup = dict(
            zip(model_df["condition"], model_df["accuracy"])
        )

        accuracies = [
            accuracy_lookup[condition] * 100
            for condition in config["conditions"]
        ]

        display_name = (
            "EfficientNet-B0"
            if model_name == "efficientnet_b0"
            else "ResNet-50"
        )

        plt.plot(
            config["labels"],
            accuracies,
            marker="o",
            label=display_name,
        )

    plt.title(config["title"])
    plt.xlabel("Severity")
    plt.ylabel("Accuracy (%)")
    plt.ylim(45, 100)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    output_path = OUTPUT_DIR / f"{plot_name}_accuracy.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")