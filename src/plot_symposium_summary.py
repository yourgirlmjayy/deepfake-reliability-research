from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_FILE = Path("results/transformation_results.csv")
OUTPUT_FILE = Path("results/figures/symposium_accuracy_drop.png")

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RESULTS_FILE)

# Use the strongest tested severity for each transformation
conditions = {
    "JPEG\ncompression": "jpeg_q30",
    "Cropping": "crop_70",
    "Resizing": "resize_30",
    "Blur": "blur_r4",
}

accuracy_drops = []

for label, condition in conditions.items():
    drops = []

    for model in ["efficientnet_b0", "resnet50"]:
        clean_accuracy = df[
            (df["model"] == model)
            & (df["condition"] == "clean")
        ]["accuracy"].iloc[0]

        transformed_accuracy = df[
            (df["model"] == model)
            & (df["condition"] == condition)
        ]["accuracy"].iloc[0]

        drops.append((clean_accuracy - transformed_accuracy) * 100)

    # Average the drop across the two models
    accuracy_drops.append(sum(drops) / len(drops))


plt.figure(figsize=(9, 6))

bars = plt.bar(
    conditions.keys(),
    accuracy_drops,
)

plt.title(
    "How Common Image Edits Affect Deepfake Detection",
    fontsize=16,
    pad=15,
)

plt.ylabel(
    "Drop in Detection Accuracy\n(percentage points)",
    fontsize=12,
)

plt.xlabel("Image Edit", fontsize=12)

plt.ylim(0, 50)
plt.grid(axis="y", alpha=0.25)

for bar, value in zip(bars, accuracy_drops):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{value:.1f}",
        ha="center",
        fontsize=12,
    )

plt.figtext(
    0.5,
    0.01,
    "Higher bars mean the detectors became less accurate after the image was changed.",
    ha="center",
    fontsize=10,
)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(OUTPUT_FILE, dpi=300)
plt.close()

print(f"Saved: {OUTPUT_FILE}")