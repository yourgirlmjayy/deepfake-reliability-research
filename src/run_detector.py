import os
import csv
import torch
import torch.nn as nn

from PIL import Image
from torchvision import models, transforms


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_URL = (
    "https://huggingface.co/Xicor9/efficientnet-b0-ffpp-c23/"
    "resolve/main/efficientnet_b0_ffpp_c23.pth"
)

# Load pretrained FaceForensics++ C23 detector
state_dict = torch.hub.load_state_dict_from_url(
    MODEL_URL,
    map_location=DEVICE
)

model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    2
)

model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    x = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probabilities = torch.softmax(logits, dim=1)[0]

    real_probability = probabilities[0].item()
    fake_probability = probabilities[1].item()

    prediction = "fake" if fake_probability >= 0.5 else "real"

    return prediction, real_probability, fake_probability


def evaluate_folder(folder, actual_label, writer):
    if not os.path.exists(folder):
        print(f"Folder not found: {folder}")
        return

    for filename in sorted(os.listdir(folder)):
        if not filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            continue

        path = os.path.join(folder, filename)

        prediction, real_prob, fake_prob = predict_image(path)

        print(
            f"{filename}: "
            f"actual={actual_label}, "
            f"predicted={prediction}, "
            f"fake_probability={fake_prob:.4f}"
        )

        writer.writerow([
            filename,
            actual_label,
            prediction,
            real_prob,
            fake_prob
        ])


os.makedirs("results", exist_ok=True)

output_file = "results/test_baseline_detector_results.csv"

with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "filename",
        "actual_label",
        "predicted_label",
        "real_probability",
        "fake_probability"
    ])

    evaluate_folder(
       "../deepfake_reliability_project/data/face_crops/test/real" ,
        "real",
        writer
    )

    evaluate_folder(
        "../deepfake_reliability_project/data/face_crops/test/fake",
        "fake",
        writer
    )

print(f"\nResults saved to {output_file}")