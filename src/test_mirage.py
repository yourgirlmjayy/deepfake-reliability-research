from datasets import load_dataset

print("Connecting to Mirage-Test...")

ds = load_dataset(
    "Yunncheng/Mirage-Test",
    split="test",
    streaming=True
)

real_examples = []
fake_examples = []

for example in ds:
    if example["content_type"] != "Human":
        continue

    if example["is_real"] == "0_real" and len(real_examples) < 5:
        real_examples.append(example)

    elif example["is_real"] == "1_fake" and len(fake_examples) < 5:
        fake_examples.append(example)

    if len(real_examples) == 5 and len(fake_examples) == 5:
        break

print("\nREAL HUMAN IMAGES")
for example in real_examples:
    print(example["file_name"])

print("\nAI-GENERATED HUMAN IMAGES")
for example in fake_examples:
    print(example["file_name"])

print(f"\nCollected {len(real_examples)} real and {len(fake_examples)} fake images.")