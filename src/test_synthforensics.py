from datasets import load_dataset

print("Connecting to SynthForensics sample...")

ds = load_dataset(
    "SynthForensics/SynthForensics_sample",
    split="train",
    streaming=True
)

print("\nDataset:")
print(ds)

example = next(iter(ds))

print("\nAvailable fields:")
print(example.keys())

print("\nFirst example:")
for key, value in example.items():
    if key == "video":
        print(f"{key}: [video object]")
    else:
        print(f"{key}: {value}")