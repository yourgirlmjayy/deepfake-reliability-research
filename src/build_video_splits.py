from pathlib import Path
import random
import shutil
import os

DATA_ROOT = Path("../deepfake_reliability_project/data")

REAL_ROOT = (
    DATA_ROOT
    / "raw/original_sequences/youtube/c23/videos"
)

FAKE_ROOT = (
    DATA_ROOT
    / "raw/manipulated_sequences/Deepfakes/c23/videos"
)

SPLIT_ROOT = DATA_ROOT / "splits"

SEED = 42


def find(parent, x):
    if parent[x] != x:
        parent[x] = find(parent, parent[x])
    return parent[x]


def union(parent, a, b):
    root_a = find(parent, a)
    root_b = find(parent, b)

    if root_a != root_b:
        parent[root_b] = root_a


real_files = {
    path.stem: path
    for path in REAL_ROOT.glob("*.mp4")
}

fake_files = list(FAKE_ROOT.glob("*.mp4"))

ids = set(real_files.keys())

fake_pairs = []

for path in fake_files:
    parts = path.stem.split("_")

    if len(parts) != 2:
        print(f"Skipping unexpected filename: {path.name}")
        continue

    a, b = parts

    ids.add(a)
    ids.add(b)
    fake_pairs.append((a, b, path))


# Build connected groups so related source IDs
# can never end up in different splits.
parent = {video_id: video_id for video_id in ids}

for a, b, _ in fake_pairs:
    union(parent, a, b)


groups = {}

for video_id in ids:
    root = find(parent, video_id)
    groups.setdefault(root, set()).add(video_id)


groups = list(groups.values())

random.seed(SEED)
random.shuffle(groups)

total_groups = len(groups)

train_end = round(total_groups * 0.70)
val_end = train_end + round(total_groups * 0.15)

split_groups = {
    "train": groups[:train_end],
    "val": groups[train_end:val_end],
    "test": groups[val_end:],
}


# Only delete the OLD generated split folders.
# Raw downloaded videos are untouched.
if SPLIT_ROOT.exists():
    shutil.rmtree(SPLIT_ROOT)


for split in ["train", "val", "test"]:
    for label in ["real", "fake"]:
        (
            SPLIT_ROOT / split / label
        ).mkdir(parents=True, exist_ok=True)


for split, group_list in split_groups.items():
    split_ids = set().union(*group_list)

    # Original videos
    for video_id in sorted(split_ids):
        source = real_files.get(video_id)

        if source is None:
            continue

        destination = (
            SPLIT_ROOT
            / split
            / "real"
            / source.name
        )

        os.symlink(
            source.resolve(),
            destination,
        )

    # Deepfake videos
    for a, b, source in fake_pairs:
        if a in split_ids and b in split_ids:
            destination = (
                SPLIT_ROOT
                / split
                / "fake"
                / source.name
            )

            os.symlink(
                source.resolve(),
                destination,
            )


print("\nVideo split complete.")

for split in ["train", "val", "test"]:
    real_count = len(
        list((SPLIT_ROOT / split / "real").glob("*.mp4"))
    )

    fake_count = len(
        list((SPLIT_ROOT / split / "fake").glob("*.mp4"))
    )

    print(
        f"{split}: "
        f"{real_count} real, "
        f"{fake_count} fake"
    )
    