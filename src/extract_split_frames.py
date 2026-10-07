from pathlib import Path
import cv2

VIDEO_ROOT = Path("data/splits")
FRAME_ROOT = Path("data/frames")
NUM_FRAMES = 5

for split in ["train", "val", "test"]:
    for label in ["real", "fake"]:
        video_dir = VIDEO_ROOT / split / label
        output_dir = FRAME_ROOT / split / label
        output_dir.mkdir(parents=True, exist_ok=True)

        for video_path in sorted(video_dir.glob("*.mp4")):
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames <= 0:
                print(f"Could not read {video_path}")
                cap.release()
                continue

            # 5 evenly spaced positions, avoiding first/last frame
            positions = [
                int((i + 1) * total_frames / (NUM_FRAMES + 1))
                for i in range(NUM_FRAMES)
            ]

            saved = 0

            for i, frame_number in enumerate(positions, start=1):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                success, frame = cap.read()

                if success:
                    filename = f"{video_path.stem}_frame{i}.jpg"
                    output_path = output_dir / filename
                    cv2.imwrite(str(output_path), frame)
                    saved += 1

            cap.release()

            print(
                f"{split}/{label}: {video_path.name} -> "
                f"{saved} frames"
            )

print("\nFrame extraction complete.")