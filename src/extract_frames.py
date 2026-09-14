from pathlib import Path
import cv2
import numpy as np

def extract_even_frames(video_path, output_dir, num_frames=10, prefix=None):
    """
    Extract evenly spaced frames from one video.
    Returns a list of saved frame paths.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise ValueError(f"Could not read frames from {video_path}")

    indices = np.linspace(0, total - 1, num=min(num_frames, total), dtype=int)
    saved = []
    stem = prefix or video_path.stem

    for i, frame_index in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
        if not ok:
            continue
        out = output_dir / f"{stem}_frame_{i:03d}.jpg"
        cv2.imwrite(str(out), frame)
        saved.append(out)

    cap.release()
    return saved
