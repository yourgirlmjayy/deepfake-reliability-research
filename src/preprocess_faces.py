import argparse
import csv
from pathlib import Path

import cv2
import dlib
import numpy as np
from skimage import transform as trans


VIDEO_ROOT = Path(
    "../deepfake_reliability_project/data/splits"
)

OUTPUT_ROOT = Path(
    "../deepfake_reliability_project/data/aligned_faces"
)

PREDICTOR_PATH = Path(
    "dlib_tools/shape_predictor_81_face_landmarks.dat"
)

DEFAULT_NUM_FRAMES = 32
OUTPUT_SIZE = 256
ALIGN_SCALE = 1.3


def get_keypoints(image_rgb, face, predictor):
    shape = predictor(image_rgb, face)

    # Same five landmark locations used by DeepfakeBench
    indices = [37, 44, 30, 49, 55]

    points = np.array(
        [
            [
                shape.part(index).x,
                shape.part(index).y,
            ]
            for index in indices
        ],
        dtype=np.float32,
    )

    return points


def align_and_crop(
    image_rgb,
    landmarks,
    output_size=OUTPUT_SIZE,
    scale=ALIGN_SCALE,
):
    # Reference points used by DeepfakeBench
    target_size = [112, 112]

    destination = np.array(
        [
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041],
        ],
        dtype=np.float32,
    )

    if target_size[1] == 112:
        destination[:, 0] += 8.0

    destination[:, 0] *= (
        output_size / target_size[0]
    )

    destination[:, 1] *= (
        output_size / target_size[1]
    )

    target_size = [output_size, output_size]

    margin_rate = scale - 1.0

    x_margin = (
        target_size[0] * margin_rate / 2.0
    )

    y_margin = (
        target_size[1] * margin_rate / 2.0
    )

    destination[:, 0] += x_margin
    destination[:, 1] += y_margin

    destination[:, 0] *= (
        target_size[0]
        / (target_size[0] + 2 * x_margin)
    )

    destination[:, 1] *= (
        target_size[1]
        / (target_size[1] + 2 * y_margin)
    )

    transform = trans.SimilarityTransform()

    success = transform.estimate(
        landmarks.astype(np.float32),
        destination,
    )

    if not success:
        return None

    matrix = transform.params[:2, :]

    aligned = cv2.warpAffine(
        image_rgb,
        matrix,
        (output_size, output_size),
    )

    return aligned


def process_video(
    video_path,
    split,
    label,
    detector,
    predictor,
    num_frames,
    writer,
):
    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        print(
            f"Could not open: {video_path}"
        )
        return 0, num_frames

    total_frames = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if total_frames <= 0:
        capture.release()

        print(
            f"No readable frames: {video_path}"
        )

        return 0, num_frames

    frame_indices = np.linspace(
        0,
        total_frames - 1,
        num_frames,
        endpoint=True,
        dtype=int,
    )

    frame_indices = set(
        frame_indices.tolist()
    )

    output_dir = (
        OUTPUT_ROOT
        / split
        / label
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved = 0
    skipped = 0

    for frame_number in range(
        total_frames
    ):
        success, frame = capture.read()

        if not success:
            break

        if frame_number not in frame_indices:
            continue

        image_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        faces = detector(
            image_rgb,
            1,
        )

        if len(faces) == 0:
            skipped += 1

            writer.writerow(
                [
                    split,
                    label,
                    video_path.name,
                    frame_number,
                    "",
                    "no_face",
                ]
            )

            continue

        # DeepfakeBench selects the largest face
        face = max(
            faces,
            key=lambda rect:
            rect.width() * rect.height(),
        )

        landmarks = get_keypoints(
            image_rgb,
            face,
            predictor,
        )

        aligned_rgb = align_and_crop(
            image_rgb,
            landmarks,
        )

        if aligned_rgb is None:
            skipped += 1

            writer.writerow(
                [
                    split,
                    label,
                    video_path.name,
                    frame_number,
                    "",
                    "alignment_failed",
                ]
            )

            continue

        aligned_bgr = cv2.cvtColor(
            aligned_rgb,
            cv2.COLOR_RGB2BGR,
        )

        filename = (
            f"{video_path.stem}"
            f"_frame{frame_number:04d}.png"
        )

        output_path = (
            output_dir / filename
        )

        cv2.imwrite(
            str(output_path),
            aligned_bgr,
        )

        writer.writerow(
            [
                split,
                label,
                video_path.name,
                frame_number,
                str(output_path),
                "saved",
            ]
        )

        saved += 1

    capture.release()

    print(
        f"{split}/{label}: "
        f"{video_path.name} -> "
        f"{saved} saved, "
        f"{skipped} skipped"
    )

    return saved, skipped


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num-frames",
        type=int,
        default=DEFAULT_NUM_FRAMES,
    )

    parser.add_argument(
        "--limit-videos",
        type=int,
        default=None,
        help=(
            "Optional number of videos "
            "per split/label for testing."
        ),
    )

    args = parser.parse_args()

    if not PREDICTOR_PATH.exists():
        raise FileNotFoundError(
            f"Predictor not found: "
            f"{PREDICTOR_PATH}"
        )

    detector = (
        dlib.get_frontal_face_detector()
    )

    predictor = dlib.shape_predictor(
        str(PREDICTOR_PATH)
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path = (
        OUTPUT_ROOT
        / "preprocessing_manifest.csv"
    )

    total_saved = 0
    total_skipped = 0

    with open(
        manifest_path,
        "w",
        newline="",
    ) as manifest_file:
        writer = csv.writer(
            manifest_file
        )

        writer.writerow(
            [
                "split",
                "label",
                "video",
                "frame_number",
                "output_path",
                "status",
            ]
        )

        for split in [
            "train",
            "val",
            "test",
        ]:
            for label in [
                "real",
                "fake",
            ]:
                video_dir = (
                    VIDEO_ROOT
                    / split
                    / label
                )

                videos = sorted(
                    video_dir.glob("*.mp4")
                )

                if (
                    args.limit_videos
                    is not None
                ):
                    videos = videos[
                        :args.limit_videos
                    ]

                for video_path in videos:
                    saved, skipped = (
                        process_video(
                            video_path,
                            split,
                            label,
                            detector,
                            predictor,
                            args.num_frames,
                            writer,
                        )
                    )

                    total_saved += saved
                    total_skipped += skipped

    print(
        "\nFace preprocessing complete."
    )

    print(
        f"Saved: {total_saved}"
    )

    print(
        f"Skipped: {total_skipped}"
    )

    print(
        f"Manifest: {manifest_path}"
    )


if __name__ == "__main__":
    main()