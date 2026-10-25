from pathlib import Path

from PIL import Image

from transformations import TRANSFORMATIONS


INPUT_ROOT = Path(
    "../deepfake_reliability_project/data/aligned_faces/test"
)

OUTPUT_ROOT = Path(
    "../deepfake_reliability_project/data/transformed_test"
)


def main():
    total = 0

    for condition, transform_fn in TRANSFORMATIONS.items():
        print(f"\n=== {condition} ===")

        for label in ["real", "fake"]:
            input_dir = INPUT_ROOT / label
            output_dir = OUTPUT_ROOT / condition / label

            output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            saved = 0

            for image_path in sorted(
                input_dir.glob("*.png")
            ):
                with Image.open(image_path) as image:
                    image = image.convert("RGB")

                    transformed = transform_fn(image)

                    output_path = (
                        output_dir / image_path.name
                    )

                    # Save transformed test inputs as PNG so
                    # we do not introduce another uncontrolled
                    # JPEG encoding step after the intended
                    # transformation.
                    transformed.save(
                        output_path,
                        format="PNG",
                    )

                saved += 1
                total += 1

            print(
                f"{label}: {saved} images"
            )

    print(
        f"\nTransformation generation complete: "
        f"{total} images"
    )


if __name__ == "__main__":
    main()