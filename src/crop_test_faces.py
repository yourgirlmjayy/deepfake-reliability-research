import os
import cv2

INPUT_ROOT = "../deepfake_reliability_project/data/frames/test"
OUTPUT_ROOT = "../deepfake_reliability_project/data/face_crops/test"

cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_detector = cv2.CascadeClassifier(cascade_path)


def crop_face(image, x, y, w, h, scale=1.3):
    height, width = image.shape[:2]

    size = int(max(w, h) * scale)
    center_x = x + w // 2
    center_y = y + h // 2

    x1 = max(center_x - size // 2, 0)
    y1 = max(center_y - size // 2, 0)

    x2 = min(x1 + size, width)
    y2 = min(y1 + size, height)

    return image[y1:y2, x1:x2]


for label in ["real", "fake"]:
    input_dir = os.path.join(INPUT_ROOT, label)
    output_dir = os.path.join(OUTPUT_ROOT, label)

    os.makedirs(output_dir, exist_ok=True)

    for filename in sorted(os.listdir(input_dir)):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        path = os.path.join(input_dir, filename)
        image = cv2.imread(path)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
        )

        if len(faces) == 0:
            print(f"No face found: {filename}")
            continue

        # use largest detected face
        x, y, w, h = max(
            faces,
            key=lambda box: box[2] * box[3]
        )

        cropped = crop_face(image, x, y, w, h)

        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, cropped)

        print(f"Saved: {output_path}")