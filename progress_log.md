## Progress Report 1
- Confirmed the project scope: detector reliability after common real-world image transformations.
- Selected FaceForensics++ as the primary baseline dataset.
- Selected c23 as the baseline working condition and c40 as a later severe-compression condition.
- Selected EfficientNet-B0 as the first baseline model.
- Defined evaluation metrics: accuracy, precision, recall, F1, ROC-AUC, confidence, false positives, and false negatives.
- Implemented image transformations for JPEG compression, cropping, resizing, blur, contrast filtering, and a screenshot/re-share proxy.
- Implemented evenly spaced video-frame extraction.
- Downloaded 10 original and 10 Deepfakes c23 videos from FaceForensics++.
- Extracted 5 evenly spaced frames per video, producing 50 real and 50 fake frames.
- Visually inspected samples from five different source videos.
- Applied the image transformation pipeline to an actual FaceForensics++ deepfake frame.

## Still pending
- Training the baseline detector.
- Evaluating the baseline detector on unchanged test data.
- Testing detector performance after image transformations.
- Testing whether the detector generalizes to more difficult or unseen deepfake examples.
