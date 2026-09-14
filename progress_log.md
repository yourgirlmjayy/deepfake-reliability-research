# Progress log

## Progress Report 1
- Confirmed the project scope: detector reliability after common real-world image transformations.
- Selected FaceForensics++ as the primary dataset.
- Selected c23 as the baseline working condition and c40 as a later severe-compression condition.
- Selected EfficientNet-B0 as the first baseline model.
- Defined evaluation metrics: accuracy, precision, recall, F1, ROC-AUC, confidence, false positives, and false negatives.
- Implemented image transformations for JPEG compression, cropping, resizing, blur, contrast filtering, and a screenshot/re-share proxy.
- Implemented evenly spaced video-frame extraction.
- Ran the transformation code on a generated test image to verify that the preprocessing pipeline executes.

## Still pending
- FaceForensics++ access/download.
- Running the pipeline on actual FaceForensics++ frames.
- Training the baseline detector.
- Recording first model results.
