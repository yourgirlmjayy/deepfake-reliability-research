# Evaluating the Reliability of Deepfake Detection for Real-World Digital Evidence

Senior Seminar project starter repository.

## Research question

How reliable is a deepfake image detector when facial images are changed by common real-world transformations such as compression, cropping, resizing, blur, screenshots/re-saving, and visual filters?

## Primary dataset

FaceForensics++ is the primary dataset. The first working subset will use:
- original videos as the REAL class
- Deepfakes videos as the FAKE class
- c23 compression as the baseline condition

Later experiments can add Face2Face, FaceSwap, NeuralTextures, and c40 compression.

## Baseline model

EfficientNet-B0 with transfer learning in PyTorch.

## Initial experiment design

1. Split data by source video before extracting frames.
2. Extract a fixed number of frames per video.
3. Train the baseline detector on unchanged c23 frames.
4. Evaluate on unchanged test frames.
5. Apply one transformation at a time to the same test frames.
6. Compare accuracy, precision, recall, F1, ROC-AUC, and confidence.
7. Record false positives and false negatives for each condition.

## Repository structure

- `src/transformations.py`: image transformations used for robustness testing
- `src/extract_frames.py`: extracts evenly spaced frames from videos
- `notebooks/01_preprocessing_demo.ipynb`: starter notebook for dataset checks and transformations
- `notes/dataset_decision.md`: why FaceForensics++ was selected
- `notes/experiment_plan.md`: planned experiments and metrics
- `evidence/preprocessing_smoke_test.png`: proof that the transformation pipeline runs on a generated test image

The smoke-test image is only a code test. It is not a FaceForensics++ sample and should not be reported as a model result.
