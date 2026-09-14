# Experiment plan

## Phase 1: Data pipeline
- Obtain FaceForensics++ access.
- Download a small c23 subset for pipeline testing.
- Separate REAL and FAKE videos.
- Follow the official train/validation/test split for final experiments.
- Extract a fixed number of frames per video.

## Phase 2: Baseline
- Fine-tune EfficientNet-B0 for binary classification.
- Evaluate unchanged c23 test frames.
- Save class predictions and fake-class probabilities.

## Phase 3: Robustness tests
Test the same held-out images after one transformation at a time:
- JPEG compression
- center crop
- downscale + upscale
- Gaussian blur
- contrast filter
- screenshot-like resampling/re-save

Use several severities where useful rather than only one setting.

## Phase 4: Analysis
For each condition, record:
- accuracy
- precision
- recall
- F1 score
- ROC-AUC
- average prediction confidence
- false positives
- false negatives

The main analysis is the CHANGE in performance from the unchanged baseline to each transformed condition.

## Phase 5: Extension
If time allows:
- compare EfficientNet-B0 with Xception
- test Face2Face, FaceSwap, and NeuralTextures
- test c40
- compare which manipulations and transformations cause the largest reliability drop
