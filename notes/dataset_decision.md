# Dataset decision

## Selected dataset: FaceForensics++

FaceForensics++ was selected because it matches the approved project better than a generic AI-image dataset. It contains original facial videos and manipulated versions created with Deepfakes, Face2Face, FaceSwap, and NeuralTextures. It also provides multiple compression levels, which directly supports the project's focus on whether detector performance changes after realistic media transformations.

## Starting subset

For Progress Report 1, start with:
- REAL: original sequences
- FAKE: Deepfakes
- baseline quality: c23

This keeps the first experiment manageable while preserving the exact research direction. After the pipeline works, expand to additional manipulation methods and c40.

## Important data-splitting rule

Train/validation/test separation must happen at the VIDEO level before frames are extracted. Frames from the same source video should never appear in both training and testing because that would inflate performance and create data leakage.
