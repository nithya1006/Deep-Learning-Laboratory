# Lab 5 — Comprehensive CNN Study with MobileNetV2

A systematic study of how weight initialization, regularization, optimization,
hyperparameters, transfer learning and cross-validation affect image
classification performance. A single architecture (MobileNetV2) and a single
dataset (Oxford-IIIT Pet) are held fixed so that each design choice can be
varied one at a time.

## Dataset

**Oxford-IIIT Pet** — 37 cat and dog breeds, loaded through `tensorflow_datasets`.

| | Images |
|---|---|
| Train | 2,944 |
| Validation | 736 |
| Test (held out) | 3,669 |

Images are resized to 224×224×3 and normalized with MobileNetV2's own
`preprocess_input` (scales pixels to [-1, 1], matching ImageNet pretraining).
The train/validation split is a deterministic 80/20 index permutation seeded at
42. The test split is untouched until final evaluation.

## Method

MobileNetV2 pretrained on ImageNet, with a new classifier head:

````
Dense(256) → [BatchNorm] → ReLU → [Dropout] → Dense(37, softmax)
````

Every study except transfer learning trains only this head. The frozen
convolutional base is therefore run over the dataset **once** and its 1280-dim
pooled features are cached to disk. Each subsequent training run then takes
seconds rather than minutes, which is what makes roughly 60 separate runs
practical — 4 initializations, 4 regularizers, 4 optimizers, 10 hyperparameter
settings, 4 fine-tuning configurations, and 8 configurations × 5 CV folds.


## What is studied

| Section | Study | Variants |
|---|---|---|
| 5 | Weight initialization | Zero, Random, Xavier/Glorot, He |
| 6 | Regularization | None, L2, Dropout, BatchNorm |
| 7 | Batch normalization | With vs without (+ numerical worked example) |
| 8 | Optimization | SGD, Momentum, RMSProp, Adam |
| 9 | Hyperparameters | LR, batch size, dropout, fine-tune LR, unfreeze depth |
| 10 | Transfer learning | Feature extraction vs fine-tuning |
| 11 | Model selection | 5-fold stratified cross-validation |
| 12 | Final evaluation | Held-out test set, confusion matrix |
| 16 | Additional exercise | Two new configurations vs the selected one |

## Running it

Open `DeepLearning_Ex5.ipynb` in [Google Colab](https://colab.research.google.com),
set **Runtime → Change runtime type → T4 GPU**, then run all cells.

## Outputs

| Plot | Content |
|---|---|
| 1 | Training loss vs epoch, by initialization |
| 2 | Validation accuracy vs epoch, by initialization |
| 3 | Train vs validation accuracy — generalization gap, by regularizer |
| 4 | Train vs validation loss, by regularizer |
| 5 | Validation accuracy with vs without BatchNorm |
| 6 | Training loss vs epoch, by optimizer |
| 7 | Validation accuracy vs epoch, by optimizer |
| 8 | Learning rate vs validation accuracy |
| 9 | Batch size vs validation accuracy |
| 10 | Dropout rate vs validation accuracy |
| 11 | Feature extraction vs fine-tuning |
| 12 | Train/validation loss before and after fine-tuning |
| 13 | 5-fold CV accuracy with SD error bars |
| 14 | Confusion matrix, 37 breeds |
| 15 | Misclassified test images (optional) |
| 16 | Additional exercise comparison |
| 17 | Fine-tuning LR and unfreezing depth |

All figures are saved to `Plots/` as 600 dpi PNGs and zipped for download by the
final cell.

Tables printed inline: initialization convergence speed, generalization gap,
optimizer comparison, convolution output sizes, hyperparameter sweeps,
fine-tuning grid, 5-fold CV folds, final metrics, per-class best/worst breeds,
confused breed pairs, overall results, additional exercise.

## Results

| Metric | Value |
|---|---|
| Mean CV accuracy | 92.04% |
| CV standard deviation | 1.02% |
| Test accuracy | 90.65% |
| Precision (macro) | 0.9077 |
| Recall (macro) | 0.9059 |
| F1-score (macro) | 0.9057 |
| Classifier head parameters | 337,445 |
| Frozen base parameters | 2,257,984 |
| Total parameters | 2,595,429 |

## Dependencies

TensorFlow/Keras · TensorFlow Datasets · NumPy · Matplotlib · scikit-learn

## Files

````
Lab 5/
├── README.md
├── DeepLearning_Ex5.ipynb
└── Plots/
````