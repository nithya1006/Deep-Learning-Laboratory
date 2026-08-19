# Experiment 3 — Convolutional Neural Networks for Image Classification

## Objective

To understand the working principle of Convolutional Neural Networks by implementing convolution, pooling, feature map visualization, and image classification using PyTorch.

## Dataset

CIFAR-10

- Training Images: 50,000
- Testing Images: 10,000
- Classes: 10
- Image Size: 32 × 32 × 3

## Model

```
Input → Conv → ReLU → MaxPool → Conv → ReLU → MaxPool → Flatten → Dense → Softmax
```

- Optimizer: Adam
- Epochs: 20
- Batch Size: 32

## Tasks

1. Load CIFAR-10, display sample images and plot class distribution
2. Implement a convolution layer and compare 3×3, 5×5 and 7×7 kernels
3. Study the effect of stride and padding on output dimensions
4. Visualize feature maps after the first convolution layer
5. Compare max pooling and average pooling
6. Build and train the CNN
7. Evaluate accuracy, precision, recall, F1-score and confusion matrix

## Results

| Metric | Value |
|---|---|
| Training Accuracy | 0.7698 |
| Testing Accuracy | 0.6784 |
| Precision | 0.6867 |
| Recall | 0.6784 |
| F1-score | 0.6788 |
| Number of Parameters | 25,578 |

## Files

```
DeepLearning_Ex3.ipynb    Notebook with all seven tasks
Plots/                    Generated plots (PNG)
```

## How to Run

Open `DeepLearning_Ex3.ipynb` in Google Colab, set the runtime to GPU, and run all cells. Plots are saved to the `plots/` folder.