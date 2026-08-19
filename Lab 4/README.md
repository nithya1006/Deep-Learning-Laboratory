# Experiment 4 — Comparative Study of Deep CNN Architectures Using Transfer Learning

## Objective

To study the evolution of deep CNN architectures, implement transfer learning and fine tuning on a pretrained model, and compare the classification performance of LeNet-5, AlexNet, VGG16, GoogleNet and ResNet50 using TensorFlow/Keras.

## Dataset

CIFAR-10

- Training Images: 45,000 (after a 10% validation split)
- Validation Images: 5,000
- Testing Images: 10,000
- Classes: 10
- Image Size: 32 × 32 × 3

## Model

```
Input (32×32×3) → VGG16 convolutional base (frozen) → GlobalAveragePooling2D → Dense(128, ReLU) → Dense(10, Softmax)
```

- Pretrained weights: ImageNet
- Optimizer: Adam
- Learning Rate: 0.001 (frozen phase), 1e-5 (fine tuning)
- Batch Size: 32
- Epochs: 15 frozen + 8 fine tuning
- Loss: Categorical Cross Entropy

## Tasks

1. Load CIFAR-10, normalize to [0,1], display ten sample images and print dataset dimensions
2. Build the VGG16 transfer learning model with a frozen base and a new classification head
3. Train the frozen model with Adam
4. Fine tune by unfreezing the last convolution block (block5) and compare accuracy before and after
5. Evaluate using accuracy, precision, recall, F1-score, confusion matrix and classification report
6. Hyperparameter study across learning rate, batch size, epochs, optimizer, dense units and frozen layers
7. Compare LeNet-5, AlexNet, VGG16, GoogleNet and ResNet50

## Results

### Performance Metrics — VGG16 after fine tuning

| Metric | Value |
|---|---|
| Training Accuracy | 0.9330 |
| Testing Accuracy | 0.7052 |
| Precision | 0.7058 |
| Recall | 0.7052 |
| F1-score | 0.7044 |
| Training Time | 645.98 s |
| Total Parameters | 14,781,642 |

Fine tuning improved validation accuracy from 0.6186 to 0.7118 (+0.0932).

### Comparison of CNN Architectures

| Model | Parameters | Accuracy (%) | Training Time |
|---|---|---|---|
| LeNet-5 | 83,126 | 53.29 | 506.58 s |
| AlexNet | 8,578,314 | 67.05 | 677.86 s |
| VGG16 | 14,781,642 | 70.52 | 645.98 s |
| GoogleNet | 22,066,346 | 59.73 | 741.47 s |
| ResNet50 | 23,851,274 | 38.32 | 532.30 s |

VGG16 with fine tuning performed best. ResNet50 performed worst because its aggressive early downsampling reduces a 32×32 input to a 1×1 feature map before pooling. The VGG16 time covers all 23 epochs including fine tuning; the other models were trained for 10 epochs.

### Hyperparameter Study

One factor changed per run, trained on a 10,000 image subset. Baseline: lr 0.001, batch 32, 10 epochs, Adam, 128 dense units, all layers frozen.

| Configuration | Value | Validation Accuracy |
|---|---|---|
| Baseline | — | 0.5640 |
| Learning Rate | 0.0001 | 0.5440 |
| Batch Size | 16 | 0.5648 |
| Batch Size | 64 | 0.5744 |
| Epochs | 20 | 0.5658 |
| Optimizer | SGD | 0.5130 |
| Dense Units | 256 | 0.5778 |
| Frozen Layers | Partial | 0.6602 |

Unfreezing the last convolution block had by far the largest effect. Every other setting changed accuracy by less than 0.02.

## Files

```
DeepLearning_Ex4.ipynb    Notebook with all tasks
plots/                    Generated plots (PNG, 600 dpi)
```

## How to Run

Open `DeepLearning_Ex4.ipynb` in Google Colab with the runtime set to GPU, or run it locally with a GPU-enabled TensorFlow install, and run all cells top to bottom. Pretrained ImageNet weights are downloaded automatically on the first run. Plots are saved to the `plots/` folder.

Requirements: `tensorflow`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`