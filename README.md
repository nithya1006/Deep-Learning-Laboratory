# Deep Learning Laboratory

Lab experiments for the Deep Learning Laboratory course. Each lab is a
self-contained folder with its own notebook, dataset details, dependency list
and execution instructions in its README.

All notebooks are written for **Google Colab**.

## Labs

| Lab | Topic | Dataset | Result |
|---|---|---|---|
| [Lab 1](Lab%201/) | Perceptron implemented from scratch in NumPy | UCI Banknote Authentication | 98.55% accuracy |
| [Lab 2](Lab%202/) | Feedforward neural network with hyperparameter tuning | Fashion-MNIST | 89.30% accuracy (optimized) |
| [Lab 3](Lab%203/) | Convolutional neural network for image classification | CIFAR-10 | 67.84% test accuracy |
| [Lab 4](Lab%204/) | Transfer learning and comparison of CNN architectures | CIFAR-10 | 70.52% test accuracy (VGG16, fine tuned) |

## Structure

```
Deep Learning Laboratory/
├── Lab 1/
│   ├── README.md
│   ├── DeepLearning_Ex1.ipynb
│   ├── data_banknote_authentication.txt
│   └── Plots/
├── Lab 2/
│   ├── README.md
│   ├── DeepLearning_Ex2.ipynb
│   └── Plots/
├── Lab 3/
│   ├── README.md
│   ├── DeepLearning_Ex3.ipynb
│   └── Plots/
└── Lab 4/
    ├── README.md
    ├── DeepLearning_Ex4.ipynb
    └── Plots/
```
## Getting Started

Open the folder for the lab you want, read its README, then upload the notebook
to [Google Colab](https://colab.research.google.com) and run all cells.

## Tools

Python · NumPy · pandas · Matplotlib · seaborn · scikit-learn · TensorFlow/Keras