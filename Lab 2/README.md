# Lab 2 — Feedforward Neural Network on Fashion-MNIST

A fully connected neural network built in Keras for image classification,
compared against a hyperparameter-optimized version.

## Files

| File | Description |
|---|---|
| `DeepLearning_Ex2.ipynb` | Source code — all tasks in one notebook |
| `Plots/` | Output figures (8 PNGs) |

## Dataset

**Fashion-MNIST** — grayscale images of clothing items.

- Source: loaded directly in code via `tensorflow.keras.datasets.fashion_mnist`
  (downloads automatically, no manual file needed)
- Train: 60,000 images · Test: 10,000 images
- Image size: 28 x 28 grayscale, flattened to 784 features
- Classes (10): T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt,
  Sneaker, Bag, Ankle Boot — 6,000 training images each

## Dependencies

```
tensorflow
scikeras
numpy
pandas
matplotlib
seaborn
scikit-learn
```

All are pre-installed in Google Colab except `scikeras`, so run this in the
first cell:

```python
!pip install scikeras
```

## How to Run (Google Colab)

1. Open [Google Colab](https://colab.research.google.com) and upload
   `DeepLearning_Ex2.ipynb` (**File → Upload notebook**).
2. Install `scikeras` as shown above (restart the runtime if Colab asks).
3. Run all cells with **Runtime → Run all**. The dataset downloads
   automatically — nothing to upload.

Plots are saved to a `plots/` folder inside the Colab session. To download them:

```python
!zip -r plots.zip plots
from google.colab import files
files.download("plots.zip")
```

**Notes:**
- Set **Runtime → Change runtime type → T4 GPU** to speed up training.
- The hyperparameter search is the slow step — 50 model fits, roughly
  20–30 minutes on CPU. Lower `n_iter`, `cv` or `SEARCH_SIZE` for a quicker run.
- Colab disconnects idle sessions, so keep the tab active during the search.

## What the Notebook Does

1. Dataset exploration — shapes, sample images, class distribution
2. Preprocessing — flatten to 784, normalize pixels to 0–1, one-hot encode labels
3. Baseline model — `784 → 128 (relu) → 64 (relu) → 10 (softmax)`, Adam,
   20 epochs, batch size 32
4. Training and evaluation — accuracy, precision, recall, F1, confusion matrix
5. Hyperparameter optimization — RandomizedSearchCV over layers, neurons,
   learning rate, batch size, epochs, optimizer, activation and dropout
6. Optimized model retrained with the best parameters
7. Comparison — metrics, confusion matrices and training time side by side

## Results

Best hyperparameters: `adam`, 256 neurons, learning rate 0.001, 2 hidden layers,
dropout 0.0, `sigmoid` activation, 30 epochs, batch size 32.

| Metric | Baseline | Optimized |
|---|---|---|
| Accuracy | 0.8831 | 0.8930 |
| Precision | 0.8835 | 0.8927 |
| Recall | 0.8831 | 0.8930 |
| F1-score | 0.8818 | 0.8924 |
| Training time (s) | 37.56 | 106.02 |
