# Lab 1 — Perceptron from Scratch

Implementation of a single-layer perceptron in NumPy to classify banknotes as
authentic or forged.

## Files

| File | Description |
|---|---|
| `DeepLearning_Ex1.ipynb` | Source code — all tasks in one notebook |
| `data_banknote_authentication.txt` | Dataset |
| `Plots/` | Output figures (11 PNGs) |

## Dataset

**UCI Banknote Authentication** — features extracted from wavelet transforms of
banknote images.

- Source: https://archive.ics.uci.edu/dataset/267/banknote+authentication
- Format: CSV, no header row
- Samples: 1372, no missing values
- Features (4): Variance, Skewness, Curtosis, Entropy
- Target: Class — 0 = Authentic (762), 1 = Forged (610)

## Dependencies

```
numpy
pandas
matplotlib
seaborn
scikit-learn
```

All of these come pre-installed in Google Colab, so no installation is needed.

## How to Run (Google Colab)

1. Open [Google Colab](https://colab.research.google.com) and upload
   `DeepLearning_Ex1.ipynb` (**File → Upload notebook**).
2. Upload the dataset to the session using the file panel on the left, or run:
   ```python
   from google.colab import files
   files.upload()   # select data_banknote_authentication.txt
   ```
3. Run all cells with **Runtime → Run all**.

Plots are saved to a `plots/` folder inside the Colab session. To download them:

```python
!zip -r plots.zip plots
from google.colab import files
files.download("plots.zip")
```

Seeds are fixed (`random_state=42`), so results are reproducible.

## What the Notebook Does

1. Dataset exploration — shape, missing values, statistics, class balance
2. EDA — histograms, correlation heatmap, scatter plot, boxplots
3. Preprocessing — StandardScaler, 80/20 stratified split
4. Perceptron implemented from scratch — step activation, weight/bias updates
5. Training — learning rate 0.05, 20 epochs
6. Evaluation — accuracy, precision, recall, F1, confusion matrix
7. Plots — error, weight and bias evolution
8. Extra — comparison with scikit-learn, learning rate sweep, XOR test,
   normalization effect, decision boundary

## Results

| Metric | Value |
|---|---|
| Accuracy | 0.9855 |
| Precision | 0.9683 |
| Recall | 1.0000 |
| F1-score | 0.9839 |

The perceptron fails on XOR, confirming a single layer cannot solve
non-linearly-separable problems.
