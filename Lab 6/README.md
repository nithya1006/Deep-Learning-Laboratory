# Lab 6 — RNN, LSTM and GRU for Sequence Learning and Video Understanding

End-to-end study of recurrent sequence learning: Vanilla RNN, LSTM and GRU are
implemented and compared on sensor-based activity recognition, extended to video
understanding with CNN feature extraction, and finished with an encoder–decoder
sequence-to-sequence task.

## Objective

- Represent sequential data in the `(N, T, F)` format and train SimpleRNN, LSTM
  and GRU classifiers under one controlled protocol.
- Compare the three models on predictive performance, parameter count and
  training cost, and study the effect of sequence length.
- Build a CNN–LSTM / CNN–GRU pipeline for action recognition in video.
- Implement an encoder–decoder model for a synthetic sequence reversal task.

## Datasets

| Dataset | Use | Source |
|---|---|---|
| UCI Human Activity Recognition Using Smartphones | Primary sequence classification (6 activities) | [archive.ics.uci.edu](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) |
| UCF101 | Video action recognition (4 classes) | [crcv.ucf.edu](https://www.crcv.ucf.edu/data/UCF101.php) |
| Synthetic reversal sequences | Sequence-to-sequence learning | Generated in the notebook |

## Models

| Model | Recurrent layer | Units | Classes |
|---|---|---|---|
| RNN | SimpleRNN | 32 | 6 |
| LSTM | LSTM | 32 | 6 |
| GRU | GRU | 32 | 6 |
| CNN–LSTM / CNN–GRU | LSTM / GRU over frozen MobileNetV2 features | 32 | 4 |
| Seq2Seq | Encoder LSTM → context state → decoder LSTM | 64 | — |

All three recurrent models share the same input, classifier head, optimizer
(Adam, 1e-3), batch size (32), epoch count (30) and evaluation protocol, so the
comparison is controlled — only the recurrent layer changes.

## Results

| Model | Accuracy | Precision | Recall | F1 | Parameters |
|---|---|---|---|---|---|
| RNN | XX.XX% | XX.XX% | XX.XX% | XX.XX% | X,XXX |
| LSTM | XX.XX% | XX.XX% | XX.XX% | XX.XX% | X,XXX |
| GRU | XX.XX% | XX.XX% | XX.XX% | XX.XX% | X,XXX |
| CNN–LSTM | XX.XX% | XX.XX% | XX.XX% | XX.XX% | XXX,XXX |
| CNN–GRU | XX.XX% | XX.XX% | XX.XX% | XX.XX% | XXX,XXX |

Sequence-to-sequence (reversal, 6 → 6): token accuracy XX.XX%, sequence accuracy XX.XX%.

Values come from an actual execution of the notebook and will vary slightly
between runs. The video test set is small, so video accuracies in particular are
coarse — a single clip moves the number by several points.

## Plots

| Plot | Content |
|---|---|
| Plot 1 | Sensor signal versus time, one window per activity |
| Plot 2 | Training and validation loss (RNN, LSTM, GRU) |
| Plot 3 | Training and validation accuracy (RNN, LSTM, GRU) |
| Plot 4 | Confusion matrices (RNN, LSTM, GRU) |
| Plot 5 | Model performance comparison |
| Plot 6 | Sequence length versus test F1-score |
| Plot 7 | Ten sampled frames from one video |
| Plot 8 | Video training and validation curves |
| Plot 9 | Video confusion matrices |

## Additional Exercises

1. 16, 32 and 64 recurrent units compared on accuracy, F1, parameters and time
2. GRU versus LSTM at matched unit counts
3. A second recurrent layer
4. Bidirectional versus unidirectional LSTM
5. Sequence length versus computational cost
6. CNN–LSTM versus CNN–GRU on identical CNN features
7. Seq2Seq with an output length different from the input length


## Dependencies

Python · NumPy · pandas · Matplotlib · scikit-learn · OpenCV · TensorFlow/Keras

All are pre-installed in Colab. The notebook installs `unrar` for the UCF101
archive.