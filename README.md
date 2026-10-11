# Generative AI with LSTM — Text Generation

Interview task deliverable: word-level LSTM text generator trained on Shakespeare's works.

## Dataset

[shakespeare.txt](https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt) — downloaded automatically inside the notebook via `keras.utils.get_file`, no manual step required. (Equivalent to concatenating Shakespeare's plays from [Project Gutenberg](https://www.gutenberg.org).)

## Contents

- `lstm_text_generation.ipynb` — full pipeline: preprocessing, model design, training, text generation, bonus architecture experiments.
- `requirements.txt` — pinned dependencies.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook lstm_text_generation.ipynb
```

## Approach

1. **Preprocessing**: lowercase, strip punctuation, word-level tokenize, build an 8k-word vocabulary (`<pad>`/`<oov>` reserved), build sliding-window `(20-token input -> next-token)` pairs, chronological train/val split (90/10) to avoid leaking near-duplicate overlapping windows across the split.
2. **Model**: `Embedding -> [LSTM -> Dropout] x N -> Dense(softmax)`, trained with sparse categorical crossentropy + Adam, early stopping and checkpointing on validation loss.
3. **Generation**: iterative next-token prediction from a seed phrase, with temperature-based sampling (vs. greedy argmax) to avoid repetitive loops.
4. **Bonus**: compares 1/2/3 stacked LSTM layers and sequence lengths of 10/20/40 tokens on validation loss/accuracy and sample quality which you can find in lstm_text_generation.ipynb at last.
