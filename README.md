# E-Commerce Review Sentiment Classifier (Discovery-to-Action)

A binary sentiment classifier for e-commerce reviews built with TensorFlow/Keras `TextVectorization` and an embedding-based neural network, structured around the **Discovery → Technical → Action (DTA)** framework.

## How to run

1. Open `ecommerce_sentiment_nn.ipynb` in Google Colab (File → Upload notebook, or drag into colab.research.google.com).
2. Runtime → Run all. No GPU needed; no external file uploads required — the notebook generates its own dataset by default (see below).
3. Total runtime: ~2-4 minutes.

To use your own data instead of the built-in synthetic dataset, use the "PATH B" cell in Section 1.1 to upload a CSV with `review_text` and `rating` columns.

## Dataset preparation (Discovery phase)

- **Source:** synthetic e-commerce review generator (1,500 rows: 500 positive-template, 500 negative-template, 500 neutral-template reviews across 18 product types), included for zero-setup reproducibility in Colab. A "PATH B" cell is provided to swap in a real CSV (e.g. Amazon or clothing-store review exports) with the same `review_text` / `rating` schema.
- **Missing values:** rows with null, empty, or whitespace-only `review_text` are dropped (no imputation, since an empty review has no usable text signal).
- **Labeling:** ratings 4-5 → label `1` (Positive); ratings 1-2 → label `0` (Negative); 3-star (neutral) reviews are **dropped** to sharpen the binary decision boundary.
- **Split:** 80% train / 10% validation / 10% test, stratified by label.

## Model architecture (Technical phase)

```
TextVectorization (max_tokens=10,000, output_sequence_length=40, lowercase + strip punctuation)
  -> Embedding(input_dim=10,000, output_dim=16)
  -> GlobalAveragePooling1D
  -> Dense(16, activation='relu')
  -> Dense(1, activation='sigmoid')
```

- Loss: `binary_crossentropy` | Optimizer: `adam` | Metric: `binary_accuracy`
- Trained up to 10 epochs with `EarlyStopping` (monitor=`val_binary_accuracy`, patience=3, restore best weights) to guard against overfitting.
- `TextVectorization` is `.adapt()`-ed only on training text to avoid vocabulary leakage from validation/test data.

## Results

| Metric | Value |
|---|---|
| Final validation binary accuracy | 1.0000 (epoch 7, early stopping triggered) |
| Held-out test binary accuracy | 1.0000 |
| Held-out test loss | 0.6265 |
| Confusion matrix (TN, FP, FN, TP) | <<< paste the 2x2 matrix from notebook cell [14] here, e.g. [[48, 2], [0, 50]] >>> |
| Precision / Recall / F1 (per class) | <<< paste the classification_report numbers from cell [14] here >>> |
| Required test sentence ("The product arrived broken and I am very unhappy") | Predicted probability <<< paste the exact number from cell [15], e.g. 0.0842 >>> → classified Negative ✅ |
| Hard-example (out-of-distribution) accuracy | <<< paste the "Hard-example accuracy: 0.XX" line from the Section 3.4 cell >>> |

**Reading the 100% test accuracy correctly:** this is a sign the in-distribution test set is too easy, not that the model has learned generalizable sentiment understanding. The synthetic generator draws each class from a small, disjoint set of templates, so the model can hit 100% by memorizing keyword-to-class associations (e.g. "broken," "terrible" → negative) rather than learning robust sentiment patterns. The test loss of 0.6265 alongside 100% accuracy is a second tell: predictions land on the correct side of 0.5 but aren't confidently near 0 or 1, meaning the model is right but not sure. Section 3.4 of the notebook ("Generalization check") tests the model on hand-written sentences with vocabulary it never saw in training — that hard-example accuracy, not the 100% above, is the honest measure of how close this model is to production-ready.

## Confidence score interpretation

The sigmoid output is a probability-shaped score, not a guaranteed-calibrated probability of correctness. It reflects distance from the model's learned decision boundary, which is sensitive to: out-of-vocabulary words collapsing to a single `[UNK]` token, and the pooling layer discarding word order (so negation patterns like "not bad" can be misread).

## Auto-flagging workflow (Action phase)

| Predicted P(positive) | Route | Rationale |
|---|---|---|
| `< 0.20` | **Auto-escalate** → priority support queue, auto-ticket created | High-confidence negative; safe to automate without human bottleneck |
| `0.20 - 0.50` | **Human review** → secondary triage queue (e.g. 24h SLA) | Likely negative but not confident enough to auto-escalate; human judgment catches the recall the strict threshold gives up |
| `>= 0.50` | **No action** → logged for analytics only | Predicted positive |

**Why 0.20 instead of the standard 0.5 boundary for auto-escalation:** 0.5 is the point of *maximum* model uncertainty, not minimum — using it as the automation trigger would auto-escalate a lot of borderline cases the model isn't actually confident about, flooding the priority queue with false alarms. Lowering the auto-escalation threshold to 0.20 trades a bit of recall (some real negatives land in the human-review band instead of auto-escalating) for much higher precision on the automated action, while the human-review band acts as a safety net rather than discarding those borderline cases entirely.

## Limitations & next steps

- **Sarcasm/irony** not reliably detected (lexically positive phrasing, semantically negative intent).
- **Context length**: reviews are truncated/padded to 40 tokens; longer multi-sentence reviews may lose information.
- **Vocabulary gaps**: brand names, typos, and slang outside the 10k-token vocabulary collapse to `[UNK]`.
- **No word order modeling**: `GlobalAveragePooling1D` ignores sequence, weakening negation/intensifier handling (an LSTM/GRU or small Transformer would likely help).
- **Domain shift**: the default dataset is synthetic/templated; before production use, retrain and validate on real labeled reviews from the target store.
- **Data drift over time**: customer language evolves (new slang, seasonal phrasing, deliberate misspellings); since the vocabulary is frozen at training time, words outside it collapse to `[UNK]` regardless of sentiment content, so performance can quietly degrade as live language drifts from the training distribution. Track OOV rate and routing-band proportions over time, and retrain periodically on freshly labeled reviews rather than treating the model as train-once.
- **Calibration**: sigmoid scores should be calibrated (e.g. temperature scaling) against real labeled data before fully trusting the 0.20/0.50 thresholds in production.
- **Monitoring**: track queue-volume breakdown and sampled human-reviewed outcomes over time to catch model drift.

## Repository contents

- `ecommerce_sentiment_nn.ipynb` — full notebook (Discovery → Technical → Action), runs end-to-end in Colab
- `README.md` — this file
