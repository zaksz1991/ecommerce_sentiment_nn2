# Neural Network Sentiment Analysis
### E-Commerce Review Classification — Discovery-to-Action (DTA) Strategy

A deep learning sentiment classifier for e-commerce product reviews, built with
Keras `TextVectorization` and an embedding-based neural network, with predictions
translated into an automated customer-support routing workflow.

---

## Project Structure

```
.
├── ecommerce_sentiment_nn.ipynb   # Main notebook (run this) — fully executed
├── generate_reviews.py            # Standalone version of the data generator
└── README.md
```

## How to Run

Open `ecommerce_sentiment_nn.ipynb` in **Google Colab**, **Jupyter**, or any
environment with `tensorflow>=2.10`, `pandas`, `scikit-learn`, and `matplotlib`
installed, and `Run all`. Total runtime is well under a minute — this model is
small and needs no GPU or external downloads. The notebook generates its own
dataset in Section 1.1, so no setup, credentials, or dataset download is required.

---

## 1. Dataset Preparation

This project uses a **synthetically generated** dataset of 1,500 e-commerce
reviews rather than a downloaded one, so the notebook is fully reproducible for
any reviewer with zero broken links or external dependencies, while still
exercising every part of a real NLP pipeline.

**What the data looks like:** each row has a `review_id`, free-text
`review_text`, and a 1–5 `rating`. Review text is built by combining one of 28
product names with sentiment-appropriate template sentences and supporting
detail phrases — producing varied, non-repetitive text rather than literally
duplicated strings. Ratings follow a realistic, polarized distribution (more
5-star and 1-star reviews than 3-star), and 2% of rows have deliberately blank
`review_text` to simulate real-world messy data.

### Cleaning & Labeling

- **Missing values**: both true `NaN` and whitespace-only/empty strings in
  `review_text` are treated as missing and dropped — there's no usable signal
  in an empty review, and imputing fake text would inject noise.
- **Binary labels**: 4–5 stars → `1` (Positive), 1–2 stars → `0` (Negative).
- **Neutral reviews dropped**: 3-star reviews are removed entirely before
  training. Neutral reviews genuinely sit between the two classes; keeping them
  in a binary task forces an arbitrary label onto text that often isn't strongly
  polarized, which muddies the decision boundary. Dropping them sharpens the
  task to "clearly positive vs. clearly negative" — also the more useful signal
  for a support-routing system, which cares most about confidently identifying
  unhappy customers.
- **Final split**: 80/20 train/test, stratified by label to preserve class
  balance in both splits.

### Text Standardization & Vectorization

- **Lowercasing + punctuation stripping** via Keras `TextVectorization`'s default
  `standardize="lower_and_strip_punctuation"`.
- **Vocabulary capped at 10,000 tokens** — generous headroom for review text
  while keeping the embedding table small enough to train quickly without
  overfitting on rare words.
- **Fixed sequence length of 100 tokens** — sized well above the observed review
  length distribution (reviews top out around 20 words in this dataset), so no
  meaningful truncation occurs.
- The vectorizer is `adapt()`-ed on **training text only**, avoiding any
  leakage of test-set vocabulary into preprocessing.

### Using a Real Dataset Instead

Provide any CSV with `review_text` and `rating` columns and skip the
`generate_reviews_csv()` call in Section 1.1 — every other cell works unmodified.

---

## 2. Model Architecture

```
TextVectorization → Embedding(10000, 16) → GlobalAveragePooling1D
                   → Dense(16, ReLU) → Dense(1, Sigmoid)
```

- **Embedding(16-dim)**: maps each token to a dense learned vector, allowing the
  model to discover that semantically similar words (e.g. "terrible" and
  "awful") end up with similar representations purely from training data
  co-occurrence — no hand-built synonym list needed.
- **GlobalAveragePooling1D**: averages token embeddings across the whole review
  into one fixed-size vector regardless of review length — the text equivalent
  of `GlobalAveragePooling2D` in image models, converting a variable-length
  sequence into a single "what did this review's words look like on average"
  summary.
- **Dense(16, ReLU) → Dense(1, Sigmoid)**: a small hidden layer followed by a
  sigmoid output producing a positive-sentiment probability in [0, 1].

**Compiled with** binary cross-entropy loss and the Adam optimizer
(`lr=5e-3`), **trained for 10 epochs**, monitoring `val_binary_accuracy`
specifically to detect overfitting (a model improving on training data while
validation accuracy stalls would indicate memorization rather than
generalization).

---

## 3. Final Model Performance

| Metric | Negative | Positive |
|---|---|---|
| Precision | 1.00 | 1.00 |
| Recall | 1.00 | 1.00 |
| F1-score | 1.00 | 1.00 |

**Overall test accuracy: 100%** (260 held-out test reviews).

**Important context:** this near-perfect score reflects the synthetic,
template-based nature of the dataset (a small, consistent ~200-word vocabulary
with highly predictive polarized words), not a claim about real-world
performance — see Limitations below. Training and validation loss/accuracy
curves track closely together throughout training with no divergence,
indicating the model is fitting the patterns it was given without overfitting
to this dataset specifically.

### Required Test Case

> *"The product arrived broken and I am very unhappy"*

**Predicted probability: 0.0183** — correctly and confidently classified as
negative, well below the 0.2 threshold discussed below.

---

## 4. Threshold Recommendation & DTA Workflow

Predicted probabilities are converted into a three-band automated routing
policy for customer support:

| Positive probability | Action | What happens |
|---|---|---|
| **< 0.2** | `AUTO_FLAG_URGENT` | Immediate priority routing to support queue, proactive outreach triggered |
| **0.2 – 0.5** | `FLAG_FOR_REVIEW` | Added to standard (non-urgent) support queue for human review |
| **≥ 0.5** | `NO_ACTION` | Continues through normal display/aggregation pipeline |

### Why < 0.2, not the 0.5 classification boundary?

0.5 is the boundary between the *classes* — it is not necessarily the right
boundary for *automation*. A review scoring 0.45 is technically "negative" but
the model isn't very confident about it. Auto-escalating every sub-0.5 review
with the same urgency as one scoring 0.02 would flood the support queue with
low-confidence cases and erode trust in the automated system over time
("crying wolf"). A 0.2 cutoff favors **precision on the urgent-escalation
action specifically**: only reviews the model is genuinely confident about
get the immediate-attention treatment, trading a small amount of recall for
much higher reliability on the highest-priority queue. Reviews in the
0.2–0.5 band aren't ignored — they're downgraded to the standard queue rather
than dropped, which balances automation speed (clear cases move instantly,
with zero human bottleneck) against review accuracy (ambiguous cases still
reach a human, just without triggering an alarm).

In production, this threshold should be tuned against real precision/recall
data and the support team's actual queue capacity — 0.2 is a reasonable,
justified starting point here, not a value the model determined on its own.

---

## 5. Limitations & Next Steps

- **Synthetic, template-based data** makes this task easier than real review
  classification; treat the reported metrics as pipeline validation, not a
  production performance estimate.
- **Observed bias toward scoring mild/neutral language as negative.** Because
  3-star reviews were intentionally excluded from training, the model never
  saw words like "okay" or "fine" attached to a positive label — we observed
  this directly while building the notebook (some genuinely mild-but-positive
  phrasing scored lower than expected). A production system should consider
  keeping a small neutral-labeled sample during training, or treat mid-range
  scores with extra caution for this reason specifically.
- **No sarcasm/irony modeling** — a review using positive words sarcastically
  ("Oh great, ANOTHER broken product") would likely be misclassified, since the
  architecture has no way to detect tone inversion.
- **No sequential/word-order modeling** — `GlobalAveragePooling1D` discards word
  order entirely, so negation patterns ("not good" vs "good") are weakly
  captured at best.
- **Vocabulary gaps** — words outside the 10,000-token vocabulary collapse to a
  single `[UNK]` token, losing whatever sentiment signal they carried.

**Next steps for production**: train on real labeled review data at scale;
calibrate thresholds against real precision-recall curves and support team
capacity; consider a sequence-aware architecture (LSTM/GRU/Transformer) for
better negation and word-order handling; add an explicit "low-confidence"
category rather than relying solely on threshold bands; monitor the live
confidence-score distribution for drift; and A/B test the auto-flagging
workflow against the existing manual process before full rollout.

---

## Author

Built as part of a Darey.io data science / ML curriculum project.
GitHub: [zaksz1991](https://github.com/zaksz1991)
