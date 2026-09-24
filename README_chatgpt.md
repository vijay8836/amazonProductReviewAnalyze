# Amazon Review ML / GenAI Analysis Project

## Files

- `amazon_review_analyzer.py` — complete Python baseline pipeline.
- `amazon_reviews_sample.csv` — small sample dataset for testing.
- `requirements.txt` — Python dependencies.
- `README.md` — setup and usage instructions.

## Goal

Estimate a trust-weighted "genuine product rating" from Amazon reviews while identifying:

- likely genuine positive reviews
- likely genuine negative reviews
- rating/text contradictions
- possible duplicate reviews
- generic reviews
- irrelevant reviews
- wrong-product reviews
- shipping/delivery issues
- seller/customer-service issues
- packaging issues
- possible anomalous/suspicious reviews

## Important limitation

The system does **not** prove that a reviewer is a competitor, paid reviewer, bot, or fraudster. It produces signals and a trust score. Strong claims about reviewer identity require additional evidence and properly labeled data.

## Setup

Use Python 3.10+.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install:

```bash
pip install -r requirements.txt
```

The first run downloads the SentenceTransformer model:

`all-MiniLM-L6-v2`

## Input

Create `amazon_reviews.csv` with at least:

```text
rating
review_text
```

Recommended additional columns:

```text
review_id
reviewer_id
review_title
verified_purchase
helpful_votes
review_date
```

## Run

```bash
python amazon_review_analyzer.py
```

The program creates:

- `review_analysis.csv`
- `review_category_report.csv`

## How the genuine rating is calculated

Each review receives a trust score between 0 and 1.

Signals that can reduce trust include:

- rating/text contradiction
- near-duplicate reviews
- wrong-product content
- low product relevance
- generic reviews
- anomaly detection

Positive signals include:

- verified purchase
- helpful votes

The final rating is a trust-weighted average:

`sum(rating * trust_score) / sum(trust_score)`

The threshold is configured as 4.0.

## Recommended production upgrade

For a production-quality system, replace the simple rule-based components with:

1. Transformer sentiment model
2. Sentence-BERT embeddings
3. LLM structured-output classifier
4. XGBoost/LightGBM supervised fake-review model
5. Reviewer behavioral features
6. Temporal burst detection
7. Graph-based coordinated-review detection
8. Calibrated probabilities
9. Confidence intervals
10. Human review for high-impact decisions

Do not use the output as proof that a particular person is a competitor or fraudulent reviewer.
