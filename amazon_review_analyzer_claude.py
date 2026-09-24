"""
=============================================================================
Amazon Cosmetic Product Review Analyzer
=============================================================================
PURPOSE:
    Analyzes Amazon product reviews to detect different reviewer groups
    (genuine, fake/promotional, competitor sabotage, wrong-product raters,
    and any other anomalous categories), then computes a "genuine rating"
    using a combination of ML classification and LLM-powered review analysis.

APPROACH OVERVIEW:
    1. Data Simulation   – Simulate ~1100 reviews matching the problem spec
                           (avg 3.93 ★, 95% rated+reviewed, 4+ known groups)
    2. Feature Engineering – Extract text & meta features per review
    3. ML Classification – Use a trained classifier (Random Forest / LR) to
                           assign each review to a behavioural cluster
    4. LLM Analysis      – Use Claude (Anthropic API) to label a sample of
                           reviews and define category names semantically
    5. Rating Correction – Compute genuine weighted average after removing
                           non-genuine reviews
    6. Decision Output   – Print whether the product is "Good" (≥ 4.0)

DEPENDENCIES (install once):
    pip install anthropic scikit-learn pandas numpy nltk

ANTHROPIC API KEY:
    Set the environment variable:  ANTHROPIC_API_KEY=sk-ant-...
    Or replace the placeholder string in the code below.
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0.  IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os
import re
import json
import random
import textwrap
import warnings
from collections import Counter

import numpy as np
import pandas as pd

# NLP / ML
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Anthropic (Claude)
import anthropic

warnings.filterwarnings("ignore")

# Download NLTK resources quietly
for resource in ["vader_lexicon", "stopwords", "punkt"]:
    nltk.download(resource, quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

STOP_WORDS = set(stopwords.words("english"))


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA SIMULATION
#     We simulate 1,100 reviews across 5 groups (the 4 specified + 1 extra)
#     so the code runs end-to-end without needing real Amazon credentials.
#     In production, replace simulate_reviews() with your real scraper output.
# ─────────────────────────────────────────────────────────────────────────────

GENUINE_POS_TEMPLATES = [
    "Absolutely love this moisturizer! My skin feels soft and hydrated all day.",
    "Great serum, noticed visible improvement in just two weeks of use.",
    "The texture is lightweight and absorbs quickly. Very happy with the results.",
    "Best foundation I have tried. Gives natural coverage and lasts all day.",
    "This sunscreen doesn't leave a white cast. Perfect for my skin tone.",
    "Amazing product! My dark spots have reduced significantly.",
    "The smell is pleasant and the formula is gentle. Highly recommend.",
    "Works as advertised. My skin tone has evened out since I started using it.",
]

GENUINE_NEG_TEMPLATES = [
    "Broke out badly after a week of use. Stopped immediately.",
    "The packaging is terrible — product leaked during shipping.",
    "Too thick and greasy for my combination skin. Not repurchasing.",
    "Disappointing. The colour was completely different from the pictures.",
    "Caused irritation and redness around my eyes. Returning this.",
    "Overpriced for what you get. The effects wore off within two hours.",
    "Smell is overpowering and gave me a headache.",
    "Did not notice any difference in my skin after a month of use.",
]

FAKE_PROMO_TEMPLATES = [
    "AMAZING PRODUCT! 5 stars!! Best thing I've ever bought! Buy it NOW!!!",
    "Incredible results overnight!! My skin is flawless thanks to this gem!",
    "World class product. I tell everyone to buy this. Zero complaints at all!!!",
    "Wow absolutely stunning results. Every person should have this in their routine.",
    "This product changed my life COMPLETELY. Cannot imagine living without it!!",
    "Perfect perfect perfect!! 10/10 would recommend to everyone on the planet!",
]

WRONG_RATING_TEMPLATES = [
    # Positive text but 1 star (inverted rating)
    ("Very happy with this product. My skin feels wonderful after using it.", 1),
    ("Great results in just a week! Love the texture and the smell.", 1),
    # Negative text but 5 stars (inverted rating)
    ("Absolutely terrible. Caused a rash. Would not recommend to anyone.", 5),
    ("Worst product I have ever used. Complete waste of money.", 5),
]

COMPETITOR_SABOTAGE_TEMPLATES = [
    "Do NOT buy from this brand. They stole my data. Completely unethical company.",
    "Garbage product. I bet the 5-star reviews are all paid. Scam alert!",
    "Reported this seller to Amazon. Their products contain harmful chemicals.",
    "Avoid at all costs. The company is fraudulent and products are counterfeit.",
    "This is a clone of [Brand X] and is clearly inferior. Do not waste money.",
    "I work in dermatology and this product is dangerous. Do not use.",
]

WRONG_PRODUCT_TEMPLATES = [
    # Review of a book mistakenly posted on a cosmetic product
    "Fascinating read. The author's prose is elegant and the plot keeps you hooked.",
    "Wonderful novel. The character development is superb. Finished in one sitting.",
    "A masterpiece of literature. I recommend this book to every reader.",
    "Good delivery, package arrived on time. The book was well wrapped.",
    "Charging cable stopped working after a week. Very poor build quality.",
    "My laptop battery drained fast. Not compatible with my device at all.",
]


def simulate_reviews(n_total: int = 1100, seed: int = 42) -> pd.DataFrame:
    """
    Simulate a realistic review dataset for a cosmetic product.

    Groups and their approximate share (must total 100%):
        genuine_positive   – 40%   genuinely liked the product
        genuine_negative   – 15%   genuinely disliked the product
        fake_promotional   – 20%   planted 5-star promotional reviews
        wrong_rating       –  5%   inverted star rating vs. review text
        competitor_sabotage– 12%   deliberate 1-star negative attacks
        wrong_product      –  8%   reviews for a completely different product

    Returns a DataFrame with columns:
        review_id, review_text, star_rating, verified_purchase,
        helpful_votes, review_length, true_group
    """
    random.seed(seed)
    np.random.seed(seed)

    group_config = [
        # (group_name, fraction, star_range, templates, fixed_star)
        ("genuine_positive",    0.40, (4, 5), GENUINE_POS_TEMPLATES,    None),
        ("genuine_negative",    0.15, (1, 3), GENUINE_NEG_TEMPLATES,    None),
        ("fake_promotional",    0.20, (5, 5), FAKE_PROMO_TEMPLATES,     5),
        ("wrong_rating",        0.05, None,   WRONG_RATING_TEMPLATES,   None),
        ("competitor_sabotage", 0.12, (1, 1), COMPETITOR_SABOTAGE_TEMPLATES, 1),
        ("wrong_product",       0.08, (1, 5), WRONG_PRODUCT_TEMPLATES,  None),
    ]

    rows = []
    review_id = 1

    for group, frac, star_range, templates, fixed_star in group_config:
        count = int(n_total * frac)

        for _ in range(count):
            # ── Pick template ──────────────────────────────────────────────
            if group == "wrong_rating":
                template = random.choice(templates)
                text, star = template  # tuple (text, inverted_star)
            else:
                template = random.choice(templates)
                text = template
                if fixed_star:
                    star = fixed_star
                elif star_range:
                    star = random.randint(*star_range)
                else:
                    star = random.randint(1, 5)

            # ── Add minor noise to text ────────────────────────────────────
            noise_words = ["!", ".", " Really.", " Truly.", " Honestly."]
            text += random.choice(noise_words)

            # ── Verified purchase (fake reviews less likely to be verified) ─
            if group in ("fake_promotional", "competitor_sabotage"):
                verified = random.random() < 0.15
            else:
                verified = random.random() < 0.75

            # ── Helpful votes (fake/sabotage get fewer) ───────────────────
            if group in ("genuine_positive", "genuine_negative"):
                helpful = np.random.poisson(lam=8)
            elif group == "wrong_rating":
                helpful = np.random.poisson(lam=2)
            else:
                helpful = np.random.poisson(lam=1)

            rows.append({
                "review_id":        review_id,
                "review_text":      text,
                "star_rating":      star,
                "verified_purchase": int(verified),
                "helpful_votes":    int(helpful),
                "review_length":    len(text.split()),
                "true_group":       group,      # ground-truth for evaluation
            })
            review_id += 1

    df = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    print(f"[DATA]  Simulated {len(df):,} reviews across {df['true_group'].nunique()} groups.")
    print(f"[DATA]  Group distribution:\n{df['true_group'].value_counts().to_string()}\n")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2.  FEATURE ENGINEERING
#     Extract numeric + text features that will feed the ML classifier.
# ─────────────────────────────────────────────────────────────────────────────

sia = SentimentIntensityAnalyzer()

# Cosmetic / product-specific lexicons
COSMETIC_KEYWORDS = {
    "skin", "moisturizer", "serum", "foundation", "sunscreen", "cream",
    "lotion", "face", "texture", "hydrat", "glow", "pore", "tone", "spf",
    "complexion", "pigment", "shade", "formula", "fragrance", "ingredient",
}

NON_COSMETIC_KEYWORDS = {
    "book", "read", "novel", "author", "plot", "character", "chapter",
    "laptop", "battery", "cable", "charging", "device", "software",
    "phone", "tablet", "keyboard",
}

FAKE_SIGNAL_WORDS = {
    "!!!",  "!!",  "amazing",  "incredible", "life-changing",
    "perfect", "flawless", "everyone", "immediately", "overnight",
}

SABOTAGE_SIGNAL_WORDS = {
    "scam", "fraud", "dangerous", "stolen", "counterfeit", "reported",
    "avoid", "unethical", "terrible", "garbage", "harmful",
}


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered feature columns to the dataframe.

    Features created:
        sentiment_pos / neg / neu / compound  – VADER scores
        exclamation_count   – number of '!' in text
        all_caps_ratio      – proportion of words in ALL CAPS
        cosmetic_keyword_hits – count of cosmetic terms found
        non_cosmetic_hits   – count of off-topic (wrong product) terms
        fake_signal_hits    – promotional language indicators
        sabotage_signal_hits– negative attack language indicators
        text_sentiment_star_diff – |VADER polarity vs. star rating| mismatch
        word_count          – number of words
    """
    records = []

    for _, row in df.iterrows():
        text  = row["review_text"].lower()
        words = word_tokenize(text)
        star  = row["star_rating"]

        vs = sia.polarity_scores(row["review_text"])

        # Capitalisation features
        raw_words = row["review_text"].split()
        caps_words = [w for w in raw_words if w.isupper() and len(w) > 1]

        # Keyword hits
        cos_hits  = sum(1 for kw in COSMETIC_KEYWORDS    if kw in text)
        nc_hits   = sum(1 for kw in NON_COSMETIC_KEYWORDS if kw in text)
        fake_hits = sum(1 for kw in FAKE_SIGNAL_WORDS     if kw in text)
        sab_hits  = sum(1 for kw in SABOTAGE_SIGNAL_WORDS if kw in text)

        # Mismatch: sentiment vs star rating
        # Normalise star to [0,1]; compound is already [-1,1] → remap to [0,1]
        norm_star      = (star - 1) / 4.0
        norm_sentiment = (vs["compound"] + 1) / 2.0
        mismatch       = abs(norm_sentiment - norm_star)

        records.append({
            "sentiment_pos":           vs["pos"],
            "sentiment_neg":           vs["neg"],
            "sentiment_neu":           vs["neu"],
            "sentiment_compound":      vs["compound"],
            "exclamation_count":       row["review_text"].count("!"),
            "all_caps_ratio":          len(caps_words) / max(len(raw_words), 1),
            "cosmetic_keyword_hits":   cos_hits,
            "non_cosmetic_hits":       nc_hits,
            "fake_signal_hits":        fake_hits,
            "sabotage_signal_hits":    sab_hits,
            "text_sentiment_star_diff":mismatch,
            "word_count":              len(raw_words),
            "verified_purchase":       row["verified_purchase"],
            "helpful_votes":           row["helpful_votes"],
            "star_rating":             star,
        })

    feat_df = pd.DataFrame(records)
    result  = pd.concat([df.reset_index(drop=True), feat_df], axis=1)
    print(f"[FEAT]  Extracted {len(feat_df.columns)} features per review.\n")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 3.  ML CLASSIFICATION (Random Forest)
#     Train a supervised classifier using the ground-truth labels from our
#     simulation (in production you'd use human-annotated labels or a
#     semi-supervised approach).
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "sentiment_pos", "sentiment_neg", "sentiment_neu", "sentiment_compound",
    "exclamation_count", "all_caps_ratio",
    "cosmetic_keyword_hits", "non_cosmetic_hits",
    "fake_signal_hits", "sabotage_signal_hits",
    "text_sentiment_star_diff", "word_count",
    "verified_purchase", "helpful_votes", "star_rating",
]


def train_and_predict(df: pd.DataFrame):
    """
    Train a Random Forest classifier on labelled features, evaluate it, then
    predict the group for every review.

    Returns:
        df  – original dataframe with a new 'predicted_group' column
        clf – the trained classifier
        le  – the LabelEncoder used for the target
    """
    le  = LabelEncoder()
    y   = le.fit_transform(df["true_group"])
    X   = df[FEATURE_COLS].fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # Evaluation on held-out test set
    y_pred = clf.predict(X_test)
    print("[ML]   Classification report (test set):")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Predict on full dataset
    X_all = scaler.transform(df[FEATURE_COLS].fillna(0))
    df["predicted_group"] = le.inverse_transform(clf.predict(X_all))

    return df, clf, le, scaler


# ─────────────────────────────────────────────────────────────────────────────
# 4.  LLM-POWERED CATEGORY ANALYSIS (Claude via Anthropic API)
#     Sample reviews from each ML-predicted group and ask Claude to:
#       a) Confirm / refine the category name in plain English
#       b) Write a one-line human-readable description
#       c) Decide whether the group should count toward the genuine rating
# ─────────────────────────────────────────────────────────────────────────────

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "YOUR_API_KEY_HERE")


def analyse_group_with_llm(
    group_name: str,
    sample_reviews: list[dict],
    client: anthropic.Anthropic,
) -> dict:
    """
    Send a sample of reviews from one group to Claude and get back a structured
    JSON analysis with keys:
        category_label    – human-friendly category name
        description       – 1-sentence description
        is_genuine        – bool: should these reviews count toward the rating?
        reasoning         – why Claude made this determination
    """
    sample_text = "\n".join(
        f"  [{i+1}] (★{r['star_rating']}) {r['review_text']}"
        for i, r in enumerate(sample_reviews[:6])  # max 6 samples
    )

    prompt = textwrap.dedent(f"""
        You are an e-commerce review quality analyst specialising in cosmetic products.

        Below are sample reviews that have been grouped together by a machine-learning
        classifier under the internal label: "{group_name}".

        Reviews:
        {sample_text}

        Analyse these reviews and respond ONLY with a valid JSON object (no markdown
        fences, no extra text) with exactly these keys:

        {{
          "category_label": "<short human-readable category name>",
          "description": "<one sentence explaining what makes these reviews distinctive>",
          "is_genuine": <true or false — should these count toward the product's real rating?>,
          "reasoning": "<2-3 sentences explaining your decision>"
        }}

        Rules:
        - A review is "genuine" only if it reflects the reviewer's honest, first-hand
          experience with THIS cosmetic product.
        - Promotional spam, competitor sabotage, inverted ratings, and off-topic reviews
          are NOT genuine.
        - is_genuine must be a JSON boolean (true/false), not a string.
    """).strip()

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text.strip()

    # Strip markdown fences if present (defensive)
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: parse manually
        result = {
            "category_label": group_name,
            "description":    "Unable to parse LLM response.",
            "is_genuine":     False,
            "reasoning":      raw[:200],
        }

    return result


def run_llm_analysis(df: pd.DataFrame) -> dict:
    """
    For each predicted group, sample reviews and run Claude analysis.
    Returns a dict keyed by group_name with Claude's structured output.
    """
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    groups = df["predicted_group"].unique()
    results = {}

    print(f"[LLM]  Analysing {len(groups)} groups with Claude…\n")

    for grp in sorted(groups):
        subset = df[df["predicted_group"] == grp].to_dict("records")
        analysis = analyse_group_with_llm(grp, subset, client)
        results[grp] = analysis
        label  = analysis.get("category_label", grp)
        genuine = analysis.get("is_genuine", False)
        print(f"       Group: {grp!r:30s} → {label!r:35s}  genuine={genuine}")

    print()
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 5.  GENUINE RATING COMPUTATION
#     Weighted average star rating using only reviews Claude marked as genuine.
#     Weight = log1p(helpful_votes + 1) × verified_purchase_bonus
# ─────────────────────────────────────────────────────────────────────────────

VERIFIED_BONUS = 1.5   # verified purchases count 1.5× more


def compute_genuine_rating(df: pd.DataFrame, llm_results: dict) -> float:
    """
    Compute a weighted average rating using only genuine reviews.

    Weight formula per review:
        w = (1 + log(helpful_votes + 1)) × (VERIFIED_BONUS if verified else 1.0)

    This upweights reviews that the community found helpful and that
    Amazon confirmed as verified purchases — both signals of authenticity.
    """
    genuine_groups = {
        grp for grp, info in llm_results.items() if info.get("is_genuine", False)
    }

    genuine_df = df[df["predicted_group"].isin(genuine_groups)].copy()

    if genuine_df.empty:
        print("[WARN]  No genuine reviews found — returning raw mean.")
        return df["star_rating"].mean()

    genuine_df["weight"] = (
        (1 + np.log1p(genuine_df["helpful_votes"]))
        * genuine_df["verified_purchase"].apply(lambda v: VERIFIED_BONUS if v else 1.0)
    )

    weighted_rating = (
        (genuine_df["star_rating"] * genuine_df["weight"]).sum()
        / genuine_df["weight"].sum()
    )

    print(f"[RATE]  Genuine reviews : {len(genuine_df):,} / {len(df):,} total")
    print(f"[RATE]  Raw avg rating  : {df['star_rating'].mean():.3f} ★")
    print(f"[RATE]  Genuine rating  : {weighted_rating:.3f} ★\n")

    return weighted_rating


# ─────────────────────────────────────────────────────────────────────────────
# 6.  REPORT GENERATION
#     Print a structured final report with all categories and the verdict.
# ─────────────────────────────────────────────────────────────────────────────

SEPARATOR = "─" * 72


def print_report(
    df: pd.DataFrame,
    llm_results: dict,
    genuine_rating: float,
) -> None:
    """
    Print a human-readable summary report of the analysis.
    """
    print(SEPARATOR)
    print("  AMAZON COSMETIC PRODUCT — REVIEW ANALYSIS REPORT")
    print(SEPARATOR)

    # Category table
    print("\n  DETECTED REVIEW CATEGORIES\n")
    header = f"  {'#':<4} {'Category Label':<35} {'Count':>6}  {'Genuine?':<10}  Description"
    print(header)
    print("  " + "-" * 100)

    for i, (grp, info) in enumerate(sorted(llm_results.items()), start=1):
        count   = len(df[df["predicted_group"] == grp])
        label   = info.get("category_label", grp)[:33]
        genuine = "✓ YES" if info.get("is_genuine") else "✗ NO"
        desc    = info.get("description", "")[:60]
        print(f"  {i:<4} {label:<35} {count:>6}  {genuine:<10}  {desc}")

    print()

    # Reasoning per category
    print("  CATEGORY REASONING\n")
    for grp, info in sorted(llm_results.items()):
        print(f"  [{info.get('category_label', grp)}]")
        reasoning = info.get("reasoning", "")
        for line in textwrap.wrap(reasoning, width=66):
            print(f"    {line}")
        print()

    # Final verdict
    print(SEPARATOR)
    print(f"  GENUINE RATING  :  {genuine_rating:.2f} ★ / 5.00")
    threshold = 4.0
    verdict   = "✅  GOOD PRODUCT" if genuine_rating >= threshold else "⚠️   BELOW THRESHOLD"
    print(f"  VERDICT         :  {verdict}  (threshold ≥ {threshold:.1f} ★)")
    print(SEPARATOR)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 7.  MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 72)
    print("  AMAZON REVIEW INTEGRITY ANALYSER — PIPELINE START")
    print("=" * 72 + "\n")

    # Step 1 — Simulate (or load) data
    print("[STEP 1]  Data Ingestion / Simulation")
    print(SEPARATOR)
    df = simulate_reviews(n_total=1100)

    # Step 2 — Feature engineering
    print("[STEP 2]  Feature Engineering")
    print(SEPARATOR)
    df = extract_features(df)

    # Step 3 — ML classification
    print("[STEP 3]  ML Classification (Random Forest)")
    print(SEPARATOR)
    df, clf, le, scaler = train_and_predict(df)

    # Step 4 — LLM analysis
    print("[STEP 4]  LLM Group Analysis (Claude)")
    print(SEPARATOR)
    llm_results = run_llm_analysis(df)

    # Step 5 — Genuine rating
    print("[STEP 5]  Genuine Rating Computation")
    print(SEPARATOR)
    genuine_rating = compute_genuine_rating(df, llm_results)

    # Step 6 — Report
    print("[STEP 6]  Final Report")
    print_report(df, llm_results, genuine_rating)


if __name__ == "__main__":
    main()
