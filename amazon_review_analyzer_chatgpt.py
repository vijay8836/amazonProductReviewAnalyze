"""
Amazon Product Review Quality Analyzer

Goal:
    Estimate a "genuine product rating" by identifying suspicious,
    contradictory, irrelevant and potentially duplicated reviews.

Input:
    amazon_reviews.csv

Expected columns:
    review_id, reviewer_id, rating, review_title, review_text,
    verified_purchase, helpful_votes, review_date
"""

import re
import warnings
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")

INPUT_FILE = "amazon_reviews.csv"
PRODUCT_DESCRIPTION = """
Cosmetic skincare product. A cosmetic product intended for personal
skin care and beauty use. Reviews should discuss the product itself,
including effectiveness, skin feel, texture, smell, ingredients,
irritation, packaging, results and value.
"""
GOOD_PRODUCT_THRESHOLD = 4.0
MIN_REVIEW_WEIGHT = 0.05


def load_data(path):
    df = pd.read_csv(path)
    for col in ["rating", "review_text"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    defaults = {
        "review_id": np.arange(len(df)),
        "reviewer_id": "unknown",
        "review_title": "",
        "verified_purchase": False,
        "helpful_votes": 0,
        "review_date": None,
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
    return df


def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def create_basic_features(df):
    df["review_text"] = df["review_text"].fillna("").astype(str)
    df["review_title"] = df["review_title"].fillna("").astype(str)
    df["clean_text"] = (
        df["review_title"] + " " + df["review_text"]
    ).apply(clean_text)
    df["text_length"] = df["clean_text"].str.len()
    df["word_count"] = df["clean_text"].str.split().str.len()
    df["exclamation_count"] = df["review_text"].str.count("!")
    df["question_count"] = df["review_text"].str.count(r"\?")
    return df


def calculate_sentiment(df):
    positive_words = {
        "good", "great", "excellent", "amazing", "love", "loved",
        "perfect", "wonderful", "awesome", "fantastic", "best",
        "nice", "soft", "smooth", "effective", "beautiful"
    }
    negative_words = {
        "bad", "terrible", "horrible", "hate", "hated", "worst",
        "awful", "poor", "waste", "irritation", "irritated",
        "burning", "broke", "useless", "disappointed"
    }

    def sentiment(text):
        words = text.split()
        if not words:
            return 0.0
        positive = sum(w in positive_words for w in words)
        negative = sum(w in negative_words for w in words)
        return (positive - negative) / max(len(words), 1)

    df["sentiment_score"] = df["clean_text"].apply(sentiment)
    return df


def calculate_rating_sentiment_consistency(df):
    def contradiction(row):
        s = row["sentiment_score"]
        r = row["rating"]
        if s > 0.04 and r <= 2:
            return 1.0
        if s < -0.04 and r >= 4:
            return 1.0
        return 0.0

    df["rating_text_contradiction"] = df.apply(contradiction, axis=1)
    return df


def detect_duplicate_reviews(df):
    texts = df["clean_text"].tolist()
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(texts)
    similarity = cosine_similarity(matrix)
    duplicate_count = np.zeros(len(df))

    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            if similarity[i, j] >= 0.90:
                duplicate_count[i] += 1
                duplicate_count[j] += 1

    df["duplicate_review_count"] = duplicate_count
    df["possible_duplicate"] = (duplicate_count > 0).astype(int)
    return df


def detect_generic_reviews(df):
    generic_phrases = [
        "good product", "great product", "nice product",
        "amazing product", "excellent product", "love it",
        "highly recommend", "very good", "very nice",
        "awesome", "good", "great", "excellent"
    ]

    def generic(text):
        t = text.lower()
        if t.strip() in generic_phrases:
            return 1
        return int(len(t.split()) <= 3)

    df["generic_review"] = df["clean_text"].apply(generic)
    return df


def calculate_product_relevance(df):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    product_embedding = model.encode(
        PRODUCT_DESCRIPTION, normalize_embeddings=True
    )
    review_embeddings = model.encode(
        df["clean_text"].tolist(),
        normalize_embeddings=True,
        show_progress_bar=True
    )
    similarity = np.dot(review_embeddings, product_embedding)
    df["product_relevance"] = similarity
    df["possible_irrelevant_review"] = (similarity < 0.25).astype(int)
    return df, model


def classify_review_topic(df):
    categories = {
        "product_quality": [
            "quality", "effective", "works", "result", "skin",
            "cream", "serum", "texture", "smell", "scent"
        ],
        "shipping_delivery": [
            "shipping", "delivery", "delivered", "package arrived",
            "late delivery"
        ],
        "seller_customer_service": [
            "seller", "customer service", "refund", "return",
            "customer support"
        ],
        "packaging": [
            "package", "packaging", "bottle", "container",
            "leaking", "seal"
        ],
        "wrong_product": [
            "book", "novel", "movie", "laptop", "phone",
            "keyboard", "shoes", "shirt", "restaurant", "hotel"
        ],
    }

    def classify(text):
        scores = {
            category: sum(keyword in text for keyword in keywords)
            for category, keywords in categories.items()
        }
        best = max(scores, key=scores.get)
        return best if scores[best] else "unknown"

    df["review_topic"] = df["clean_text"].apply(classify)
    df["wrong_product_review"] = (
        df["review_topic"] == "wrong_product"
    ).astype(int)
    return df


def calculate_reviewer_features(df):
    stats = (
        df.groupby("reviewer_id")
        .agg(
            review_count=("review_id", "count"),
            average_rating=("rating", "mean"),
            rating_std=("rating", "std"),
        )
        .reset_index()
    )
    df = df.merge(stats, on="reviewer_id", how="left")
    df["high_volume_reviewer"] = (df["review_count"] >= 20).astype(int)
    return df


def detect_anomalies(df):
    feature_columns = [
        "rating", "word_count", "text_length", "sentiment_score",
        "exclamation_count", "duplicate_review_count", "generic_review",
        "rating_text_contradiction", "product_relevance", "helpful_votes"
    ]
    X = df[feature_columns].fillna(0)
    X_scaled = StandardScaler().fit_transform(X)

    model = IsolationForest(
        n_estimators=300, contamination=0.10, random_state=42
    )
    predictions = model.fit_predict(X_scaled)
    df["anomaly_prediction"] = predictions
    df["anomaly_score"] = model.decision_function(X_scaled)
    df["possible_anomaly"] = (predictions == -1).astype(int)
    return df


def calculate_trust_score(df):
    score = np.ones(len(df))

    verified = (
        df["verified_purchase"].astype(str).str.lower()
        .isin(["true", "1", "yes"])
    )
    score += verified.astype(float) * 0.15

    helpful = np.log1p(
        pd.to_numeric(df["helpful_votes"], errors="coerce").fillna(0)
    )
    helpful = helpful / (helpful.max() + 1e-8)
    score += helpful * 0.10

    score -= df["rating_text_contradiction"] * 0.25
    score -= df["possible_duplicate"] * 0.25
    score -= df["generic_review"] * 0.10
    score -= df["possible_irrelevant_review"] * 0.30
    score -= df["wrong_product_review"] * 0.50
    score -= df["possible_anomaly"] * 0.20

    score = np.clip(score, 0, 1.5) / 1.5
    df["trust_score"] = score
    return df


def assign_final_category(row):
    if row["wrong_product_review"]:
        return "WRONG_PRODUCT"
    if row["possible_irrelevant_review"]:
        return "IRRELEVANT"
    if row["rating_text_contradiction"]:
        return (
            "RATING_TEXT_CONTRADICTION_POSITIVE"
            if row["rating"] >= 4
            else "RATING_TEXT_CONTRADICTION_NEGATIVE"
        )
    if row["possible_duplicate"]:
        return "POSSIBLE_DUPLICATE"
    if row["generic_review"]:
        return "GENERIC_REVIEW"
    if row["possible_anomaly"]:
        return "POSSIBLE_SUSPICIOUS_REVIEW"
    if row["sentiment_score"] > 0.02 and row["rating"] >= 4:
        return "LIKELY_GENUINE_POSITIVE"
    if row["sentiment_score"] < -0.02 and row["rating"] <= 2:
        return "LIKELY_GENUINE_NEGATIVE"
    return "NORMAL_REVIEW"


def calculate_genuine_rating(df):
    valid = df[df["trust_score"] >= MIN_REVIEW_WEIGHT].copy()
    total_weight = valid["trust_score"].sum()
    if total_weight == 0:
        return np.nan
    return (valid["rating"] * valid["trust_score"]).sum() / total_weight


def generate_category_report(df):
    report = (
        df.groupby("final_category")
        .agg(
            review_count=("review_id", "count"),
            average_rating=("rating", "mean"),
            average_trust=("trust_score", "mean"),
        )
        .sort_values("review_count", ascending=False)
    )
    report["percentage"] = report["review_count"] / len(df) * 100
    return report


def main():
    print("=" * 70)
    print("AMAZON REVIEW QUALITY ANALYZER")
    print("=" * 70)

    df = load_data(INPUT_FILE)
    print(f"\nNumber of reviews: {len(df)}")
    print(f"Original average rating: {df['rating'].mean():.3f}")

    df = create_basic_features(df)
    df = calculate_sentiment(df)
    df = calculate_rating_sentiment_consistency(df)
    df = detect_duplicate_reviews(df)
    df = detect_generic_reviews(df)
    df, _ = calculate_product_relevance(df)
    df = classify_review_topic(df)
    df = calculate_reviewer_features(df)
    df = detect_anomalies(df)
    df = calculate_trust_score(df)
    df["final_category"] = df.apply(assign_final_category, axis=1)

    genuine_rating = calculate_genuine_rating(df)
    category_report = generate_category_report(df)

    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)
    print(f"\nOriginal rating: {df['rating'].mean():.2f}")
    print(f"Genuine-weighted rating estimate: {genuine_rating:.2f}")

    if genuine_rating > GOOD_PRODUCT_THRESHOLD:
        print("Product classification: GOOD PRODUCT")
    else:
        print("Product classification: NOT ABOVE 4.0 THRESHOLD")

    print("\nREVIEW CATEGORIES")
    print(category_report.to_string())

    output_columns = [
        "review_id", "reviewer_id", "rating", "review_text",
        "sentiment_score", "rating_text_contradiction",
        "duplicate_review_count", "generic_review",
        "product_relevance", "wrong_product_review",
        "possible_irrelevant_review", "possible_anomaly",
        "trust_score", "final_category"
    ]

    df[output_columns].to_csv("review_analysis.csv", index=False)
    category_report.to_csv("review_category_report.csv")

    print("\nSaved:")
    print("  review_analysis.csv")
    print("  review_category_report.csv")


if __name__ == "__main__":
    main()
