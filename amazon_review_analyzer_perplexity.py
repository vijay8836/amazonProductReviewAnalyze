import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from textblob import TextBlob
import nltk
from collections import Counter
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class AmazonReviewAnalyzer:
    """Analyze cosmetic-product reviews and estimate a cleaned rating.

    Important: This is a screening model, not proof that a review is fake
    or that a reviewer is a competitor. Those conclusions require evidence
    unavailable in ordinary review text.
    """

    def __init__(self, n_clusters=5, anomaly_contamination=0.15):
        self.n_clusters = n_clusters
        self.anomaly_contamination = anomaly_contamination
        self.df = None
        self.categories = {}
        self.genuine_rating = None
        self.vectorizer = None
        self.model = None

    def load_csv(self, csv_file, review_col="reviewText", rating_col="overall",
                 verified_col="verified_purchase"):
        """Load a CSV containing review text and ratings."""
        df = pd.read_csv(csv_file)
        required = {review_col, rating_col}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")

        rename = {review_col: "review", rating_col: "rating"}
        if verified_col in df.columns:
            rename[verified_col] = "verified_purchase"
        self.df = df.rename(columns=rename).copy()
        self.df["review"] = self.df["review"].fillna("").astype(str)
        self.df["rating"] = pd.to_numeric(self.df["rating"], errors="coerce")
        self.df = self.df.dropna(subset=["rating"])
        self.df = self.df[self.df["rating"].between(1, 5)].reset_index(drop=True)
        if "verified_purchase" not in self.df:
            self.df["verified_purchase"] = np.nan
        return self.df

    def load_sample_data(self, n_reviews=1200, seed=42):
        """Create demo data. Replace this with load_csv for real data."""
        rng = np.random.default_rng(seed)
        positive = [
            "Love this lipstick, perfect color and lasts all day",
            "Amazing moisturizer, my skin feels soft and hydrated",
            "Great foundation, matches my skin tone perfectly",
            "Best mascara, no smudging and easy to remove",
        ]
        negative = [
            "The product irritated my skin and the color was wrong",
            "Poor quality and it did not last as advertised",
            "The packaging arrived damaged and the product was dry",
        ]
        fake = [
            "BEST PRODUCT EVER 5 STARS BUY NOW",
            "Perfect amazing love it highly recommended",
            "Number one product in the world absolutely perfect",
        ]
        competitor = [
            "Worst product ever do not waste your money",
            "Terrible fake product and complete scam",
            "Awful quality nothing works avoid this seller",
        ]
        wrong_product = [
            "Great book, could not put it down",
            "Excellent novel, highly recommend this story",
            "Perfect read for a book club discussion",
        ]
        mistaken = [
            "Terrible product 5 stars",
            "I love it 1 star",
        ]

        reviews, ratings = [], []
        for _ in range(n_reviews):
            x = rng.random()
            if x < 0.10:
                reviews.append(rng.choice(fake)); ratings.append(5)
            elif x < 0.18:
                reviews.append(rng.choice(competitor)); ratings.append(1)
            elif x < 0.23:
                reviews.append(rng.choice(wrong_product)); ratings.append(int(rng.integers(1, 6)))
            elif x < 0.30:
                reviews.append(rng.choice(mistaken)); ratings.append(int(rng.choice([1, 5])))
            elif rng.random() < 0.70:
                reviews.append(rng.choice(positive)); ratings.append(int(rng.choice([4, 5], p=[.45, .55])))
            else:
                reviews.append(rng.choice(negative)); ratings.append(int(rng.choice([1, 2, 3], p=[.25, .50, .25])))

        self.df = pd.DataFrame({
            "review": reviews,
            "rating": ratings,
            "verified_purchase": rng.choice([True, False], n_reviews, p=[.85, .15]),
        })
        return self.df

    @staticmethod
    def preprocess_text(text):
        text = re.sub(r"[^a-zA-Z\s]", " ", str(text).lower())
        tokens = word_tokenize(text)
        stops = set(stopwords.words("english"))
        return " ".join(t for t in tokens if t not in stops and len(t) > 2)

    @staticmethod
    def _contains_any(text, words):
        return any(re.search(rf"\b{re.escape(w)}\b", text) for w in words)

    def analyze(self):
        if self.df is None or self.df.empty:
            raise ValueError("Load data before analysis")

        self.df["clean_text"] = self.df["review"].map(self.preprocess_text)
        self.df["sentiment"] = self.df["review"].map(
            lambda x: TextBlob(str(x)).sentiment.polarity
        )
        self.df["text_length"] = self.df["review"].str.len()
        self.df["exclamation_count"] = self.df["review"].str.count("!")

        self.vectorizer = TfidfVectorizer(
            max_features=3000, stop_words="english", ngram_range=(1, 2),
            min_df=2
        )
        X = self.vectorizer.fit_transform(self.df["clean_text"])

        k = min(self.n_clusters, max(2, len(self.df) // 50))
        self.model = KMeans(n_clusters=k, random_state=42, n_init=10)
        self.df["cluster"] = self.model.fit_predict(X)

        iso = IsolationForest(
            contamination=self.anomaly_contamination, random_state=42
        )
        self.df["anomaly_score"] = -iso.fit_predict(X)
        self.df["is_anomaly"] = self.df["anomaly_score"].eq(1)

        # Sentiment/rating consistency: neutral text is not automatically bad.
        self.df["rating_text_mismatch"] = (
            ((self.df["rating"] >= 4) & (self.df["sentiment"] < -0.15)) |
            ((self.df["rating"] <= 2) & (self.df["sentiment"] > 0.15))
        )
        self.df["wrong_product"] = self.df["clean_text"].map(
            lambda x: self._contains_any(x, ["book", "novel", "chapter", "read", "story", "author"])
        )
        self.df["promotional_pattern"] = self.df["review"].str.lower().map(
            lambda x: (
                self._contains_any(x, ["buy now", "best product ever", "number one", "highly recommended"]) or
                x.count("!") >= 4 or len(x) > 250 and x.upper() == x
            )
        )
        self.df["category"] = self.df.apply(self._assign_category, axis=1)
        self._build_cluster_report(X)
        self._calculate_genuine_rating()
        return self.df

    def _assign_category(self, row):
        if row["wrong_product"]:
            return "Wrong-product review"
        if row["rating_text_mismatch"]:
            return "Rating-text mismatch / possible rating mistake"
        if row["promotional_pattern"] and row["rating"] >= 4:
            return "Promotional or low-information review"
        if row["is_anomaly"] and row["rating"] >= 4:
            return "Anomalous positive review"
        if row["is_anomaly"] and row["rating"] <= 2:
            return "Anomalous negative review"
        if row["rating"] >= 4:
            return "Likely genuine positive"
        if row["rating"] <= 2:
            return "Likely genuine negative"
        return "Likely genuine mixed/neutral"

    def _build_cluster_report(self, X):
        names = self.vectorizer.get_feature_names_out()
        report = {}
        for cluster in sorted(self.df["cluster"].unique()):
            mask = self.df["cluster"].eq(cluster)
            mean_values = X[mask].mean(axis=0).A1
            top_words = [names[i] for i in mean_values.argsort()[-10:][::-1]]
            part = self.df.loc[mask]
            report[int(cluster)] = {
                "size": int(mask.sum()),
                "average_rating": round(float(part["rating"].mean()), 3),
                "top_words": top_words,
                "categories": part["category"].value_counts().to_dict(),
            }
        self.categories = report

    def _calculate_genuine_rating(self):
        # Exclude clear text mismatches, wrong-product reviews, and low-information
        # promotional reviews. Anomaly flags are treated as evidence, not proof.
        excluded = (
            self.df["wrong_product"] |
            self.df["rating_text_mismatch"] |
            self.df["promotional_pattern"]
        )
        included = self.df.loc[~excluded, "rating"]
        if len(included) < max(30, int(len(self.df) * 0.10)):
            included = self.df.loc[~self.df["wrong_product"], "rating"]
        self.genuine_rating = float(included.mean())
        self.df["used_for_genuine_rating"] = ~excluded
        return self.genuine_rating

    def report(self):
        if self.genuine_rating is None:
            raise ValueError("Run analyze() first")
        print("\nAMAZON COSMETIC REVIEW ANALYSIS")
        print("=" * 45)
        print(f"Reviews analyzed: {len(self.df):,}")
        print(f"Observed average: {self.df['rating'].mean():.2f}/5")
        print(f"Estimated genuine rating: {self.genuine_rating:.2f}/5")
        print("Decision:", "GOOD PRODUCT" if self.genuine_rating > 4 else "CAUTION")
        print("\nDetected categories:")
        counts = self.df["category"].value_counts()
        for category, count in counts.items():
            print(f"- {category}: {count:,} ({count / len(self.df):.1%})")
        print("\nNote: Competitor sabotage cannot be reliably proven from review text alone.")
        print("Treat anomalous negative reviews as a risk flag, not a factual accusation.")

    def save_outputs(self, prefix="review_analysis"):
        self.df.to_csv(f"{prefix}_classified_reviews.csv", index=False)
        pd.DataFrame(self.categories).T.to_csv(f"{prefix}_clusters.csv")

        plt.figure(figsize=(8, 5))
        self.df["rating"].value_counts().sort_index().plot(kind="bar", color="steelblue")
        plt.axvline(self.genuine_rating - 1, color="red", linestyle="--",
                    label=f"Genuine rating: {self.genuine_rating:.2f}")
        plt.title("Observed Rating Distribution")
        plt.xlabel("Rating")
        plt.ylabel("Number of reviews")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{prefix}_rating_distribution.png", dpi=200)
        plt.close()


def main():
    analyzer = AmazonReviewAnalyzer(n_clusters=5, anomaly_contamination=0.15)

    # Demo mode. For real data, replace this line with:
    # analyzer.load_csv("amazon_reviews.csv", review_col="reviewText", rating_col="overall")
    analyzer.load_sample_data(n_reviews=1200)

    analyzer.analyze()
    analyzer.report()
    analyzer.save_outputs()


if __name__ == "__main__":
    main()
