# Lifecycle stage 6 — Model Training

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


df = pd.read_csv(
    "data/processed/cleaned_reviews.csv"
)

# Remove rows with missing review or sentiment
df = df.dropna(
    subset=["clean_review", "sentiment"]
)

# Make sure review text is string
df["clean_review"] = df["clean_review"].astype(str)

X = df["clean_review"]
y = df["sentiment"]


vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000
)

model.fit(
    X_train,
    y_train
)


joblib.dump(
    model,
    "models/sentiment_model.pkl"
)

joblib.dump(
    vectorizer,
    "models/tfidf_vectorizer.pkl"
)

print("Training Completed")
print("Total records:", len(df))
print("Training records:", X_train.shape[0])
print("Testing records:", X_test.shape[0])