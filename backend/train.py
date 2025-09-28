# backend/train.py
import argparse
import os
import re
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
import joblib

def parse_hashtags(s):
    return [t.lstrip("#").lower() for t in re.findall(r"#?\w+", str(s))]

def clean_labels(df, min_count=3, max_frac=0.95):
    """Remove hashtags that are too rare or too common"""
    tag_counts = df["tags"].explode().value_counts()
    rare_tags = tag_counts[tag_counts < min_count].index
    common_tags = tag_counts[tag_counts > max_frac * len(df)].index

    df["tags"] = df["tags"].apply(
        lambda tags: [t for t in tags if t not in rare_tags and t not in common_tags]
    )
    df = df[df["tags"].map(len) > 0]  # drop rows with no tags left
    return df

def main(args):
    print("Loading dataset...")
    df = pd.read_csv(args.data)
    df["tags"] = df["Hashtags"].apply(parse_hashtags)
    df = clean_labels(df)

    print(f"Dataset size after cleaning: {len(df)} rows")
    print(f"Unique hashtags: {len(set([t for tags in df['tags'] for t in tags]))}")

    # Label binarization
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(df["tags"])

    # Embeddings
    print("Loading SentenceTransformer model...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    print("Encoding texts into embeddings...")
    X = embedder.encode(df["Text"].astype(str).tolist(), show_progress_bar=True)

    # Train/test split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    # Train classifier
    print("Training classifier...")
    clf = OneVsRestClassifier(LogisticRegression(max_iter=200))
    clf.fit(X_train, Y_train)

    # Evaluate
    Y_pred = clf.predict(X_test)
    print("Evaluation Results:")
    print("F1 (micro):", f1_score(Y_test, Y_pred, average="micro"))
    print("F1 (macro):", f1_score(Y_test, Y_pred, average="macro"))
    print("\nClassification Report:\n", classification_report(Y_test, Y_pred, target_names=mlb.classes_))

    # Save
    os.makedirs(os.path.dirname(args.model), exist_ok=True)
    joblib.dump({"embedder": embedder, "clf": clf, "mlb": mlb}, args.model)
    print(f"Saved model to {args.model}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to CSV dataset")
    parser.add_argument("--model", required=True, help="Path to save model")
    args = parser.parse_args()
    main(args)
