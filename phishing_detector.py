import re

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from datasets import load_dataset
from scipy.sparse import hstack, csr_matrix

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


DATASET_NAME = "puyang2025/seven-phishing-email-datasets"


def count_urls(text):
    """Count URLs present in an email."""

    if not isinstance(text, str):
        return 0

    urls = re.findall(
        r"https?://\S+|www\.\S+",
        text,
        flags=re.IGNORECASE
    )

    return len(urls)


def load_email_data():
    """Load the public phishing email dataset."""

    print("Loading phishing email dataset...")

    dataset = load_dataset(DATASET_NAME)

    train_df = dataset["train"].to_pandas()
    test_df = dataset["test"].to_pandas()

    return train_df, test_df


def prepare_data(train_df, test_df):
    """Clean and prepare the dataset."""

    required_columns = ["text", "label"]

    for column in required_columns:
        if column not in train_df.columns:
            raise ValueError(
                f"Required column '{column}' was not found."
            )

    train_df = train_df.dropna(
        subset=["text", "label"]
    )

    test_df = test_df.dropna(
        subset=["text", "label"]
    )

    train_df["text"] = (
        train_df["text"]
        .astype(str)
        .str.strip()
    )

    test_df["text"] = (
        test_df["text"]
        .astype(str)
        .str.strip()
    )

    return train_df, test_df


def create_features(train_df, test_df):
    """Create TF-IDF text features and URL features."""

    print("\nExtracting text features...")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=15000
    )

    X_train_text = vectorizer.fit_transform(
        train_df["text"]
    )

    X_test_text = vectorizer.transform(
        test_df["text"]
    )

    print("Extracting URL features...")

    train_urls = train_df["text"].apply(
        count_urls
    )

    test_urls = test_df["text"].apply(
        count_urls
    )

    X_train_url = csr_matrix(
        train_urls.values.reshape(-1, 1)
    )

    X_test_url = csr_matrix(
        test_urls.values.reshape(-1, 1)
    )

    X_train = hstack([
        X_train_text,
        X_train_url
    ])

    X_test = hstack([
        X_test_text,
        X_test_url
    ])

    return X_train, X_test, vectorizer


def train_model(X_train, y_train):
    """Train the Logistic Regression classifier."""

    print("\nTraining machine learning model...")

    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(
        X_train,
        y_train
    )

    print("Training completed.")

    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model."""

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\n==============================")
    print("MODEL PERFORMANCE")
    print("==============================")

    print(
        f"Accuracy: {accuracy * 100:.2f}%"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Safe",
                "Phishing"
            ]
        )
    )

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    plt.figure(
        figsize=(7, 5)
    )

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[
            "Safe",
            "Phishing"
        ],
        yticklabels=[
            "Safe",
            "Phishing"
        ]
    )

    plt.title(
        "Phishing Email Detection - Confusion Matrix"
    )

    plt.xlabel(
        "Predicted Label"
    )

    plt.ylabel(
        "Actual Label"
    )

    plt.tight_layout()

    plt.savefig(
        "confusion_matrix.png",
        dpi=300
    )

    plt.show()

    return predictions


def analyze_email(
    model,
    vectorizer,
    email_text
):
    """Analyze a new email."""

    text_features = vectorizer.transform(
        [email_text]
    )

    url_count = count_urls(
        email_text
    )

    url_feature = csr_matrix(
        [[url_count]]
    )

    features = hstack([
        text_features,
        url_feature
    ])

    prediction = model.predict(
        features
    )[0]

    probabilities = model.predict_proba(
        features
    )[0]

    if prediction == 1:

        result = "PHISHING"
        confidence = probabilities[1]

    else:

        result = "SAFE"
        confidence = probabilities[0]

    print("\n==============================")
    print("EMAIL ANALYSIS")
    print("==============================")

    print(
        f"URLs detected: {url_count}"
    )

    print(
        f"Prediction: {result}"
    )

    print(
        f"Confidence: {confidence * 100:.2f}%"
    )


def main():

    print("===================================")
    print("   PHISHING EMAIL DETECTION MODEL")
    print("===================================")

    train_df, test_df = load_email_data()

    train_df, test_df = prepare_data(
        train_df,
        test_df
    )

    print(
        f"\nTraining emails: {len(train_df)}"
    )

    print(
        f"Testing emails: {len(test_df)}"
    )

    print("\nTraining label distribution:")

    print(
        train_df["label"].value_counts()
    )

    X_train, X_test, vectorizer = create_features(
        train_df,
        test_df
    )

    y_train = train_df["label"].astype(int)

    y_test = test_df["label"].astype(int)

    model = train_model(
        X_train,
        y_train
    )

    evaluate_model(
        model,
        X_test,
        y_test
    )

    print("\nEnter an email to analyze.")

    email_text = input(
        "\nEmail: "
    )

    analyze_email(
        model,
        vectorizer,
        email_text
    )


if __name__ == "__main__":
    main()
