import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def build_pipeline() -> Pipeline:
    # Realistic baseline: use only body text and constrain vocabulary.
    # This avoids "shortcut" signals from sender/flags on this synthetic dataset.
    preprocessor = ColumnTransformer(
        transformers=[
            ("email_text_tfidf", TfidfVectorizer(max_features=2), "email_text"),
        ]
    )

    model = LogisticRegression(max_iter=1000, solver="saga", C=0.2)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train logistic regression phishing email classifier."
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to input dataset (.xlsx or .csv).",
    )
    parser.add_argument(
        "--model-out",
        type=str,
        default="phishing_logreg_model.joblib",
        help="Path to save trained model pipeline.",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    if data_path.suffix.lower() == ".xlsx":
        df = pd.read_excel(data_path)
    elif data_path.suffix.lower() == ".csv":
        df = pd.read_csv(data_path)
    else:
        raise ValueError("Unsupported file type. Use .xlsx or .csv")

    required_columns = [
        "email_text",
        "label",
    ]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = df[["email_text"]]
    y = df["label"].astype(str).str.lower()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=4, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(pipeline, args.model_out)
    print(f"\nSaved model pipeline to: {args.model_out}")


if __name__ == "__main__":
    main()
