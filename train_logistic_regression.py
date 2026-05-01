# Phishing: TF-IDF on email body + LogisticRegression (separate from LA crime).

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
    # TF-IDF on email text only, small vocabulary for this demo dataset
    pre = ColumnTransformer(
        [("tfidf", TfidfVectorizer(max_features=2), "email_text")],
    )
    clf = LogisticRegression(max_iter=1000, solver="saga", C=0.2)
    return Pipeline([("pre", pre), ("clf", clf)])


def main():
    p = argparse.ArgumentParser(description="phishing email: TF-IDF + logreg")
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--model-out", type=str, default="phishing_logreg_model.joblib")
    args = p.parse_args()

    path = Path(args.data)
    if not path.exists():
        raise FileNotFoundError(path)

    # load data
    if path.suffix.lower() == ".xlsx":
        df = pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError("use .csv or .xlsx")
    for col in ("email_text", "label"):
        if col not in df.columns:
            raise ValueError(f"need column: {col}")

    # x and y
    X = df[["email_text"]]
    y = df["label"].astype(str).str.lower()

    # 80/20 train test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=4, stratify=y
    )

    # model
    pl = build_pipeline()
    pl.fit(X_train, y_train)
    y_pred = pl.predict(X_test)
    print("\nclassification report:")
    print(classification_report(y_test, y_pred))
    print("confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(pl, args.model_out)
    print(f"\nsaved: {args.model_out}")


if __name__ == "__main__":
    main()
