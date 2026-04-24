"""
Hyperparameter search for LogisticRegression and RandomForest on the same
train/test split (with rare-category bucketing from train_la_crime).
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import RandomizedSearchCV

from train_la_crime import (
    DEFAULT_DATA,
    build_pipeline,
    make_train_test_with_collapse,
)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Random search over hyperparameters; report best CV + test accuracy."
    )
    p.add_argument("--data", type=str, default=str(DEFAULT_DATA))
    p.add_argument("--max-rows", type=int, default=100_000)
    p.add_argument("--target", choices=("eight", "fine"), default="eight")
    p.add_argument("--top-k", type=int, default=15)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--premise-top-n", type=int, default=30)
    p.add_argument("--weapon-top-n", type=int, default=25)
    p.add_argument("--district-top-n", type=int, default=100)
    p.add_argument("--n-iter", type=int, default=12, help="Random samples per model")
    p.add_argument("--cv", type=int, default=2, help="CV folds (2 is faster)")
    p.add_argument(
        "--n-iter-lr",
        type=int,
        default=None,
        help="Override n_iter for logistic regression (default: min(n-iter, 8))",
    )
    args = p.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    max_rows = None if args.max_rows == 0 else args.max_rows
    X_train, X_test, y_train, y_test, _, _ = make_train_test_with_collapse(
        data_path,
        max_rows,
        args.top_k,
        args.target,
        args.test_size,
        args.random_state,
        args.premise_top_n,
        args.weapon_top_n,
        args.district_top_n,
    )

    print(
        f"Data: {len(y_train) + len(y_test):,} rows | train {len(y_train):,} | test {len(y_test):,}\n"
    )

    n_iter_lr = args.n_iter_lr if args.n_iter_lr is not None else min(args.n_iter, 8)
    n_iter_rf = max(1, args.n_iter)

    # --- LogisticRegression ---
    pipe_lr = build_pipeline("logreg", args.random_state, logreg_C=0.5)
    param_lr = {
        "clf__C": [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        search_lr = RandomizedSearchCV(
            pipe_lr,
            param_distributions=param_lr,
            n_iter=min(n_iter_lr, len(param_lr["clf__C"])),
            cv=args.cv,
            scoring="accuracy",
            random_state=args.random_state,
            n_jobs=1,
            verbose=1,
        )
        print("Tuning LogisticRegression…")
        search_lr.fit(X_train, y_train)
    y_pred_lr = search_lr.predict(X_test)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    print("=== LogisticRegression (tuned) ===")
    print(f"Best params: {search_lr.best_params_}")
    print(f"Best CV accuracy (mean): {search_lr.best_score_:.4f}")
    print(f"Test accuracy: {acc_lr:.4f}\n")

    # --- RandomForest ---
    pipe_rf = build_pipeline(
        "rf",
        args.random_state,
        rf_n_estimators=200,
        rf_max_depth=28,
        rf_min_samples_leaf=1,
        rf_max_features="sqrt",
    )
    param_rf = {
        "clf__n_estimators": [100, 200, 300, 400],
        "clf__max_depth": [16, 24, 32, 40, None],
        "clf__min_samples_leaf": [1, 2, 4, 6],
        "clf__max_features": [
            "sqrt",
            0.25,
            0.35,
        ],
    }
    with warnings.catch_warnings():
        search_rf = RandomizedSearchCV(
            pipe_rf,
            param_distributions=param_rf,
            n_iter=n_iter_rf,
            cv=args.cv,
            scoring="accuracy",
            random_state=args.random_state + 1,
            n_jobs=1,
            verbose=1,
        )
        print("Tuning RandomForest…")
        search_rf.fit(X_train, y_train)
    y_pred_rf = search_rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    print("=== RandomForest (tuned) ===")
    print(f"Best params: {search_rf.best_params_}")
    print(f"Best CV accuracy (mean): {search_rf.best_score_:.4f}")
    print(f"Test accuracy: {acc_rf:.4f}\n")
    print(classification_report(y_test, y_pred_rf, zero_division=0))

    print("Summary (test accuracy):")
    print(f"  LogisticRegression: {acc_lr:.4f}")
    print(f"  Random Forest:     {acc_rf:.4f}")


if __name__ == "__main__":
    main()
