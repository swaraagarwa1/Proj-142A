"""
Train/test the LA crime pipeline (same logic as train_la_crime.py), print metrics,
and save a figure: confusion matrix + per-class F1 bar chart.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from train_la_crime import (
    DEFAULT_DATA,
    build_pipeline,
    make_train_test_with_collapse,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate LA crime model and save plots.")
    p.add_argument("--data", type=str, default=str(DEFAULT_DATA))
    p.add_argument("--max-rows", type=int, default=200_000)
    p.add_argument("--target", choices=("eight", "fine"), default="eight")
    p.add_argument("--top-k", type=int, default=15)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--estimator", choices=("logreg", "rf"), default="logreg")
    p.add_argument(
        "--plot-out",
        type=str,
        default="la_crime_evaluation.png",
        help="Path for saved figure (PNG).",
    )
    p.add_argument("--premise-top-n", type=int, default=30)
    p.add_argument("--weapon-top-n", type=int, default=25)
    p.add_argument("--district-top-n", type=int, default=100)
    p.add_argument("--logreg-C", type=float, default=0.1)
    p.add_argument("--rf-n-estimators", type=int, default=300)
    p.add_argument(
        "--rf-max-depth", type=int, default=24, help="0 = unlimited (None)"
    )
    p.add_argument("--rf-min-samples-leaf", type=int, default=1)
    p.add_argument("--rf-max-features", type=str, default="0.35")
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
    n_rows = len(y_train) + len(y_test)

    try:
        rff = float(args.rf_max_features)
    except ValueError:
        rff = args.rf_max_features
    rf_depth = None if args.rf_max_depth == 0 else int(args.rf_max_depth)

    pipe = build_pipeline(
        args.estimator,
        args.random_state,
        logreg_C=args.logreg_C,
        rf_n_estimators=args.rf_n_estimators,
        rf_max_depth=rf_depth,
        rf_min_samples_leaf=args.rf_min_samples_leaf,
        rf_max_features=rff,
    )
    print(f"Fitting {args.estimator} on {len(X_train):,} training rows…", flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        pipe.fit(X_train, y_train)
    print("Done fitting.", flush=True)

    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Estimator: {args.estimator}")
    print(f"Test rows: {len(y_test)} | Test accuracy: {acc:.4f}\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    classes = pipe.named_steps["clf"].classes_
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        labels=classes,
        normalize="true",
        ax=axes[0],
        colorbar=True,
        values_format=".2f",
    )
    axes[0].set_title("Confusion matrix (normalized by true label)")
    axes[0].tick_params(axis="x", rotation=45)

    _, _, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=classes, zero_division=0
    )
    x = np.arange(len(classes))
    axes[1].bar(x, f1, color="steelblue", edgecolor="navy", alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(classes, rotation=45, ha="right")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("F1 score")
    axes[1].set_title("Per-class F1 (test set)")
    axes[1].axhline(acc, color="gray", linestyle="--", linewidth=1, label=f"Accuracy {acc:.2f}")
    axes[1].legend(loc="upper right")

    fig.suptitle(
        f"LA crime type model — {args.estimator} — {n_rows:,} rows (subset)",
        fontsize=12,
    )
    fig.tight_layout()
    out = Path(args.plot_out)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {out.resolve()}")


if __name__ == "__main__":
    main()
