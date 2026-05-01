# Same data + model as train_la_crime.py, then confusion matrix + F1 bar chart PNG.

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
    build_logreg_pipeline,
    make_train_test_with_collapse,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default=str(DEFAULT_DATA))
    p.add_argument("--max-rows", type=int, default=200_000)
    p.add_argument(
        "--target", choices=("coarse4", "eight", "fine"), default="coarse4"
    )
    p.add_argument("--top-k", type=int, default=15)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--plot-out", type=str, default="la_crime_evaluation.png")
    p.add_argument("--premise-top-n", type=int, default=30)
    p.add_argument("--weapon-top-n", type=int, default=25)
    p.add_argument("--district-top-n", type=int, default=100)
    p.add_argument(
        "--logreg-c", "--logreg-C", type=float, default=0.1, dest="logreg_c"
    )
    args = p.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    # load + train/test + rare categories (same as training script)
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
    n = len(y_train) + len(y_test)

    # fit logreg pipeline
    model = build_logreg_pipeline(
        logreg_c=args.logreg_c, random_state=args.random_state
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"test accuracy: {acc:.4f}\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    # two plots: confusion matrix, per-class F1
    classes = model.named_steps["clf"].classes_
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, labels=classes, normalize="true", ax=axes[0],
        colorbar=True, values_format=".2f",
    )
    axes[0].set_title("confusion (row = true class)")
    axes[0].tick_params(axis="x", rotation=45)
    _, _, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=classes, zero_division=0
    )
    x = np.arange(len(classes))
    axes[1].bar(x, f1, color="steelblue", edgecolor="navy", alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(classes, rotation=45, ha="right")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("F1")
    axes[1].set_title("F1 by class (test)")
    axes[1].axhline(
        acc, color="gray", linestyle="--", label=f"accuracy {acc:.2f}"
    )
    axes[1].legend()
    fig.suptitle(
        f"LA crime logreg  target={args.target}  n={n:,} rows", fontsize=12
    )
    fig.tight_layout()
    out = Path(args.plot_out)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out.resolve()}")


if __name__ == "__main__":
    main()
