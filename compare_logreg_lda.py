from __future__ import annotations

# Same train/test as train_la_crime.py: LogisticRegression vs LDA vs RandomForest.
# Writes two PNGs (coarse4 and eight). LDA and RF use dense one-hot (more RAM than logreg).
#
#   python compare_logreg_lda.py
#   python compare_logreg_lda.py --max-rows 80000   # if out of memory
#
# RF defaults per target match CV-accuracy tuning (~200k rows, tune_rf_la_crime.py --scoring accuracy).

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    precision_recall_fscore_support,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from train_la_crime import (
    DEFAULT_DATA,
    build_lda_pipeline,
    build_logreg_pipeline,
    build_rf_pipeline,
    make_train_test_with_collapse,
)


def _align_proba(pipe, X, class_order: np.ndarray) -> np.ndarray:
    """Reorder predict_proba columns to match class_order (canonical LR order)."""
    proba = pipe.predict_proba(X)
    cc = pipe.named_steps["clf"].classes_
    idx = np.array([np.where(cc == c)[0][0] for c in class_order])
    return proba[:, idx]


def _micro_average_roc(
    y_true,
    y_score: np.ndarray,
    classes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """One ROC curve: micro-averaged one-vs-rest across classes (multiclass standard)."""
    y_true = np.asarray(y_true)
    classes = np.asarray(classes)
    if len(classes) < 2:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), float("nan")
    Y = label_binarize(y_true, classes=list(classes))
    if Y.ndim == 1:
        Y = np.column_stack([1 - Y, Y])
    fpr, tpr, _ = roc_curve(Y.ravel(), y_score.ravel())
    return fpr, tpr, float(auc(fpr, tpr))


def _plot_block(ax_cm, ax_f1, y_test, y_pred, labels, title: str, acc: float) -> None:
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        labels=labels,
        normalize="true",
        ax=ax_cm,
        colorbar=True,
        values_format=".2f",
    )
    ax_cm.set_title(f"{title}\nacc={acc:.3f}")
    ax_cm.tick_params(axis="x", rotation=45)
    _, _, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=labels, zero_division=0
    )
    x = np.arange(len(labels))
    ax_f1.bar(x, f1, color="steelblue", edgecolor="navy", alpha=0.85)
    ax_f1.set_xticks(x)
    ax_f1.set_xticklabels(labels, rotation=45, ha="right")
    ax_f1.set_ylim(0, 1)
    ax_f1.set_ylabel("F1")
    ax_f1.set_title(f"{title} F1 (test)")


def compare_one_target(
    data_path: Path,
    max_rows: int | None,
    target: str,
    out_png: Path,
    *,
    top_k: int,
    test_size: float,
    random_state: int,
    premise_top_n: int,
    weapon_top_n: int,
    district_top_n: int,
    logreg_c: float,
    rf_n_estimators: int,
    rf_max_depth: int | None,
) -> None:
    X_train, X_test, y_train, y_test, _, _ = make_train_test_with_collapse(
        data_path,
        max_rows,
        top_k,
        target,
        test_size,
        random_state,
        premise_top_n,
        weapon_top_n,
        district_top_n,
    )
    n = len(y_train) + len(y_test)

    lr = build_logreg_pipeline(logreg_c=logreg_c, random_state=random_state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        lr.fit(X_train, y_train)
    pred_lr = lr.predict(X_test)
    acc_lr = accuracy_score(y_test, pred_lr)
    labels = lr.named_steps["clf"].classes_

    lda = build_lda_pipeline()
    lda.fit(X_train, y_train)
    pred_lda = lda.predict(X_test)
    acc_lda = accuracy_score(y_test, pred_lda)

    rf = build_rf_pipeline(
        random_state=random_state,
        n_estimators=rf_n_estimators,
        max_depth=rf_max_depth,
    )
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, pred_rf)

    proba_lr = _align_proba(lr, X_test, labels)
    proba_lda = _align_proba(lda, X_test, labels)
    proba_rf = _align_proba(rf, X_test, labels)

    print(f"\n--- target={target}  n={n:,} ---")
    print(f"LogisticRegression  test accuracy: {acc_lr:.4f}")
    print(f"LDA                 test accuracy: {acc_lda:.4f}")
    print(f"RandomForest        test accuracy: {acc_rf:.4f}")

    fig = plt.figure(figsize=(18, 13))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.05, 1.05, 1.0])
    axes_cm = [fig.add_subplot(gs[0, j]) for j in range(3)]
    axes_f1 = [fig.add_subplot(gs[1, j]) for j in range(3)]
    ax_roc = fig.add_subplot(gs[2, :])

    _plot_block(axes_cm[0], axes_f1[0], y_test, pred_lr, labels, "LogisticRegression", acc_lr)
    _plot_block(axes_cm[1], axes_f1[1], y_test, pred_lda, labels, "LDA", acc_lda)
    _plot_block(axes_cm[2], axes_f1[2], y_test, pred_rf, labels, "RandomForest", acc_rf)

    roc_styles = [
        ("LogisticRegression", proba_lr, "#1f77b4"),
        ("LDA", proba_lda, "#ff7f0e"),
        ("RandomForest", proba_rf, "#2ca02c"),
    ]
    for name, P, color in roc_styles:
        fpr, tpr, roc_auc = _micro_average_roc(y_test, P, labels)
        ax_roc.plot(fpr, tpr, lw=2, color=color, label=f"{name} (micro-AUC={roc_auc:.3f})")
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.45, label="Chance")
    ax_roc.set_xlim(0.0, 1.0)
    ax_roc.set_ylim(0.0, 1.05)
    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("True positive rate")
    ax_roc.set_title(
        "Micro-averaged ROC (one-vs-rest, test set)\n"
        "Aggregates all classes into pooled positives vs negatives."
    )
    ax_roc.legend(loc="lower right", fontsize=9)
    ax_roc.set_aspect("equal")

    fig.suptitle(
        f"LA crime: LogReg vs LDA vs RF  |  target={target}  |  n={n:,}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_png.resolve()}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Compare LogisticRegression, LDA, RandomForest; save 2 PNGs."
    )
    p.add_argument("--data", type=str, default=str(DEFAULT_DATA))
    p.add_argument(
        "--max-rows",
        type=int,
        default=200_000,
        help="Row cap; lower if dense models run out of memory.",
    )
    p.add_argument("--top-k", type=int, default=15)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--premise-top-n", type=int, default=30)
    p.add_argument("--weapon-top-n", type=int, default=25)
    p.add_argument("--district-top-n", type=int, default=100)
    p.add_argument(
        "--logreg-c", "--logreg-C", type=float, default=0.1, dest="logreg_c"
    )
    p.add_argument(
        "--rf-n-estimators-coarse4",
        type=int,
        default=300,
        help="RF trees for coarse4 figure (default: accuracy-CV tuned).",
    )
    p.add_argument(
        "--rf-max-depth-coarse4",
        type=int,
        default=32,
        help="RF max_depth coarse4; use 0 for None.",
    )
    p.add_argument(
        "--rf-n-estimators-eight",
        type=int,
        default=300,
        help="RF trees for eight-class figure (default: accuracy-CV tuned).",
    )
    p.add_argument(
        "--rf-max-depth-eight",
        type=int,
        default=28,
        help="RF max_depth eight; use 0 for None.",
    )
    p.add_argument(
        "--out-coarse4",
        type=str,
        default="la_crime_compare_coarse4_logreg_lda_rf.png",
    )
    p.add_argument(
        "--out-eight",
        type=str,
        default="la_crime_compare_eight_logreg_lda_rf.png",
    )
    args = p.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    max_rows = None if args.max_rows == 0 else args.max_rows
    base_kw = dict(
        top_k=args.top_k,
        test_size=args.test_size,
        random_state=args.random_state,
        premise_top_n=args.premise_top_n,
        weapon_top_n=args.weapon_top_n,
        district_top_n=args.district_top_n,
        logreg_c=args.logreg_c,
    )
    depth_c4 = (
        None if args.rf_max_depth_coarse4 == 0 else args.rf_max_depth_coarse4
    )
    depth_8 = None if args.rf_max_depth_eight == 0 else args.rf_max_depth_eight

    compare_one_target(
        data_path,
        max_rows,
        "coarse4",
        Path(args.out_coarse4),
        **base_kw,
        rf_n_estimators=args.rf_n_estimators_coarse4,
        rf_max_depth=depth_c4,
    )
    compare_one_target(
        data_path,
        max_rows,
        "eight",
        Path(args.out_eight),
        **base_kw,
        rf_n_estimators=args.rf_n_estimators_eight,
        rf_max_depth=depth_8,
    )


if __name__ == "__main__":
    main()
