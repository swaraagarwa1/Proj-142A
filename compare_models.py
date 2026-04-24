"""
Same train/test split as train_la_crime.py: compare LogisticRegression vs
HistGradientBoosting for eight-class and (optionally) coarse4 targets.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score
import warnings

from train_la_crime import DEFAULT_DATA, build_pipeline, make_train_test_with_collapse


def _fit_score(
    estimator: str,
    random_state: int,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    *,
    logreg_C: float,
    hgb_max_iter: int,
    hgb_max_depth: int,
    hgb_lr: float,
    hgb_l2: float,
) -> tuple[float, float, float, float]:
    pipe = build_pipeline(
        estimator,
        random_state,
        logreg_C=logreg_C,
        hgb_max_iter=hgb_max_iter,
        hgb_max_depth=hgb_max_depth,
        hgb_learning_rate=hgb_lr,
        hgb_l2=hgb_l2,
    )
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        pipe.fit(X_train, y_train)
    train_s = time.perf_counter() - t0
    y_pred = pipe.predict(X_test)
    acc = float((y_pred == y_test).mean())
    f1m = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1w = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    return acc, f1m, f1w, train_s


def main() -> None:
    p = argparse.ArgumentParser(
        description="LogReg vs HGB on the same 80/20 split (per target)."
    )
    p.add_argument("--data", type=str, default=str(DEFAULT_DATA))
    p.add_argument("--max-rows", type=int, default=200_000)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--logreg-C", type=float, default=0.1)
    p.add_argument("--hgb-max-iter", type=int, default=200)
    p.add_argument("--hgb-max-depth", type=int, default=16)
    p.add_argument("--hgb-lr", type=float, default=0.08)
    p.add_argument("--hgb-l2", type=float, default=0.0)
    p.add_argument("--top-k", type=int, default=15)
    p.add_argument("--premise-top-n", type=int, default=30)
    p.add_argument("--weapon-top-n", type=int, default=25)
    p.add_argument("--district-top-n", type=int, default=100)
    p.add_argument(
        "--targets",
        type=str,
        default="eight,coarse4",
        help="Comma-separated: eight, coarse4, fine (default: eight,coarse4).",
    )
    args = p.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")

    max_rows = None if args.max_rows == 0 else args.max_rows
    target_list = [t.strip() for t in args.targets.split(",") if t.strip()]

    rows: list[dict] = []
    for target in target_list:
        X_tr, X_te, y_tr, y_te, _, _ = make_train_test_with_collapse(
            data_path,
            max_rows,
            args.top_k,
            target,
            args.test_size,
            args.random_state,
            args.premise_top_n,
            args.weapon_top_n,
            args.district_top_n,
        )
        n = len(y_tr) + len(y_te)
        k = int(pd.concat([y_tr, y_te]).nunique())
        for est in ("logreg", "hgb"):
            acc, f1m, f1w, tsec = _fit_score(
                est,
                args.random_state,
                X_tr,
                X_te,
                y_tr,
                y_te,
                logreg_C=args.logreg_C,
                hgb_max_iter=args.hgb_max_iter,
                hgb_max_depth=args.hgb_max_depth,
                hgb_lr=args.hgb_lr,
                hgb_l2=args.hgb_l2,
            )
            rows.append(
                {
                    "target": target,
                    "estimator": est,
                    "n_rows": n,
                    "n_classes": k,
                    "acc": acc,
                    "f1_macro": f1m,
                    "f1_weighted": f1w,
                    "train_sec": tsec,
                }
            )

    df = pd.DataFrame(rows)
    print("\nSame random_state + test_size as train_la_crime.py; train is (1 - test_size).")
    print("\n" + df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
